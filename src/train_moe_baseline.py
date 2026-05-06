import argparse
import contextlib
import os
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from deep_ep_comm import DeepEPDispatcher
from triton_grouped_gemm import persistent_grouped_gemm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE prefill baseline with synthetic data.")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--batch-size", type=int, default=4)        #  每卡的 batch size (本地：1~4； A100: 8~32， 看显存)
    parser.add_argument("--seq-len", type=int, default=4096),       #  单卡总tokens数 = batch_size * seq_len
    parser.add_argument("--ffn-size", type=int, default=2048)       #  专家 FFN 隐层维度（DeepSeek-V3 风格可小于 hidden）
    parser.add_argument("--num-routed-experts", type=int, default=256)
    parser.add_argument("--num-shared-experts", type=int, default=1)
    parser.add_argument("--routed-top-k", type=int, default=8)
    parser.add_argument("--expert-kernel", choices=["torch", "triton"], default="triton")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--comm-backend", choices=["deepep", "torch_a2a"], default="deepep")
    parser.add_argument("--deepep-num-sms", type=int, default=16)   #  DeepEP 使用的 SM 数量 (A100: 16)
    parser.add_argument("--micro-batch-size", type=int, default=4096)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--compare-deepep-kernels", action="store_true")
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--ref-comm-backend", choices=["deepep", "torch_a2a"], default="torch_a2a")
    parser.add_argument("--ref-expert-kernel", choices=["torch", "triton"], default="torch")
    parser.add_argument("--check-atol", type=float, default=5e-2)
    parser.add_argument("--check-rtol", type=float, default=5e-2)
    args = parser.parse_args()
    if args.comm_backend == "torch_a2a" and args.expert_kernel == "triton":
        raise ValueError("Mode torch_a2a + triton is disabled for now.")
    if args.ref_comm_backend == "torch_a2a" and args.ref_expert_kernel == "triton":
        raise ValueError("Reference mode torch_a2a + triton is disabled for now.")
    return args


def init_distributed() -> Tuple[bool, int, int, int]:   # torchrun 启动时会设置这些环境变量：WORLD_SIZE, RANK, LOCAL_RANK
    world_size = int(os.environ.get("WORLD_SIZE", "1"))    # 从环境变量里取 WORLD_SIZE，默认1
    distributed = world_size > 1                           
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return distributed, world_size, rank, local_rank


def get_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int, rank: int) -> None:
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


@contextlib.contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_size, bias=False)
        self.w2 = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


@dataclass
class RouterStats:       # 路由统计信息的数据结构，用于把moe的路由统计指标打包返回
    entropy: float          # (1) 路由分布熵
    load_std: float         # (2) 各expert负载的标准差
    load_cv: float          # (3) 负载变异系数 (std / mean)，衡量不均衡程度


@dataclass
class StepMetrics:
    step_time_ms: float
    step_tps: float
    router_entropy: float
    expert_load_std: float
    expert_load_cv: float


class SimpleMoE(nn.Module):   # 简化版 MoE 模型，包含路由、专家计算、通信后端和overlap配置
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        num_routed_experts: int,
        num_shared_experts: int,
        routed_top_k: int,
        ep_group: dist.ProcessGroup | None = None,
        comm_backend: str = "deepep",
        deepep_num_sms: int = 16,    # 影响 DeepEP 侧用多少 SM 做通信相关 Kernel
        micro_batch_size: int = 4096,
        expert_kernel: str = "triton",
        enable_nvtx: bool = False,
    ):
        super().__init__()
        self.ep_group = ep_group
        self.world_size = ep_group.size() if ep_group is not None else 1
        self.rank = ep_group.rank() if ep_group is not None else 0
        if num_routed_experts % self.world_size != 0:
            raise ValueError(
                f"num_routed_experts ({num_routed_experts}) must be divisible by world_size ({self.world_size})"
            )
        self.num_routed_experts = num_routed_experts
        self.num_local_routed_experts = num_routed_experts // self.world_size
        self.local_expert_offset = self.rank * self.num_local_routed_experts
        self.num_shared_experts = num_shared_experts
        self.routed_top_k = routed_top_k
        self.comm_backend = comm_backend
        self.router = nn.Linear(hidden_size, num_routed_experts, bias=False)   # 路由层，将输入token映射到路由专家索引
        self.routed_experts = nn.ModuleList(
            [ExpertMLP(hidden_size, ffn_size) for _ in range(self.num_local_routed_experts)]   # 本地路由专家分片
        )
        self.shared_experts = nn.ModuleList(
            [ExpertMLP(hidden_size, ffn_size) for _ in range(num_shared_experts)]    # 共享专家
        )
        self.micro_batch_size = micro_batch_size
        self.expert_kernel = expert_kernel
        self.enable_nvtx = enable_nvtx
        self.dispatcher = None
        if self.world_size > 1 and self.comm_backend not in ("deepep", "torch_a2a"):
            raise ValueError(f"Unsupported comm backend: {self.comm_backend}")
        if self.comm_backend == "deepep":
            if ep_group is None:
                raise ValueError("--comm-backend deepep requires distributed execution with torchrun.")
            self.dispatcher = DeepEPDispatcher(  # 构建 DeepEPDispatcher，用于分布式通信
                group=ep_group,
                hidden_size=hidden_size,
                num_experts=num_routed_experts,
                num_sms=deepep_num_sms,
            )
        elif self.comm_backend == "torch_a2a":
            if ep_group is None:
                raise ValueError("--comm-backend torch_a2a requires distributed execution with torchrun.")
            if self.expert_kernel == "triton":
                raise ValueError("Mode torch_a2a + triton is disabled for now.")

    def _build_expert_weight_stacks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # [E_local, H, F] and [E_local, F, H]
        w1 = torch.stack([expert.w1.weight.t().contiguous() for expert in self.routed_experts], dim=0)
        w2 = torch.stack([expert.w2.weight.t().contiguous() for expert in self.routed_experts], dim=0)
        return w1, w2

    def _global_to_local_expert_ids(self, expert_ids: torch.Tensor) -> torch.Tensor:
        local_ids = expert_ids.to(torch.int64) - self.local_expert_offset
        valid = (local_ids >= 0) & (local_ids < self.num_local_routed_experts)
        local_ids = local_ids.masked_fill(~valid, -1)
        return local_ids

    def _run_routed_experts_torch(self, tokens: torch.Tensor, expert_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.zeros_like(tokens)
        counts = torch.zeros(self.num_local_routed_experts, device=tokens.device, dtype=torch.float32)
        for expert_id in range(self.num_local_routed_experts):
            mask = expert_ids == expert_id
            n = int(mask.sum().item())
            if n == 0:
                continue
            counts[expert_id] += n
            out[mask] = self.routed_experts[expert_id](tokens[mask])
        return out, counts

    def _run_routed_experts_grouped(self, tokens: torch.Tensor, expert_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run routed experts with persistent grouped GEMM for one assignment vector.
        `expert_ids` is per-token expert index for this routing pass (e.g. one top-k slot).
        """
        out = torch.zeros_like(tokens)
        counts = torch.zeros(self.num_local_routed_experts, device=tokens.device, dtype=torch.float32)
        valid = expert_ids >= 0
        if not bool(valid.any()):
            return out, counts

        valid_tokens = tokens[valid]
        valid_experts = expert_ids[valid].to(torch.int64)
        sorted_experts, perm = torch.sort(valid_experts)
        grouped_tokens = valid_tokens[perm]

        num_assignments = grouped_tokens.size(0)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(num_assignments, device=perm.device, dtype=perm.dtype)

        counts_i32 = torch.bincount(sorted_experts, minlength=self.num_local_routed_experts).to(torch.int32)
        offsets = torch.zeros((self.num_local_routed_experts + 1,), device=tokens.device, dtype=torch.int32)
        offsets[1:] = torch.cumsum(counts_i32, dim=0)
        counts.copy_(counts_i32.float())

        gemm_dtype = grouped_tokens.dtype if grouped_tokens.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        grouped_tokens_gemm = grouped_tokens.to(gemm_dtype)
        b1, b2 = self._build_expert_weight_stacks()
        b1 = b1.to(gemm_dtype)
        b2 = b2.to(gemm_dtype)
        hidden = persistent_grouped_gemm(grouped_tokens_gemm, b1, offsets)
        hidden = F.gelu(hidden)
        grouped_out = persistent_grouped_gemm(hidden, b2, offsets)

        out_valid = grouped_out[inv_perm].to(out.dtype)
        out[valid] = out_valid
        return out, counts

    def _run_routed_experts(self, tokens: torch.Tensor, expert_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.expert_kernel == "triton":
            return self._run_routed_experts_grouped(tokens, expert_ids)
        return self._run_routed_experts_torch(tokens, expert_ids)

    def _all_to_all_var(self, send_tensor: torch.Tensor, send_splits: torch.Tensor, recv_splits: torch.Tensor) -> torch.Tensor:
        send_splits_list = [int(v) for v in send_splits.tolist()]
        recv_splits_list = [int(v) for v in recv_splits.tolist()]
        out_shape = (int(recv_splits.sum().item()),) + tuple(send_tensor.shape[1:])
        recv_tensor = torch.empty(out_shape, dtype=send_tensor.dtype, device=send_tensor.device)
        dist.all_to_all_single(
            recv_tensor,
            send_tensor.contiguous(),
            output_split_sizes=recv_splits_list,
            input_split_sizes=send_splits_list,
            group=self.ep_group,
        )
        return recv_tensor

    def _run_routed_experts_torch_a2a(self, tokens: torch.Tensor, global_expert_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EP baseline with torch.distributed all_to_all_single + local torch experts.
        Returns per-token outputs (aligned with local input order) and local expert counts.
        """
        out = torch.zeros_like(tokens)
        counts = torch.zeros(self.num_local_routed_experts, device=tokens.device, dtype=torch.float32)

        if self.world_size == 1:
            local_ids = self._global_to_local_expert_ids(global_expert_ids)
            return self._run_routed_experts_torch(tokens, local_ids)

        valid = global_expert_ids >= 0
        if not bool(valid.any()):
            return out, counts

        token_idx = torch.arange(tokens.size(0), device=tokens.device, dtype=torch.int64)
        valid_tokens = tokens[valid]
        valid_token_idx = token_idx[valid]
        valid_global_experts = global_expert_ids[valid].to(torch.int64)

        dst_rank = torch.div(valid_global_experts, self.num_local_routed_experts, rounding_mode='floor').to(torch.int64)
        local_expert_ids = valid_global_experts - dst_rank * self.num_local_routed_experts

        sorted_dst, perm = torch.sort(dst_rank)
        send_tokens = valid_tokens[perm]
        send_token_idx = valid_token_idx[perm]
        send_local_expert_ids = local_expert_ids[perm]

        send_splits = torch.bincount(sorted_dst, minlength=self.world_size).to(torch.int32)
        recv_splits = torch.empty_like(send_splits)
        dist.all_to_all_single(recv_splits, send_splits, group=self.ep_group)

        recv_tokens = self._all_to_all_var(send_tokens, send_splits, recv_splits)
        recv_token_idx = self._all_to_all_var(send_token_idx, send_splits, recv_splits)
        recv_local_expert_ids = self._all_to_all_var(send_local_expert_ids, send_splits, recv_splits)

        recv_out, recv_counts = self._run_routed_experts_torch(recv_tokens, recv_local_expert_ids)
        counts += recv_counts

        # Send computed expert outputs back to original source ranks.
        back_out = self._all_to_all_var(recv_out, recv_splits, send_splits)
        back_token_idx = self._all_to_all_var(recv_token_idx, recv_splits, send_splits)
        out[back_token_idx] = back_out
        return out, counts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, RouterStats]:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)   # [num_tokens, num_routed_experts]
        topk_prob, topk_idx = torch.topk(probs, k=self.routed_top_k, dim=-1)   # [num_tokens, routed_top_k]

        out = torch.zeros_like(x)    #  初始化最终输出缓冲区，用于累加每个expert的输出
        local_token_count = torch.zeros(self.num_local_routed_experts, device=x.device, dtype=torch.float32)  # 本地专家的 tokens 计数器

        if self.comm_backend == "torch_a2a":
            num_tokens = x.size(0)
            micro = max(1, self.micro_batch_size)
            num_chunks = (num_tokens + micro - 1) // micro
            for chunk_idx in range(num_chunks):
                start = chunk_idx * micro
                end = min((chunk_idx + 1) * micro, num_tokens)
                chunk_x = x[start:end]
                chunk_topk_idx = topk_idx[start:end]
                chunk_topk_prob = topk_prob[start:end]

                chunk_out = torch.zeros_like(chunk_x)
                for k in range(self.routed_top_k):
                    global_expert_ids = chunk_topk_idx[:, k]
                    gate = chunk_topk_prob[:, k]
                    y, counts = self._run_routed_experts_torch_a2a(chunk_x, global_expert_ids)
                    local_token_count += counts
                    chunk_out += y * gate.unsqueeze(-1)
                out[start:end] = chunk_out
        elif self.dispatcher is None:   #  单卡本地 fallback
            for k in range(self.routed_top_k):
                expert_ids = self._global_to_local_expert_ids(topk_idx[:, k])
                gate = topk_prob[:, k]
                y, counts = self._run_routed_experts(x, expert_ids)
                local_token_count += counts
                out += y * gate.unsqueeze(-1)
        else:    #  deepEP 通信后端
            num_tokens = x.size(0)      # 当前输入的总 token 数 (单卡)
            micro = max(1, self.micro_batch_size)
            num_chunks = (num_tokens + micro - 1) // micro  #  分几次发
            compute_stream = torch.cuda.Stream(device=x.device) if x.is_cuda else None

            def dispatch_chunk(chunk_idx: int, previous_event: object | None = None):
                start = chunk_idx * micro
                end = min((chunk_idx + 1) * micro, num_tokens)
                chunk_x = x[start:end]
                # DeepEP intranode kernels expect low-precision activations for dispatch/combine.
                if chunk_x.dtype not in (torch.bfloat16, torch.float16):
                    chunk_x = chunk_x.to(torch.bfloat16)
                with nvtx_range(f"deepep_dispatch_chunk_{chunk_idx}", self.enable_nvtx):
                    result = self.dispatcher.dispatch(       # 返回的是deep_ep_comm.py中的 DispatchResult 对象
                        x=chunk_x,
                        topk_idx=topk_idx[start:end].to(torch.int64),
                        topk_weights=topk_prob[start:end].float(),     #  [chunk_size, routed_top_k]
                        previous_event=previous_event,
                    )
                return start, end, result

            # Warm start first chunk communication.
            chunk_start, chunk_end, current = dispatch_chunk(0, previous_event=None)
            pending_combine = None

            for chunk_idx in range(num_chunks):
                # Pre-issue next chunk communication so it can overlap current chunk compute.
                next_chunk = None
                if chunk_idx + 1 < num_chunks:
                    next_chunk = dispatch_chunk(chunk_idx + 1, previous_event=current.event)

                stream_ctx = torch.cuda.stream(compute_stream) if compute_stream is not None else contextlib.nullcontext()
                with stream_ctx:
                    with nvtx_range(f"expert_compute_chunk_{chunk_idx}", self.enable_nvtx):
                        # Wait current chunk communication completion before consuming recv tensors on compute stream.
                        current.event.current_stream_wait()
                        recv_x = current.recv_x
                        recv_topk_idx = current.recv_topk_idx
                        recv_topk_weights = current.recv_topk_weights

                        recv_y = torch.zeros_like(recv_x)
                        if recv_topk_idx.dim() == 2:
                            for k in range(recv_topk_idx.size(1)):
                                expert_ids = recv_topk_idx[:, k]
                                gate = recv_topk_weights[:, k]
                                y, counts = self._run_routed_experts(recv_x, expert_ids)
                                local_token_count += counts
                                recv_y += y * gate.unsqueeze(-1)
                        else:
                            y, counts = self._run_routed_experts(recv_x, recv_topk_idx)
                            local_token_count += counts
                            recv_y += y * recv_topk_weights.unsqueeze(-1)

                    with nvtx_range(f"deepep_combine_chunk_{chunk_idx}", self.enable_nvtx):
                        combined_chunk, combine_event = self.dispatcher.combine(
                            recv_y,
                            current.handle,
                            previous_event=current.event,
                        )

                # Retire previous combine result here so we do not block compute stream.
                if pending_combine is not None:
                    prev_start, prev_end, prev_chunk, prev_event = pending_combine
                    prev_event.current_stream_wait()
                    out[prev_start:prev_end] = prev_chunk
                pending_combine = (chunk_start, chunk_end, combined_chunk, combine_event)

                if next_chunk is not None:
                    chunk_start, chunk_end, current = next_chunk

            if pending_combine is not None:
                prev_start, prev_end, prev_chunk, prev_event = pending_combine
                prev_event.current_stream_wait()
                out[prev_start:prev_end] = prev_chunk

            if compute_stream is not None:
                torch.cuda.current_stream(device=x.device).wait_stream(compute_stream)

        if self.num_shared_experts > 0:
            # Shared experts are dense FFNs applied to all tokens (DeepSeek-style shared experts).
            shared_sum = torch.zeros_like(x)
            for shared_expert in self.shared_experts:
                shared_sum += shared_expert(x)
            out += shared_sum / max(1, self.num_shared_experts)

        token_count = torch.zeros(self.num_routed_experts, device=x.device, dtype=torch.float32)
        token_count[self.local_expert_offset:self.local_expert_offset + self.num_local_routed_experts] = local_token_count
        if self.world_size > 1:
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM, group=self.ep_group)

        mean_load = token_count.mean().clamp_min(1e-6)
        entropy = (-probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean().item()
        stats = RouterStats(
            entropy=entropy,
            load_std=token_count.std().item(),
            load_cv=(token_count.std() / mean_load).item(),
        )
        return out, stats


class TinyMoEModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        num_routed_experts: int,
        num_shared_experts: int,
        routed_top_k: int,
        ep_group: dist.ProcessGroup | None = None,
        comm_backend: str = "deepep",
        deepep_num_sms: int = 24,
        micro_batch_size: int = 4096,
        expert_kernel: str = "triton",
        enable_nvtx: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.moe = SimpleMoE(
            hidden_size,
            ffn_size,
            num_routed_experts,
            num_shared_experts,
            routed_top_k,
            ep_group=ep_group,
            comm_backend=comm_backend,
            deepep_num_sms=deepep_num_sms,
            micro_batch_size=micro_batch_size,
            expert_kernel=expert_kernel,
            enable_nvtx=enable_nvtx,
        )
        self.head = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, RouterStats]:
        h = self.norm(x)
        h, stats = self.moe(h)
        return self.head(h), stats


def synthetic_batch(
    batch_size: int, seq_len: int, hidden_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=dtype)


def reduce_mean(value: float, device: torch.device, distributed: bool) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def run_forward_once(
    model: nn.Module,
    x: torch.Tensor,
    use_amp: bool,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, RouterStats]:
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                y, stats = model(x)
        else:
            y, stats = model(x)
    return y, stats


def measure_single_step(
    model: nn.Module,
    x: torch.Tensor,
    use_amp: bool,
    dtype: torch.dtype,
    device: torch.device,
    world_size: int,
    distributed: bool,
    batch_size: int,
    seq_len: int,
) -> StepMetrics:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, stats = model(x)
        else:
            _, stats = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    step_tokens = batch_size * seq_len * world_size
    return StepMetrics(
        step_time_ms=dt * 1000.0,
        step_tps=step_tokens / max(dt, 1e-9),
        router_entropy=reduce_mean(stats.entropy, device, distributed),
        expert_load_std=reduce_mean(stats.load_std, device, distributed),
        expert_load_cv=reduce_mean(stats.load_cv, device, distributed),
    )


def compare_deepep_kernels_once(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    local_rank: int,
    world_size: int,
    distributed: bool,
    ep_group: dist.ProcessGroup | None,
    dtype: torch.dtype,
    use_amp: bool,
) -> None:
    if args.comm_backend != "deepep":
        raise ValueError("--compare-deepep-kernels requires --comm-backend deepep.")
    x_dtype = torch.float32 if use_amp else dtype
    x = synthetic_batch(args.batch_size, args.seq_len, args.hidden_size, device, x_dtype)
    results: dict[str, StepMetrics] = {}

    for kernel in ("torch", "triton"):
        model = TinyMoEModel(
            args.hidden_size,
            args.ffn_size,
            args.num_routed_experts,
            args.num_shared_experts,
            args.routed_top_k,
            ep_group=ep_group,
            comm_backend="deepep",
            deepep_num_sms=args.deepep_num_sms,
            micro_batch_size=args.micro_batch_size,
            expert_kernel=kernel,
            enable_nvtx=args.enable_nvtx,
        ).to(device)
        model.eval()
        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        for _ in range(max(0, args.warmup_steps)):
            _ = run_forward_once(model, x, use_amp=use_amp, dtype=dtype)
        results[kernel] = measure_single_step(
            model=model,
            x=x,
            use_amp=use_amp,
            dtype=dtype,
            device=device,
            world_size=world_size,
            distributed=distributed,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        if distributed:
            dist.barrier()

    if rank == 0:
        torch_res = results["torch"]
        triton_res = results["triton"]
        print("deepep-kernel-compare: one measured step per kernel")
        print(
            "  deepep+torch: "
            f"step_time_ms={torch_res.step_time_ms:.2f} "
            f"step_tps={torch_res.step_tps:.1f} "
            f"router_entropy={torch_res.router_entropy:.4f} "
            f"expert_load_std={torch_res.expert_load_std:.2f} "
            f"expert_load_cv={torch_res.expert_load_cv:.4f}"
        )
        print(
            "  deepep+triton(persistent_grouped_gemm): "
            f"step_time_ms={triton_res.step_time_ms:.2f} "
            f"step_tps={triton_res.step_tps:.1f} "
            f"router_entropy={triton_res.router_entropy:.4f} "
            f"expert_load_std={triton_res.expert_load_std:.2f} "
            f"expert_load_cv={triton_res.expert_load_cv:.4f}"
        )
        print(
            "  compare: "
            f"time_ratio(torch/triton)={torch_res.step_time_ms / max(triton_res.step_time_ms, 1e-9):.3f}x "
            f"tps_ratio(triton/torch)={triton_res.step_tps / max(torch_res.step_tps, 1e-9):.3f}x "
            f"time_delta_ms={torch_res.step_time_ms - triton_res.step_time_ms:.2f}"
        )


def run_correctness_check(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    distributed: bool,
    ep_group: dist.ProcessGroup | None,
    dtype: torch.dtype,
    use_amp: bool,
) -> None:
    model_ref = TinyMoEModel(
        args.hidden_size,
        args.ffn_size,
        args.num_routed_experts,
        args.num_shared_experts,
        args.routed_top_k,
        ep_group=ep_group,
        comm_backend=args.ref_comm_backend,
        deepep_num_sms=args.deepep_num_sms,
        micro_batch_size=args.micro_batch_size,
        expert_kernel=args.ref_expert_kernel,
        enable_nvtx=args.enable_nvtx,
    ).to(device)
    model_test = TinyMoEModel(
        args.hidden_size,
        args.ffn_size,
        args.num_routed_experts,
        args.num_shared_experts,
        args.routed_top_k,
        ep_group=ep_group,
        comm_backend=args.comm_backend,
        deepep_num_sms=args.deepep_num_sms,
        micro_batch_size=args.micro_batch_size,
        expert_kernel=args.expert_kernel,
        enable_nvtx=args.enable_nvtx,
    ).to(device)
    model_test.load_state_dict(model_ref.state_dict())
    model_ref.eval()
    model_test.eval()

    x_dtype = torch.float32 if use_amp else dtype
    x = synthetic_batch(args.batch_size, args.seq_len, args.hidden_size, device, x_dtype)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if distributed:
        dist.barrier()

    y_ref, stats_ref = run_forward_once(model_ref, x, use_amp, dtype)
    y_test, stats_test = run_forward_once(model_test, x, use_amp, dtype)

    diff = (y_test.float() - y_ref.float()).abs()
    max_abs = diff.max()
    mean_abs = diff.mean()
    ref_mean_abs = y_ref.float().abs().mean().clamp_min(1e-8)
    rel_mean = mean_abs / ref_mean_abs

    stats_diff = torch.tensor(
        [
            abs(stats_test.entropy - stats_ref.entropy),
            abs(stats_test.load_std - stats_ref.load_std),
            abs(stats_test.load_cv - stats_ref.load_cv),
        ],
        device=device,
        dtype=torch.float32,
    )

    if distributed:
        dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
        dist.all_reduce(mean_abs, op=dist.ReduceOp.SUM)
        mean_abs /= dist.get_world_size()
        dist.all_reduce(rel_mean, op=dist.ReduceOp.SUM)
        rel_mean /= dist.get_world_size()
        dist.all_reduce(stats_diff, op=dist.ReduceOp.MAX)

    ok = bool((max_abs <= args.check_atol) and (rel_mean <= args.check_rtol))
    ok_tensor = torch.tensor([1 if ok else 0], device=device, dtype=torch.int32)
    if distributed:
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    ok_all = bool(ok_tensor.item() == 1)

    if rank == 0:
        print(
            "correctness: "
            f"test=({args.comm_backend}, {args.expert_kernel}) "
            f"ref=({args.ref_comm_backend}, {args.ref_expert_kernel}) "
            f"max_abs={float(max_abs.item()):.6e} "
            f"mean_abs={float(mean_abs.item()):.6e} "
            f"rel_mean={float(rel_mean.item()):.6e} "
            f"router_delta_max=[entropy:{float(stats_diff[0].item()):.6e}, "
            f"load_std:{float(stats_diff[1].item()):.6e}, "
            f"load_cv:{float(stats_diff[2].item()):.6e}] "
            f"thresholds=[atol:{args.check_atol:.3e}, rtol:{args.check_rtol:.3e}] "
            f"result={'PASS' if ok_all else 'FAIL'}"
        )

    if distributed:
        dist.barrier()
    if not ok_all:
        raise RuntimeError("Correctness check failed: output mismatch exceeds tolerance.")


def main() -> None:
    args = parse_args()
    distributed, world_size, rank, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    dtype = get_dtype(args.dtype)
    set_seed(args.seed, rank)

    ffn_size = args.ffn_size
    ep_group = dist.group.WORLD if distributed else None
    use_amp = dtype in (torch.float16, torch.bfloat16)  # Auto Mixed Precision，自动混合精度

    if args.check_correctness:
        if rank == 0:
            print(
                "correctness-mode: running one-shot comparison "
                f"test=({args.comm_backend}, {args.expert_kernel}) vs "
                f"ref=({args.ref_comm_backend}, {args.ref_expert_kernel})"
            )
        run_correctness_check(
            args=args,
            device=device,
            rank=rank,
            distributed=distributed,
            ep_group=ep_group,
            dtype=dtype,
            use_amp=use_amp,
        )
        if distributed:
            dist.destroy_process_group()
        return

    if args.compare_deepep_kernels:
        compare_deepep_kernels_once(
            args=args,
            device=device,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed=distributed,
            ep_group=ep_group,
            dtype=dtype,
            use_amp=use_amp,
        )
        if distributed:
            dist.destroy_process_group()
        return

    model = TinyMoEModel(
        args.hidden_size,
        ffn_size,
        args.num_routed_experts,
        args.num_shared_experts,
        args.routed_top_k,
        ep_group=ep_group,
        comm_backend=args.comm_backend,
        deepep_num_sms=args.deepep_num_sms,
        micro_batch_size=args.micro_batch_size,
        expert_kernel=args.expert_kernel,
        enable_nvtx=args.enable_nvtx,
    ).to(device)
    model.eval()               #  设置为评估模式，禁用 dropout 等训练相关的层
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if rank == 0:
        print(
            f"start: mode=prefill, world_size={world_size}, dtype={args.dtype}, "
            f"tokens/step={args.batch_size * args.seq_len * world_size}, "
            f"comm_backend={args.comm_backend}, amp={use_amp}, "
            f"routed_experts={args.num_routed_experts}, shared_experts={args.num_shared_experts}, "
            f"routed_top_k={args.routed_top_k}, micro_batch_size={args.micro_batch_size}, "
            f"expert_kernel={args.expert_kernel}, nvtx={args.enable_nvtx}"
        )

    warmup_steps = max(0, args.warmup_steps)
    if rank == 0:
        print(f"warmup: running {warmup_steps} steps (excluded from metrics)")
    for _ in range(warmup_steps):
        x_dtype = torch.float32 if use_amp else dtype
        x = synthetic_batch(args.batch_size, args.seq_len, args.hidden_size, device, x_dtype)
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    model(x)
            else:
                model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)    # 等当前GPU上所有异步CUDA完成
    if distributed:
        dist.barrier()   # 等待所有进程到达同步点

    total_tokens = 0
    total_time = 0.0

    for step in range(1, args.steps + 1):
        x_dtype = torch.float32 if use_amp else dtype
        x = synthetic_batch(args.batch_size, args.seq_len, args.hidden_size, device, x_dtype)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _, stats = model(x)
            else:
                _, stats = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        step_tokens = args.batch_size * args.seq_len * world_size
        total_tokens += step_tokens
        total_time += dt

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            step_tps = step_tokens / max(dt, 1e-9)
            avg_tps = total_tokens / max(total_time, 1e-9)
            entropy = reduce_mean(stats.entropy, device, distributed)
            load_std = reduce_mean(stats.load_std, device, distributed)
            load_cv = reduce_mean(stats.load_cv, device, distributed)
            if rank == 0:
                print(
                    f"step={step:04d} "
                    f"step_time_ms={dt * 1000:.2f} "
                    f"step_tps={step_tps:.1f} "
                    f"avg_tps={avg_tps:.1f} "
                    f"router_entropy={entropy:.4f} "
                    f"expert_load_std={load_std:.2f} "
                    f"expert_load_cv={load_cv:.4f}"
                )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this baseline.")
    main()
