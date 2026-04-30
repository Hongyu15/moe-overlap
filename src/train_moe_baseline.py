import argparse
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE baseline with synthetic data.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)        #  每卡的 batch size (本地：1~4； A100: 8~32， 看显存)
    parser.add_argument("--seq-len", type=int, default=4096),
    parser.add_argument("--ffn-multiplier", type=int, default=4)    #  FFN 的维度放大倍数 (本地: 2~4; A100: 4)
    parser.add_argument("--num-experts", type=int, default=16)      #  专家数量 (本地: NAN; A100: 64)
    parser.add_argument("--top-k", type=int, default=2)             #  每个样本选择多少个专家 (本地: 2; A100: 2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--use-deepep", action="store_true")        #  使用 DeepEP 作为通信后端 (本地: False; A100: True)
    parser.add_argument("--deepep-num-sms", type=int, default=16)   #  DeepEP 使用的 SM 数量 (A100: 24)
    return parser.parse_args()


def init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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


class SimpleMoE(nn.Module):   # 简化版 MoE 模型，包含路由、专家计算、通信后端和overlap配置
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        num_experts: int,
        top_k: int,
        ep_group: dist.ProcessGroup | None = None,
        use_deepep: bool = False,
        deepep_num_sms: int = 24,    # 影响 DeepEP 侧用多少 SM 做通信相关 Kernel
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)   # 路由层，将输入token映射到专家索引
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_size, ffn_size) for _ in range(num_experts)]   # 专家列表，每个专家是一个 MLP 层
        )
        self.ep_group = ep_group
        self.dispatcher = None   # 通信后端，None表示使用 baseline dispatcher；True 时会构建 DeepEPDispatcher
        if use_deepep:
            if ep_group is None:
                raise ValueError("--use-deepep requires distributed execution with torchrun.")
            self.dispatcher = DeepEPDispatcher(  # 构建 DeepEPDispatcher，用于分布式通信
                group=ep_group,
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_sms=deepep_num_sms,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, RouterStats]:
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        topk_prob, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)

        out = torch.zeros_like(x)    #  初始化最终输出缓冲区，用于累加每个expert的输出
        token_count = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)  #  初始化每个expert的token计数器

        if self.dispatcher is None:   #  没有使用 DeepEP，使用 baseline dispatcher，显式索引路由，易于调试
            # Baseline dispatcher: explicit index routing, easy to instrument.
            for k in range(self.top_k):
                expert_ids = topk_idx[:, k]
                gate = topk_prob[:, k]
                for expert_id in range(self.num_experts):
                    mask = expert_ids == expert_id
                    n = int(mask.sum().item())
                    if n == 0:
                        continue
                    token_count[expert_id] += n
                    y = self.experts[expert_id](x[mask])
                    out[mask] += y * gate[mask].unsqueeze(-1)
        else:    #  deepEP 通信后端
            dispatch = self.dispatcher.dispatch(
                x=x,
                topk_idx=topk_idx.to(torch.int64),
                topk_weights=topk_prob.float(),
            )
            recv_x = dispatch.recv_x
            recv_topk_idx = dispatch.recv_topk_idx
            recv_topk_weights = dispatch.recv_topk_weights
            recv_y = torch.zeros_like(recv_x)
            for expert_id in range(self.num_experts):
                mask = recv_topk_idx == expert_id
                if mask.dim() == 2:
                    mask = mask.any(dim=1)
                n = int(mask.sum().item())
                if n == 0:
                    continue
                token_count[expert_id] += n
                y = self.experts[expert_id](recv_x[mask])
                # For top-k>1 DeepEP returns per-token topk indices/weights;
                # baseline keeps a simple sum over selected experts.
                if recv_topk_weights.dim() == 2:
                    w = recv_topk_weights[mask].mean(dim=1, keepdim=True)
                else:
                    w = recv_topk_weights[mask].unsqueeze(-1)
                recv_y[mask] += y * w

            out, _ = self.dispatcher.combine(recv_y, dispatch.handle, previous_event=dispatch.event)

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
        num_experts: int,
        top_k: int,
        ep_group: dist.ProcessGroup | None = None,
        use_deepep: bool = False,
        deepep_num_sms: int = 24,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.moe = SimpleMoE(
            hidden_size,
            ffn_size,
            num_experts,
            top_k,
            ep_group=ep_group,
            use_deepep=use_deepep,
            deepep_num_sms=deepep_num_sms,
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


def main() -> None:
    args = parse_args()
    distributed, world_size, rank, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    dtype = get_dtype(args.dtype)
    set_seed(args.seed, rank)

    ffn_size = args.hidden_size * args.ffn_multiplier
    ep_group = dist.group.WORLD if distributed else None
    # FP16/BF16: keep weights in FP32 and use autocast (pure half weights + AdamW tends to NaN).
    use_amp = dtype in (torch.float16, torch.bfloat16)
    model = TinyMoEModel(
        args.hidden_size,
        ffn_size,
        args.num_experts,
        args.top_k,
        ep_group=ep_group,
        use_deepep=args.use_deepep,
        deepep_num_sms=args.deepep_num_sms,
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if rank == 0:
        print(
            f"start: world_size={world_size}, dtype={args.dtype}, "
            f"tokens/step={args.batch_size * args.seq_len * world_size}, "
            f"use_deepep={args.use_deepep}, amp={use_amp}"
        )

    total_tokens = 0
    total_time = 0.0

    for step in range(1, args.steps + 1):
        x_dtype = torch.float32 if use_amp else dtype
        x = synthetic_batch(args.batch_size, args.seq_len, args.hidden_size, device, x_dtype)
        target = x.detach().float()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                pred, stats = model(x)
        else:
            pred, stats = model(x)
        loss = F.mse_loss(pred.float(), target)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        step_tokens = args.batch_size * args.seq_len * world_size
        total_tokens += step_tokens
        total_time += dt

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            step_tps = step_tokens / max(dt, 1e-9)
            avg_tps = total_tokens / max(total_time, 1e-9)
            loss_val = reduce_mean(float(loss.item()), device, distributed)
            entropy = reduce_mean(stats.entropy, device, distributed)
            load_std = reduce_mean(stats.load_std, device, distributed)
            load_cv = reduce_mean(stats.load_cv, device, distributed)
            if rank == 0:
                print(
                    f"step={step:04d} "
                    f"loss={loss_val:.6f} "
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
