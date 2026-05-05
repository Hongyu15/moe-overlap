import argparse
import time

import torch

from triton_grouped_gemm import persistent_grouped_gemm, reference_grouped_gemm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark Triton persistent grouped GEMM")
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--tokens-per-expert", type=int, default=512)
    parser.add_argument(
        "--load-mode",
        choices=["uniform", "hotspot", "zipf"],
        default="uniform",
        help="Token distribution across experts.",
    )
    parser.add_argument(
        "--hot-expert-ratio",
        type=float,
        default=0.25,
        help="Hot expert ratio used when --load-mode=hotspot.",
    )
    parser.add_argument(
        "--hot-token-ratio",
        type=float,
        default=0.8,
        help="Token ratio assigned to hot experts when --load-mode=hotspot.",
    )
    parser.add_argument(
        "--zipf-alpha",
        type=float,
        default=1.2,
        help="Zipf alpha used when --load-mode=zipf (larger => more imbalanced).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-cv",
        type=float,
        default=None,
        help="Target coefficient of variation (std/mean) for expert token counts. Overrides --load-mode when set.",
    )
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-size", type=int, default=4096)
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--atomic-queue",
        action="store_true",
        help="Use global atomic dequeue for tiles (dynamic balance across CTAs).",
    )
    return parser.parse_args()


def build_counts(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    total_tokens = args.num_experts * args.tokens_per_expert

    if args.target_cv is not None:
        if args.target_cv < 0:
            raise ValueError("--target-cv must be non-negative.")
        if args.target_cv == 0:
            return torch.full((args.num_experts,), args.tokens_per_expert, device=device, dtype=torch.int32)

        # For lognormal weights 正态分布: cv = sqrt(exp(sigma^2) - 1).
        sigma = float(torch.sqrt(torch.tensor(torch.log(torch.tensor(args.target_cv * args.target_cv + 1.0)))))
        noise = torch.randn((args.num_experts,), device=device)
        weights = torch.exp(noise * sigma)
        weights = weights / weights.sum()
        counts = torch.floor(weights * total_tokens).to(torch.int32)
        remaining = total_tokens - int(counts.sum().item())
        if remaining > 0:
            frac = (weights * total_tokens) - counts.float()
            idx = torch.argsort(frac, descending=True)[:remaining]
            counts[idx] += 1
        return counts

    if args.load_mode == "uniform":
        return torch.full((args.num_experts,), args.tokens_per_expert, device=device, dtype=torch.int32)

    if args.load_mode == "hotspot":
        num_hot = max(1, int(round(args.num_experts * args.hot_expert_ratio)))
        num_hot = min(num_hot, args.num_experts)
        num_cold = args.num_experts - num_hot

        hot_tokens = int(round(total_tokens * args.hot_token_ratio))
        hot_tokens = max(num_hot, min(hot_tokens, total_tokens - num_cold))
        cold_tokens = total_tokens - hot_tokens

        counts = torch.zeros((args.num_experts,), device=device, dtype=torch.int32)
        counts[:num_hot] = hot_tokens // num_hot
        counts[num_hot:] = 0 if num_cold == 0 else cold_tokens // num_cold

        # Distribute remainders to keep exact token total.
        hot_rem = hot_tokens - int(counts[:num_hot].sum().item())
        cold_rem = cold_tokens - int(counts[num_hot:].sum().item())
        if hot_rem > 0:
            counts[:hot_rem] += 1
        if cold_rem > 0 and num_cold > 0:
            counts[num_hot:num_hot + cold_rem] += 1
        return counts

    # Zipf-style long-tail counts.
    ranks = torch.arange(1, args.num_experts + 1, device=device, dtype=torch.float32)
    probs = ranks.pow(-args.zipf_alpha)
    probs = probs / probs.sum()
    counts = torch.floor(probs * total_tokens).to(torch.int32)
    remaining = total_tokens - int(counts.sum().item())
    if remaining > 0:
        counts[:remaining] += 1
    return counts


def build_offsets(counts: torch.Tensor) -> torch.Tensor:
    if counts.dim() != 1:
        raise ValueError("counts must be a 1D tensor.")
    offsets = torch.zeros((counts.numel() + 1,), device=counts.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(args.seed)

    counts = build_counts(args, device)   # 确定各个expert上的tokens数量
    offsets = build_offsets(counts)       # 计算前缀和
    m_total = int(offsets[-1].item())
    a = torch.randn((m_total, args.hidden_size), device=device, dtype=dtype)
    b = torch.randn((args.num_experts, args.hidden_size, args.ffn_size), device=device, dtype=dtype)

    print(
        f"[load] mode={'target-cv' if args.target_cv is not None else args.load_mode} total_tokens={m_total} "
        f"min={int(counts.min().item())} max={int(counts.max().item())} "
        f"mean={float(counts.float().mean().item()):.1f} "
        f"cv={float((counts.float().std() / counts.float().mean().clamp_min(1e-9)).item()):.4f}"
    )

    use_atomic = args.atomic_queue
    with torch.no_grad():
        c_ref = reference_grouped_gemm(a, b, offsets)   #  Pytorch 串行 grouped matmul
        c_tri = persistent_grouped_gemm(                # 用 Triton persistent grouped GEMM 计算
            a, b, offsets, num_sms=args.num_sms, use_atomic_queue=use_atomic
        )
        max_diff = (c_ref.float() - c_tri.float()).abs().max().item()
        print(f"[check] mode={'atomic' if use_atomic else 'striped'} max_abs_diff={max_diff:.6f}")

    for _ in range(args.warmup):
        _ = persistent_grouped_gemm(
            a, b, offsets, num_sms=args.num_sms, use_atomic_queue=use_atomic
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = persistent_grouped_gemm(
            a, b, offsets, num_sms=args.num_sms, use_atomic_queue=use_atomic
        )
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - t0) * 1000 / args.iters

    for _ in range(args.warmup):
        _ = reference_grouped_gemm(a, b, offsets)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        _ = reference_grouped_gemm(a, b, offsets)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) * 1000 / args.iters

    print(
        f"[perf] mode={'atomic' if use_atomic else 'striped'} "
        f"triton_ms={triton_ms:.3f} torch_loop_ms={torch_ms:.3f} "
        f"speedup={torch_ms / max(triton_ms, 1e-9):.3f}x"
    )


if __name__ == "__main__":
    main()
