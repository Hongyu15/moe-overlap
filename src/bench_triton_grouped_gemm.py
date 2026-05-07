import argparse
import time

import torch

from triton_grouped_gemm import persistent_grouped_gemm, reference_grouped_gemm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark Triton persistent grouped GEMM")
    parser.add_argument("--num-routed-experts", type=int, default=256)
    parser.add_argument("--num-shared-experts", type=int, default=1)
    parser.add_argument("--routed-top-k", type=int, default=8)
    parser.add_argument("--tokens-per-expert", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--ffn-size", type=int, default=2048)
    parser.add_argument("--num-sms", type=int, default=84)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--test-mode", choices=["both", "triton", "torch"], default="both")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--atomic-queue",
        action="store_true",
        help="Use global atomic dequeue for tiles (dynamic balance across CTAs).",
    )
    return parser.parse_args()


def build_counts(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.full(
        (args.num_routed_experts,),
        args.tokens_per_expert,
        device=device,
        dtype=torch.int32,
    )


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
    b = torch.randn((args.num_routed_experts, args.hidden_size, args.ffn_size), device=device, dtype=dtype)

    print(
        f"[moe] routed_experts={args.num_routed_experts} shared_experts={args.num_shared_experts} "
        f"topk={args.routed_top_k} hidden={args.hidden_size} ffn={args.ffn_size} "
        f"[load] mode=uniform total_tokens={m_total} "
        f"min={int(counts.min().item())} max={int(counts.max().item())} "
        f"mean={float(counts.float().mean().item()):.1f} "
        f"cv={float((counts.float().std() / counts.float().mean().clamp_min(1e-9)).item()):.4f}"
    )

    use_atomic = args.atomic_queue
    run_triton = args.test_mode in ("both", "triton")
    run_torch = args.test_mode in ("both", "torch")
    triton_ms = None
    torch_ms = None

    triton_kwargs = dict(
        num_sms=args.num_sms,
        use_atomic_queue=use_atomic,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )

    with torch.no_grad():
        c_ref = None
        c_tri = None
        if run_torch:
            c_ref = reference_grouped_gemm(a, b, offsets)
        if run_triton:
            c_tri = persistent_grouped_gemm(a, b, offsets, **triton_kwargs)
        if run_torch and run_triton:
            max_diff = (c_ref.float() - c_tri.float()).abs().max().item()
            print(f"[check] mode={'atomic' if use_atomic else 'striped'} max_abs_diff={max_diff:.6f}")

    if run_triton:
        for _ in range(args.warmup):
            _ = persistent_grouped_gemm(a, b, offsets, **triton_kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = persistent_grouped_gemm(a, b, offsets, **triton_kwargs)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) * 1000 / args.iters

    if run_torch:
        for _ in range(args.warmup):
            _ = reference_grouped_gemm(a, b, offsets)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = reference_grouped_gemm(a, b, offsets)
        torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - t0) * 1000 / args.iters

    if run_triton and run_torch:
        print(
            f"[perf] mode={'atomic' if use_atomic else 'striped'} "
            f"triton_ms={triton_ms:.3f} torch_loop_ms={torch_ms:.3f} "
            f"speedup={torch_ms / max(triton_ms, 1e-9):.3f}x"
        )
    elif run_triton:
        print(f"[perf] mode={'atomic' if use_atomic else 'striped'} test_mode=triton triton_ms={triton_ms:.3f}")
    else:
        print(f"[perf] mode={'atomic' if use_atomic else 'striped'} test_mode=torch torch_loop_ms={torch_ms:.3f}")


if __name__ == "__main__":
    main()
