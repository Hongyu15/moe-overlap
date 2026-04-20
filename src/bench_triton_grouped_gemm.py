import argparse
import time

import torch

from triton_grouped_gemm import persistent_grouped_gemm, reference_grouped_gemm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark Triton persistent grouped GEMM")
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--tokens-per-expert", type=int, default=512)
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


def build_offsets(num_experts: int, tokens_per_expert: int, device: torch.device) -> torch.Tensor:
    counts = torch.full((num_experts,), tokens_per_expert, device=device, dtype=torch.int32)
    offsets = torch.zeros((num_experts + 1,), device=device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    offsets = build_offsets(args.num_experts, args.tokens_per_expert, device)
    m_total = int(offsets[-1].item())
    a = torch.randn((m_total, args.hidden_size), device=device, dtype=dtype)
    b = torch.randn((args.num_experts, args.hidden_size, args.ffn_size), device=device, dtype=dtype)

    use_atomic = args.atomic_queue
    with torch.no_grad():
        c_ref = reference_grouped_gemm(a, b, offsets)
        c_tri = persistent_grouped_gemm(
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
