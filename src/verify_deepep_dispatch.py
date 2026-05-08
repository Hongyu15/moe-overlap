#!/usr/bin/env python3
"""
Standalone DeepEP dispatch sanity check (intranode only: num_rdma_bytes=0 in DeepEPDispatcher).

What it does:
  - Runs one dispatch per rank with synthetic tokens whose first hidden column encodes (sender_rank, local_row).
  - Waits on DeepEP's async event, then inspects recv_x / recv_topk_idx / handle rank_prefix_matrix.
  - Reports whether rows are globally sorted by LOCAL expert id (usually False for DeepEP).
  - Reports within each sender-rank segment (rows [prev, cum_end) from rank_prefix_matrix[:, rank]) whether expert ids are sorted.

Usage (must match DeepEP intranode topology, NVSHMEM built):
  torchrun --nproc_per_node=2 src/verify_deepep_dispatch.py \\
      --hidden-size 512 --num-tokens 32 --num-experts 8 --top-k 1 --deepep-num-sms 16

Requires PYTHONPATH to include the repo `src` directory (same as train_moe_baseline).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# Repo-local DeepEP (same as deep_ep_comm)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEEPEP_ROOT = _REPO_ROOT / "thirdParty" / "DeepEP"
if str(_DEEPEP_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEEPEP_ROOT))

from deep_ep_comm import DeepEPDispatcher  # noqa: E402


def _import_deep_ep():
    import deep_ep  # type: ignore

    return deep_ep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify DeepEP dispatch layout across GPUs.")
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-tokens", type=int, default=32, help="Local tokens per rank before dispatch.")
    p.add_argument("--num-experts", type=int, default=8, help="Global experts; must divide world_size.")
    p.add_argument("--top-k", type=int, default=1, help="Dispatch top-k columns (1 is easiest to read).")
    p.add_argument("--deepep-num-sms", type=int, default=16)
    return p.parse_args()


def _analyze_receiver(
    rank: int,
    world_size: int,
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    rank_prefix_matrix: torch.Tensor,
    num_local_experts: int,
) -> None:
    n = recv_x.size(0)
    print(f"\n========== Receiver rank {rank} ==========")
    print(f"recv_x shape: {tuple(recv_x.shape)}  dtype={recv_x.dtype}")
    print(f"recv_topk_idx shape: {tuple(recv_topk_idx.shape)}  dtype={recv_topk_idx.dtype}")

    # DeepEP stores LOCAL expert index on each rank (0 .. num_local_experts-1)
    if recv_topk_idx.dim() == 2:
        col0 = recv_topk_idx[:, 0].long()
    else:
        col0 = recv_topk_idx.long()

    print(f"First min(n,16) local expert ids: {col0[: min(n, 16)].tolist()}")

    if n >= 2:
        mono = bool(torch.all(col0[:-1] <= col0[1:]).item())
        print(f"Rows globally non-decreasing by local expert id? {mono} (DeepEP usually False)")

    # Segments by sender rank: rows belong to sender i in [prev, rank_prefix_matrix[i, rank])
    rpm = rank_prefix_matrix
    prev = 0
    print("Per-sender segments (recv row ranges on this rank):")
    for sender in range(world_size):
        end = int(rpm[sender, rank].item())
        seg = col0[prev:end]
        if seg.numel() >= 2:
            seg_sorted = bool(torch.all(seg[:-1] <= seg[1:]).item())
        else:
            seg_sorted = True
        print(
            f"  sender_rank={sender}: rows [{prev}, {end})  count={end - prev}  "
            f"sorted_by_local_expert={seg_sorted}"
        )
        if end > prev:
            print(f"    expert ids sample: {seg[: min(seg.numel(), 12)].tolist()}")
        prev = end
    if prev != n:
        print(f"  WARNING: prefix ends at {prev} but recv rows={n}")

    counts = torch.bincount(col0.clamp(min=0), minlength=num_local_experts)
    print(f"bincount(local expert id, len={num_local_experts}): {counts.tolist()}")

    # Decode probe column if present (see synthetic build below)
    tag = recv_x[:, 0].float()
    send_rank_guess = torch.div(tag, 10000, rounding_mode="floor").long()
    local_row_guess = (tag - send_rank_guess.float() * 10000).long()
    print(f"decoded sender_rank from recv_x[:,0] first 12: {send_rank_guess[:12].tolist()}")
    print(f"decoded local_row     first 12: {local_row_guess[:12].tolist()}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if args.num_experts % world_size != 0:
        raise ValueError("num_experts must divide world_size.")
    num_local = args.num_experts // world_size

    deep_ep = _import_deep_ep()
    ep_group = dist.group.WORLD

    dispatcher = DeepEPDispatcher(
        ep_group,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        num_sms=args.deepep_num_sms,
    )

    t = args.num_tokens
    h = args.hidden_size
    k = args.top_k

    x = torch.zeros((t, h), device=device, dtype=torch.bfloat16)
    # Tag rows so recv side can see which rank / which row was sent (column 0 only).
    for i in range(t):
        x[i, 0] = float(rank * 10000 + i)

    # Deterministic routing: each local row i uses global experts (base+i), (+j per slot if top-k>1).
    topk_idx = torch.empty((t, k), device=device, dtype=deep_ep.topk_idx_t)
    topk_weights = torch.ones((t, k), device=device, dtype=torch.float32)
    for i in range(t):
        base = (rank * t + i) % args.num_experts
        for j in range(k):
            topk_idx[i, j] = (base + j) % args.num_experts

    result = dispatcher.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        previous_event=None,
    )
    result.event.current_stream_wait()     #  等 DeepEP这次 dispatch在GPU上真正完成

    recv_x = result.recv_x
    recv_topk_idx = result.recv_topk_idx
    handle = result.handle
    rank_prefix_matrix = handle[0]

    _analyze_receiver(rank, world_size, recv_x, recv_topk_idx, rank_prefix_matrix, num_local)

    dist.barrier()
    if rank == 0:
        print(
            "\nNotes:\n"
            "  - If decoded sender_rank segments match sender_rank labels, tokens crossed GPUs as expected.\n"
            "  - DeepEP recv_x row order follows sender-rank prefixes (rank_prefix_matrix), not sorted local expert.\n"
            "  - Use bincount vs expected routing to sanity-check counts.\n"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
