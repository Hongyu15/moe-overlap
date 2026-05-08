#!/usr/bin/env python3
"""
One-shot DeepEP dispatch; after sync, each rank prints recv_x, recv_topk_idx, recv_topk_weights
in turn (barriers so output does not mix).

  torchrun --nproc_per_node=2 src/verify_deepep_dispatch.py --hidden-size 512 --num-tokens 32 --num-experts 8

PYTHONPATH must include this repo's `src`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEEPEP_ROOT = _REPO_ROOT / "thirdParty" / "DeepEP"
if str(_DEEPEP_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEEPEP_ROOT))

from deep_ep_comm import DeepEPDispatcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-tokens", type=int, default=32)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--deepep-num-sms", type=int, default=16)
    return p.parse_args()


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

    import deep_ep  # type: ignore

    dispatcher = DeepEPDispatcher(
        dist.group.WORLD,
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        num_sms=args.deepep_num_sms,
    )

    t, h, k = args.num_tokens, args.hidden_size, args.top_k
    x = torch.zeros((t, h), device=device, dtype=torch.bfloat16)
    for i in range(t):
        x[i, 0] = float(rank * 10000 + i)

    topk_idx = torch.empty((t, k), device=device, dtype=deep_ep.topk_idx_t)
    topk_weights = torch.ones((t, k), device=device, dtype=torch.float32)
    for i in range(t):
        base = (rank * t + i) % args.num_experts
        for j in range(k):
            topk_idx[i, j] = (base + j) % args.num_experts

    result = dispatcher.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, previous_event=None)
    result.event.current_stream_wait()

    recv_x = result.recv_x
    recv_topk_idx = result.recv_topk_idx
    recv_topk_weights = result.recv_topk_weights

    for r in range(world_size):
        dist.barrier()
        if rank == r:
            print(f"\n----- rank {rank} recv -----")
            print("recv_x:\n", recv_x.cpu())
            print("recv_topk_idx:\n", recv_topk_idx.cpu())
            print("recv_topk_weights:\n", recv_topk_weights.cpu())
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
