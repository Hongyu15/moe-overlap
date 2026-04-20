from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class DispatchResult:
    recv_x: torch.Tensor
    recv_topk_idx: torch.Tensor
    recv_topk_weights: torch.Tensor
    handle: tuple
    event: object


class DeepEPDispatcher:
    """
    Thin wrapper around DeepEP dispatch/combine APIs for intranode experiments.
    """

    def __init__(self, group: dist.ProcessGroup, hidden_size: int, num_experts: int, num_sms: int = 24):
        self.group = group
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self._buffer = None
        self._deep_ep = self._import_deep_ep()
        self._deep_ep.Buffer.set_num_sms(num_sms)
        self._build_buffer()

    def _import_deep_ep(self):
        # Let project-local thirdParty/DeepEP be importable without global install.
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        deepep_root = repo_root / "thirdParty" / "DeepEP"
        if str(deepep_root) not in sys.path:
            sys.path.insert(0, str(deepep_root))
        try:
            import deep_ep  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import deep_ep. Build/install DeepEP first, e.g. "
                "`cd thirdParty/DeepEP && NVSHMEM_DIR=/path/to/nvshmem python setup.py build`."
            ) from exc
        return deep_ep

    def _build_buffer(self) -> None:
        hidden_bytes = self.hidden_size * 2  # BF16
        num_nvl_bytes = 0
        for cfg in (
            self._deep_ep.Buffer.get_dispatch_config(self.group.size()),
            self._deep_ep.Buffer.get_combine_config(self.group.size()),
        ):
            num_nvl_bytes = max(
                num_nvl_bytes,
                int(cfg.get_nvl_buffer_size_hint(hidden_bytes, self.group.size())),
            )
        self._buffer = self._deep_ep.Buffer(self.group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event: Optional[object] = None,
    ) -> DispatchResult:
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = (
            self._buffer.get_dispatch_layout(
                topk_idx,
                self.num_experts,
                previous_event=previous_event,
                async_finish=True,
                allocate_on_comm_stream=previous_event is not None,
            )
        )
        recv_x, recv_topk_idx, recv_topk_weights, _, handle, event = self._buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        if isinstance(recv_x, tuple):
            raise RuntimeError("This baseline expects BF16 path (tensor), not FP8 tuple.")
        return DispatchResult(
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            handle=handle,
            event=event,
        )

    def combine(self, expert_out: torch.Tensor, handle: tuple, previous_event: Optional[object] = None) -> Tuple[torch.Tensor, object]:
        combined_x, _, event = self._buffer.combine(
            expert_out,
            handle,
            async_finish=True,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
        )
        return combined_x, event

