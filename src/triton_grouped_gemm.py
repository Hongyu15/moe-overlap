from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "triton is required for triton_grouped_gemm.py. Install with `pip install triton`."
    ) from exc

"""
Grouped GEMM for MoE expert compute (A_i @ B_i), with two scheduling modes:

1) **Striped persistent**: `num_sms` blocks stride over a pre-built work list
   (`work_id += num_workers`). Same total work as atomic mode, easier to reason about.

2) **Atomic work queue** (default, `use_atomic_queue=True`): all blocks compete for the next tile
   index via `atomic_add` on a device counter. This is the common GPU pattern for
   **dynamic load balance** (often loosely called "work stealing" in papers; strictly
   it is **global task dequeue**, not thief-side stealing from peer queues).

**DeepEP / overlap note:** DeepEP `dispatch` gives `recv_x` and metadata; you must
layout tokens **contiguous per local expert** in `a_cat` (or repack) so that
`expert_offsets[e]` / `expert_offsets[e+1]` delimit rows for expert `e`. Build
offsets from `num_recv_tokens_per_expert_list` via `expert_offsets_from_recv_counts`.
Communication overlap is then orchestrated **outside** this kernel (separate CUDA
stream + events): while the next chunk is in flight, launch this kernel on tiles
that are already valid in `a_cat` / `c_cat` (double-buffer or wavefront).
"""


@triton.jit
def _gemm_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    expert_offsets_ptr,     # [Num_experts+1]，前缀和偏移，表示每个expert在 a_cat/c_cat 中的行区间
    expert_id,
    tile_m,
    tile_n,
    k_size,
    n_size,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """One output tile for one expert."""
    expert_m_start = tl.load(expert_offsets_ptr + expert_id).to(tl.int32)
    expert_m_end = tl.load(expert_offsets_ptr + expert_id + 1).to(tl.int32)
    expert_m = expert_m_end - expert_m_start      # 当前专家在拼接矩阵中占多少行

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)    # 当前tile在M维覆盖哪些行
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)    # 当前tile在N维覆盖哪些列
    offs_k = tl.arange(0, BLOCK_K)    #  K方向索引

    # 构造 A 子块 （BLOCK_M, BLOCK_N）的每个元素的地址矩阵
    a_ptrs = a_ptr + (expert_m_start + offs_m[:, None]) * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + expert_id * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_iter = 0
    while k_iter < k_size:
        a_mask = (offs_m[:, None] < expert_m) & ((k_iter + offs_k[None, :]) < k_size)
        b_mask = ((k_iter + offs_k[:, None]) < k_size) & (offs_n[None, :] < n_size)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    if IS_BF16:
        c = acc.to(tl.bfloat16)
    else:
        c = acc.to(tl.float16)
    c_ptrs = c_ptr + (expert_m_start + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < expert_m) & (offs_n[None, :] < n_size)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _persistent_grouped_gemm_kernel_striped(
    a_ptr,
    b_ptr,
    c_ptr,
    work_expert_ptr,
    work_tile_m_ptr,
    work_tile_n_ptr,
    expert_offsets_ptr,
    num_work_items,
    k_size,
    n_size,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    worker_id = tl.program_id(axis=0)
    num_workers = tl.num_programs(axis=0)
    work_id = worker_id
    while work_id < num_work_items:
        expert_id = tl.load(work_expert_ptr + work_id).to(tl.int32)
        tile_m = tl.load(work_tile_m_ptr + work_id).to(tl.int32)
        tile_n = tl.load(work_tile_n_ptr + work_id).to(tl.int32)
        _gemm_tile(
            a_ptr,
            b_ptr,
            c_ptr,
            expert_offsets_ptr,
            expert_id,
            tile_m,
            tile_n,
            k_size,
            n_size,
            stride_am,
            stride_ak,
            stride_be,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            IS_BF16,
        )
        work_id += num_workers


@triton.jit
def _persistent_grouped_gemm_kernel_atomic(
    a_ptr,                   # a_cat, [M, K] 输入指针
    b_ptr,                   # b_expert, [E, K, N] 专家权重指针
    c_ptr,                   # c_cat, [M, N] 输出指针
    work_expert_ptr,         # 任务队列指针，[num_work]，每个元素是expert_id
    work_tile_m_ptr,         # 任务队列指针，[num_work]，每个元素是tile_m，即输出矩阵在M维度的tile编号
    work_tile_n_ptr,         # 任务队列指针，[num_work]，每个元素是tile_n，即输出矩阵在N维度的tile编号
    expert_offsets_ptr,      # 专家偏移量指针, [E+1],前缀和偏移
    num_work_items,          # 总任务数，也就是线性任务队列的长度
    k_size,             # GEMM 的K维大小
    n_size,             # GEMM 的输出列数，也就是b_expert的N维度大小
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    counter_ptr,                   # 全局原子计数器指针，用于动态取任务
    NUM_THREADS: tl.constexpr,     # 每个block中的线程数，也就是warp数 * 每个warp中的线程数
    BLOCK_M: tl.constexpr,         # tile 的M维大小
    BLOCK_N: tl.constexpr,         # tile 的N维大小
    BLOCK_K: tl.constexpr,         # tile 的K维大小
    IS_BF16: tl.constexpr,
    MAX_ITERS: tl.constexpr,       # 最大迭代次数，用于防止死循环
):
    """
    Persistent workers dequeue tile indices with a global atomic (fetch-add).
    Only thread 0 of each CTA performs the masked RMW; other lanes must not feed
    undefined `raw` into a reduction (e.g. `tl.max`), so we broadcast via `tl.sum`
    over lanes that are zero except the leader.

    Uses a dynamic `while` instead of `tl.static_range(MAX_ITERS)` so large
    `num_work` does not compile thousands of unrolled copies of this body.
    """
    tid = tl.arange(0, NUM_THREADS)     # [NUM_THREADS]数组，生成一个从 0 到 NUM_THREADS-1 的整数数组
    leader = tid == 0     # 布尔数组，[NUM_THREADS],true表示0号线程
    # Scalar global counter as a per-lane pointer block (same address, offset 0).
    zero_off = tl.zeros((NUM_THREADS,), tl.int32)
    counter_addrs = counter_ptr + zero_off   # [NUM_THREADS]数组，初始化warp内每个线程的计数器地址，为了后面的mask，只让0号线程执行原子操作
    cont = tl.full((), 1, tl.int32)     # 标量，初始化为 1
    guard = tl.full((), 0, tl.int32)    # 标量，初始化为 0
    while cont != 0:
        inc = tl.where(leader, tl.full((NUM_THREADS,), 1, tl.int32), tl.zeros((NUM_THREADS,), tl.int32))  # 把 leader 数组从布尔数组变成 0、1 数组
        raw = tl.atomic_add(counter_addrs, inc, mask=leader, sem="relaxed")    #  原子操作，计数器地址相同，+1返回旧值
        work_id = tl.sum(tl.where(leader, raw.to(tl.int32), tl.zeros((NUM_THREADS,), tl.int32)))  # 把raw数组（比如[7, 0, 0, 0]）变成标量 work_id = 7
        sentinel = work_id >= num_work_items     # 布尔标量，看当前任务计数器是否已经超过总任务数
        if work_id < num_work_items:
            expert_id = tl.load(work_expert_ptr + work_id).to(tl.int32)  # 读出这个任务属于哪个expert
            tile_m = tl.load(work_tile_m_ptr + work_id).to(tl.int32)  # 这个任务的输出矩阵在M维方向的tile索引
            tile_n = tl.load(work_tile_n_ptr + work_id).to(tl.int32)  # 这个任务的输出矩阵在N维方向的tile索引
            _gemm_tile(          #  执行一个tile的GEMM，并把结果写回到c_ptr, 只算一个(BLOCK_M, BLOCK_N)的小块
                a_ptr,
                b_ptr,
                c_ptr,
                expert_offsets_ptr, 
                expert_id,
                tile_m,
                tile_n,
                k_size,
                n_size,
                stride_am,
                stride_ak,
                stride_be,
                stride_bk,
                stride_bn,
                stride_cm,
                stride_cn,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                IS_BF16,
            )
        guard = guard + 1     # 迭代计数器 +1， 记录已经循环多少次
        exit_now = sentinel.logical_or(guard >= MAX_ITERS)   # 满足任意条件就退出循环： (1)work_id >= num_work_items 任务取完 (2) guard >= MAX_ITER 循环次数达到上限
        cont = tl.where(exit_now, tl.full((), 0, tl.int32), tl.full((), 1, tl.int32))  # 要退出： cont = 0； 继续循环: cont = 1


# 把每个 expert 矩阵乘 拆成一堆 tile 任务，组成一个线性任务队列，用于后续调度
def _build_work_queue(
    expert_offsets: torch.Tensor,    # 专家编号偏移量[E+1]，表示每个expert在 a_cat 中的行区间
    n_size: int,                     # 输出矩阵的 N 维度大小
    block_m: int,                    # tile 的 M 维大小
    block_n: int,                    # tile 的 N 维大小
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert expert_offsets.dim() == 1
    num_experts = expert_offsets.numel() - 1
    work_expert = []
    work_tile_m = []
    work_tile_n = []
    for expert_id in range(num_experts):   # 遍历每个 expert, 都拆成tile任务
        #  当前专家的token数, 输出矩阵的 M 维度大小
        m_size = int((expert_offsets[expert_id + 1] - expert_offsets[expert_id]).item())
        if m_size <= 0:
            continue
        tiles_m = (m_size + block_m - 1) // block_m   # 当前专家的token数，能拆成多少个 tile_m
        tiles_n = (n_size + block_n - 1) // block_n   # 当前专家的token数，能拆成多少个 tile_n
        for tm in range(tiles_m):
            for tn in range(tiles_n):
                work_expert.append(expert_id)   # 
                work_tile_m.append(tm)
                work_tile_n.append(tn)
    device = expert_offsets.device
    return (
        torch.tensor(work_expert, device=device, dtype=torch.int32),
        torch.tensor(work_tile_m, device=device, dtype=torch.int32),
        torch.tensor(work_tile_n, device=device, dtype=torch.int32),
    )


def expert_offsets_from_recv_counts(
    num_recv_tokens_per_expert: Union[torch.Tensor, List[int]],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """
    Build `expert_offsets` of shape [E+1] from per-expert token counts (DeepEP-style).

    After dispatch, rows in `recv_x` for local experts should be concatenated in
    expert order so that expert `e` occupies rows [offsets[e], offsets[e+1]).

    Args:
        num_recv_tokens_per_expert: length E (local experts), int tensor or list
        device: if tensor is CPU and you need GPU offsets, set target device
    """
    if isinstance(num_recv_tokens_per_expert, list):
        t = torch.tensor(num_recv_tokens_per_expert, dtype=torch.int32)
    else:
        t = num_recv_tokens_per_expert.to(torch.int32).flatten()
    if device is not None:
        t = t.to(device)
    zeros = torch.zeros(1, dtype=torch.int32, device=t.device)
    cum = torch.cumsum(t, dim=0)
    return torch.cat([zeros, cum], dim=0).to(dtype=dtype)


def persistent_grouped_gemm(
    a_cat: torch.Tensor,               # [M, K] 输入矩阵A，各个expert的token按expert顺序拼接后的矩阵
    b_expert: torch.Tensor,            # [E, K, N] 权重矩阵B，表示每个expert一套 K*N 的权重
    expert_offsets: torch.Tensor,      # [E+1]前缀和偏移，表示每个expert在 a_cat 中的行区间
    num_sms: int = 24,                 # 持久化 worker 数(kernel grid size)
    block_m: int = 128,                # tile 的 M 维大小
    block_n: int = 128,                # tile 的 N 维大小
    block_k: int = 32,                 # tile 的 K 维分块步长
    use_atomic_queue: bool = True,    # 是否启用原子全局队列“动态取任务”（默认开启）
    work_counter: Optional[torch.Tensor] = None,    # 原子计数器，是任务队列的取号器，在 use_atomic_queue = True时，每个block都会做一次 atomic_ass(counter, 1), 拿到唯一的work_id，去处理对应的 tile
) -> torch.Tensor:
    """
    Compute grouped GEMM with persistent workers (striped or atomic dequeue).

    Args:
        use_atomic_queue: if True, use global atomic counter for dynamic tile assignment.
        work_counter: optional int32 tensor shape [1]; reset to 0 by this function when
            use_atomic_queue is True. If None, a temporary buffer is allocated.
    """
    if a_cat.device.type != "cuda" or b_expert.device.type != "cuda":
        raise ValueError("persistent_grouped_gemm requires CUDA tensors.")
    if a_cat.dim() != 2 or b_expert.dim() != 3:
        raise ValueError("Shapes must be a_cat [M, K], b_expert [E, K, N].")
    if a_cat.size(1) != b_expert.size(1):
        raise ValueError("K dimension mismatch between a_cat and b_expert.")
    if expert_offsets.numel() != b_expert.size(0) + 1:
        raise ValueError("expert_offsets must have shape [num_experts + 1].")
    if expert_offsets.device != a_cat.device:
        raise ValueError("expert_offsets must be on the same device as input tensors.")
    if a_cat.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("a_cat dtype must be bf16/fp16.")
    if b_expert.dtype != a_cat.dtype:
        raise ValueError("b_expert dtype should match a_cat dtype.")

    m_total, k_size = a_cat.shape
    n_size = b_expert.shape[2]
    c_cat = torch.empty((m_total, n_size), device=a_cat.device, dtype=a_cat.dtype)

    work_expert, work_tile_m, work_tile_n = _build_work_queue(
        expert_offsets, n_size=n_size, block_m=block_m, block_n=block_n
    )
    if work_expert.numel() == 0:
        c_cat.zero_()
        return c_cat

    is_bf16 = a_cat.dtype == torch.bfloat16
    num_warps = 8                      #  8个warp，每个warp有32个线程
    num_threads = num_warps * 32
    grid = (num_sms,)
    expert_off_i32 = expert_offsets.to(torch.int32)
    num_work = work_expert.numel()     # 总任务数，也就是线性任务队列的长度

    if use_atomic_queue:
        if work_counter is None:
            work_counter = torch.zeros(1, device=a_cat.device, dtype=torch.int32)
        else:
            if work_counter.numel() != 1 or work_counter.dtype != torch.int32:
                raise ValueError("work_counter must be int32 tensor of shape [1].")
            work_counter = work_counter.to(a_cat.device)
        work_counter.zero_()
        # Each Block may dequeue many tiles; upper bound per Block <= total tiles (+ slack).
        max_iters = min(65536, int(num_work) + int(num_sms) + 32)
        _persistent_grouped_gemm_kernel_atomic[grid](
            a_cat,
            b_expert,
            c_cat,
            work_expert,
            work_tile_m,
            work_tile_n,
            expert_off_i32,
            num_work,
            k_size,
            n_size,
            a_cat.stride(0),
            a_cat.stride(1),
            b_expert.stride(0),
            b_expert.stride(1),
            b_expert.stride(2),
            c_cat.stride(0),
            c_cat.stride(1),
            work_counter,
            NUM_THREADS=num_threads,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            IS_BF16=is_bf16,
            MAX_ITERS=max_iters,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        _persistent_grouped_gemm_kernel_striped[grid](
            a_cat,
            b_expert,
            c_cat,
            work_expert,
            work_tile_m,
            work_tile_n,
            expert_off_i32,
            num_work,
            k_size,
            n_size,
            a_cat.stride(0),
            a_cat.stride(1),
            b_expert.stride(0),
            b_expert.stride(1),
            b_expert.stride(2),
            c_cat.stride(0),
            c_cat.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            IS_BF16=is_bf16,
            num_warps=num_warps,
            num_stages=3,
        )
    return c_cat


def reference_grouped_gemm(
    a_cat: torch.Tensor, b_expert: torch.Tensor, expert_offsets: torch.Tensor
) -> torch.Tensor:
    out = torch.empty((a_cat.size(0), b_expert.size(2)), device=a_cat.device, dtype=a_cat.dtype)
    num_experts = b_expert.size(0)
    for e in range(num_experts):
        s = int(expert_offsets[e].item())
        t = int(expert_offsets[e + 1].item())
        if t <= s:
            continue
        out[s:t] = a_cat[s:t] @ b_expert[e]
    return out