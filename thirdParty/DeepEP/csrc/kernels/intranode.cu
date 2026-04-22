#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace deep_ep {

namespace intranode {

// 总体目标：计算"谁发给谁多少"的通信计划
// 1. 每个GPU要发送给其他GPU多少Token
// 2. 每个GPU的每个专家将收到多少Token
// 3. 为多通道并行通信预分配工作负载
template <int kNumRanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,   // const输入参数：长度为 kNumRanks 的数组，当前 rank 发给各个 target rank 的 token 数
                                int* moe_recv_counter_mapped,   // 待写入：当前 rank 的moe接收端收到的token总数，int标量；Host映射内存
                                const int* num_tokens_per_expert,  // const输入参数：长度为 num_experts 的数组，当前 rank 发给各个全局 expert 的 token 数
                                int* moe_recv_expert_counter_mapped,  // 待写入：当前 rank 每个本地 expert 收到的 token 数量
                                int num_experts,   // 全局 expert 数
                                int num_tokens,    // 本地 rank 持有的 token 数
                                int num_channels,  // 并行通信的 channel 数
                                const bool* is_token_in_rank,   // 本地 token i 是否需要发往 rank j，布尔矩阵[num_tokens, kNumRanks]
                                int* channel_prefix_matrix,
                                int* rank_prefix_matrix_copy,  // (3)存“谁发给谁多少”的完整8*8矩阵，后续dispatch阶段靠它计算偏移量
                                int num_memset_int,
                                int expert_alignment,  //  每个 expert 接收的token数要向上对齐到 expert_alignment 的整数倍
                                void** buffer_ptrs,       // (2)指针数组，buffer_ptrs[3]指向的是3号卡的显存缓冲区，有了这个GPU才能通过nvlink将“发货清单”写到别的rank上
                                int** barrier_signal_ptrs,  // (2)用于机内8张卡的同步，大家写完清单后，通过这些指针，通过这些指针指向的内存位“打卡”，确认全机箱写完了
                                int rank) {   // (2)当前 GPU 的机内编号
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;

    if (sm_id == 0) {    // 全局统计与同步，多个SM并发访问远程GPU显存会产生竞争，因此一个SM足够

        // Step 1: 填写通信矩阵，per_rank_buffer + per_expert_buffer
        // Barrier first
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);  // 同步机制(三层屏障)：1. 启动屏障，确保所有GPU都启动kernel并到达同步点

        /*   
            双buffer布局，rank-to-rank 数量矩阵 + rank-to-local_expert 数量矩阵
            +------------------------+------------------------+
            | per_rank_buffer        | per_expert_buffer     |
            | [kNumRanks][kNumRanks] | [kNumRanks][experts]  |
            +------------------------+------------------------+  
        */
        int *per_rank_buffer, *per_expert_buffer;
        if (thread_id < kNumRanks) {    
            per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);     // buffer_ptrs 指向所有GPU显存的指针数组
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
        }

        // After this loop:
        //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
        //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks) {
            per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];  // 只填一行，rank行 所有列
            #pragma unroll
            for (int i = 0; i < num_experts_per_rank; ++i)
                per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[thread_id * num_experts_per_rank + i];
        }

        // Wait for all ranks to be finished       
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);    // 同步机制(三层屏障)：2. 数据就绪屏障，等待所有GPU完成通信矩阵的写入

        // Step 2: 根据自己通信矩阵，计算前缀和 local_per_rank_buffer + local_per_expert_buffer
        // Sum per-rank counts and return to CPU
        // Also pre-compute the prefix sum for data sending
        auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
        if (thread_id < kNumRanks) {
            #pragma unroll
            for (int i = 1; i < kNumRanks; ++i)
                local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
            if (thread_id == rank)
                *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];  // 复制，当前rank收多少token
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
        if (thread_id < num_experts_per_rank) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            moe_recv_expert_counter_mapped[thread_id] = sum;                  // 复制，当前rank的每一个expert收多少token
        }
        __syncthreads();

        // Copy rank size prefix matrix to another tensor
        #pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
            rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];  // 完整地把前缀和矩阵拷贝下来（dispatch要用），存副本。因为local_per_rank_buffer这个区域可能在下一个kernel启动时被复用，要避免race condition

        // Extra memset for later communication queue
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            local_per_expert_buffer[i] = 0;   // expert rank需要及时清0，避免race condition（因为我们已经记录了每个local expert的接收token数moe_recv_expert_counter_mapped，因此expert buffer可以清零）

        // Barrier
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);    // 同步机制(三层屏障)：2. 清理屏障, 确保所有GPU完成缓冲区清理，避免后续kernel的竞争
    } else {    // 其他 block 在为接下来的多channel并发搬运， 计算每个通道的offset
        int dst_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {   // 一个warp领取一个channel，
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);   // 每个channel负责一部分token的通信，比如4096个token，4个channel，每个channel负责1024个token

            // Iterate over tokens
            int count = 0;
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
                count += is_token_in_rank[i * kNumRanks + dst_rank];         // warp用所有线程统计自己通道内要发送给目标rank的token数
            count = warp_reduce_sum(count);
            if (elect_one_sync())
                channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
        }
        __syncthreads();

        // Pre-compute prefix sum for all channels
        if (thread_id == 0) {
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
        }
    }
}

//          MoE的异步通信中，最难的不是搬数据，而是确定每个数据该搬达到对面的哪个精确的位置
// Host 端轻量级 kernel. 发起notify_dispatch
void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,       // (1)计算channel_prefix_matrix, 告诉当前GPU每个channel(warp)，我要发的这批token应该在目标gpu的buffer的什么offset开始写
                     int* rank_prefix_matrix_copy,
                     int num_memset_int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)        \
    LAUNCH_KERNEL(&cfg,                           \
                  notify_dispatch<ranks>,         \
                  num_tokens_per_rank,            \
                  moe_recv_counter_mapped,        \
                  num_tokens_per_expert,          \
                  moe_recv_expert_counter_mapped, \
                  num_experts,                    \
                  num_tokens,                     \
                  num_channels,                   \
                  is_token_in_rank,               \
                  channel_prefix_matrix,          \
                  rank_prefix_matrix_copy,        \
                  num_memset_int,                 \
                  expert_alignment,               \
                  buffer_ptrs,                    \
                  barrier_signal_ptrs,            \
                  rank);                          \
    break

    constexpr int kNumThreads = 128;      //  一个线程块128个线程
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

    SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_dispatch(
    const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
    // A simplified version for cached handles
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    // Copy and clean
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto ptr = static_cast<int*>(buffer_ptrs[rank]);
    #pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
        ptr[i] = rank_prefix_matrix[i];
    #pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads)
        ptr[kNumRanks * kNumRanks + i] = 0;

    // Barrier after cleaning
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** barrier_signal_ptrs,
                            int rank,
                            int num_ranks,
                            cudaStream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                                   \
    LAUNCH_KERNEL(&cfg, cached_notify_dispatch<ranks>, rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 128, stream);
    SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(int4* recv_x,
                                                           float* recv_x_scales,
                                                           int* recv_src_idx,
                                                           topk_idx_t* recv_topk_idx,
                                                           float* recv_topk_weights,
                                                           int* recv_channel_offset,
                                                           int* send_head,
                                                           const int4* x,
                                                           const float* x_scales,
                                                           const topk_idx_t* topk_idx,
                                                           const float* topk_weights,
                                                           const bool* is_token_in_rank,
                                                           const int* channel_prefix_matrix,
                                                           int num_tokens,
                                                           int num_worst_tokens,
                                                           int hidden_int4,
                                                           int num_topk,
                                                           int num_experts,
                                                           int num_scales,
                                                           int scale_token_stride,
                                                           int scale_hidden_stride,
                                                           void** buffer_ptrs,
                                                           int rank,
                                                           int num_max_send_tokens,
                                                           int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    // Several warps are response for a single rank
    const auto num_threads_per_rank = kNumThreads / kNumRanks;
    const auto num_channels = num_sms / 2;
    const auto responsible_rank = (static_cast<int>(thread_id)) / num_threads_per_rank;
    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    const auto responsible_channel = sm_id / 2;

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    // Calculate pointers by the specific layout
    // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) +
                                       kNumRanks * kNumRanks * sizeof(int));
    int target_rank = is_sender ? rank : responsible_rank;
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    // Channel buffer metadata
    // Senders are responsible for tails, and receivers are responsible for heads
    // Stored on the receiver side
    // The retired signals are actually boolean flags, but to align with 16 bytes, we make it `int64_t`
    // `start_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

    // Channel data buffers, stored on the receiver side
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
    // `topk_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(topk_idx_t)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    // `x_scales_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_scales * sizeof(float)
    auto channel_x_buffers = Buffer<int4>(     //  token 本身
        ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers =             //  token 的编号，原始序列中的位置
        Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_idx_buffers = Buffer<topk_idx_t>(   //  token 选择的专家编号
        ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_topk_weights_buffers =        //  token 对应的专家权重
        Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_x_scales_buffers = Buffer<float>(     //  反量化需要的 scale
        ptr, num_channels_total * num_recv_buffer_tokens * num_scales, channel_rank_offset * num_recv_buffer_tokens * num_scales);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto half_hidden_int4 = hidden_int4 / 2;
    auto half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);
    uint32_t tma_phase = 0;
    if (elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();
#endif

    if (is_sender) {
        // Workers for sending
        constexpr int num_send_warps = kNumThreads / 32;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        const auto send_thread_id = thread_id;
        const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        if (send_warp_id_in_rank == 0 and elect_one_sync()) {
            int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
            st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
            value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
            st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
        }
        __syncwarp();

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        // Iterate over all tokens and send by chunks
        int cached_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            // NOTES: the head index received by different warps may not be the same
            auto start_time = clock64();
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for dispatch senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            int chunk_token_idx = 0;
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                // NOTES: for the same token, the warp assigned to save `send_head` may be different from the warp assigned to send the
                // following data
                if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
                    send_head[token_idx * kNumRanks + responsible_rank] =
                        is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;

                // Skip if not selected
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                // Get an empty slot
                int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    auto shifted_x = x + token_idx * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);

                    // Copy source index
                    if (elect_one_sync())
                        channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    if (lane_id < num_topk) {
                        // Top-k index
                        int recv_expert_begin = responsible_rank * num_experts_per_rank,
                            recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
                        auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
                        channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                        // Top-k weights
                        auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                        channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = weight_value;
                    }

                    // Copy `x_scales`
                    #pragma unroll
                    for (int i = lane_id; i < num_scales; i += 32) {
                        auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                        channel_x_scales_buffers[dst_slot_idx * num_scales + i] = __ldg(x_scales + offset);
                    }
                }

                // Move token index
                chunk_token_idx++, token_idx++;
            }

            // Move tail index
            // NOTES: here all warps should share the same new tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
        }
    } else {
        // Workers for receiving and copying into buffer
        constexpr int num_recv_warps = kNumThreads / 32;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto recv_thread_id = thread_id;
        const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

        // Calculate offset first
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;

        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
                ;
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
                ;
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            if (recv_warp_id_in_rank == 0)
                recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        // Shared tail indices for different warps
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];

        auto start_time = clock64();
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // NOTES: unlike the sender, the receiver must ensure that the tail indices hold by different warps are the same
            while (recv_thread_id_in_rank == 0) {
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());

                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx) {
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                           rank,
                           responsible_channel,
                           num_tokens_to_recv);
                    trap();
                }
            }

            // Synchronize queue tail
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
                auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
#ifndef DISABLE_SM90_FEATURES
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    tma_store_wait<0>();
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer, shifted_buffer_x_int4 + i * half_hidden_int4, tma_mbarrier, half_hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                        mbarrier_wait(tma_mbarrier, tma_phase);
                        tma_store_1d(tma_buffer, shifted_recv_x_int4 + i * half_hidden_int4, half_hidden_bytes, false);
                    }
                }
                __syncwarp();
#else
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4, shifted_buffer_x_int4, ld_nc_global, st_na_global);
#endif
            }

            // Copy `src_idx`
            #pragma unroll 4
            for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
                 chunk_idx += 32 * num_recv_warps_per_rank)
                recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
                    ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);

            // Copy `topk_idx` and `topk_weights`
            #pragma unroll 4
            for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += 32 * num_recv_warps_per_rank) {
                int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;
                auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
                recv_topk_idx[recv_idx] = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
                recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
            }

            // Copy `x_scales`
            #pragma unroll 4
            for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += 32 * num_recv_warps_per_rank) {
                int chunk_idx = i / num_scales, scales_idx = i % num_scales;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
                    ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + scales_idx);
            }

            // Move queue
            cached_channel_head_idx += num_recv_tokens;
            total_offset += num_recv_tokens;
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

            // Exit
            num_tokens_to_recv -= num_recv_tokens;
        }
    }

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        const auto num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
        const auto clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_sms * kNumThreads;
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;
    }
}

// Host端启动器，负责为真正 dispatch Kernel 准备弹药并完成launch
//   根据 gating 结果，把本地的token扔到机内对应的专家手里
//   参数分类：(1)接收端； (2)发送端; (3)路由与元数据(导航); (4)环境配置

void dispatch(void* recv_x,                     // (1)核心负载: 接收到的token特征向量 (Hidden States)
              float* recv_x_scales,             // (1)量化缩放：如果用FP8/INT8，这是对应的Scale
              int* recv_src_idx,                // (1)溯源索引：记录这个token原来在哪个rank的哪个位置（用于最后combine）
              topk_idx_t* recv_topk_idx,        // (1)token对应专家的ID
              float* recv_topk_weights,         // (1)token对应的Gating权重
              int* recv_channel_offset,         // (1)接收计数器：记录当前每个channel已经写了多少数据
              int* send_head,                   // (1)发送进度：用于多线程同步，标记当前发送到了什么位置
              const void* x,                            // (2)本地原始token
              const float* x_scales,                    // (2)本地token的量化scale
              const topk_idx_t* topk_idx,               // (2)Gating网络算出来的路由结果
              const float* topk_weights,                // (2)Gating网络算出来的路由权重
              const bool* is_token_in_rank,             // (2)路由掩码：bool矩阵，标记 token i 是否要去 rank j
              const int* channel_prefix_matrix,                 // (3)两级前缀和账本，记录每个rank每个channel的起始偏移量，kernel靠它实现无锁写入
              int num_tokens,                                           // (4)当前GPU上token数               
              int num_worst_tokens,                                     // (4)
              int hidden_int4,                                          // (4)隐藏层维度(Hidden Dim), 以 int4 (16 bytes)为单位，为了适配向量化搬运
              int num_topk,                                             // (4)top-K个专家
              int num_experts,
              int num_scales,
              int scale_token_stride,
              int scale_hidden_stride,
              void** buffer_ptrs,                               // (3)UVA的灵魂：一个数组指针，存储全机房所有GPU接收缓冲区的地址
              int rank,
              int num_ranks,
              cudaStream_t stream,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens) {
    constexpr int kNumThreads = 768;                //  每个 threadblock 中的线程总数，768/32 = 24个warps. （H100/A100每个SM承载的线程上限是2048个，所以当前每个SM可以放下2个block，不设置成每个block有1024个现成可能是寄存器/shared memory不够用了）. Dispatch是一个通信密集的kernel，大部分时间都在等数据从显存搬到SM，或者等数据从NVLink传过来，24个warps提供了足够多的warp，一部分等在数据而挂起，另一部分可以去执行计算（latency hiding）。但是如果warp太多，调度器本身的切换开销很大
    constexpr int kNumTMABytesPerWarp = 8192;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);   // H100上会指定分配shared memory大小为192KB
#endif

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                      \
    {                                                                    \
        auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                               \
        LAUNCH_KERNEL(&cfg,                                              \
                      kernel,                                            \
                      reinterpret_cast<int4*>(recv_x),                   \
                      recv_x_scales,                                     \
                      recv_src_idx,                                      \
                      recv_topk_idx,                                     \
                      recv_topk_weights,                                 \
                      recv_channel_offset,                               \
                      send_head,                                         \
                      reinterpret_cast<const int4*>(x),                  \
                      x_scales,                                          \
                      topk_idx,                                          \
                      topk_weights,                                      \
                      is_token_in_rank,                                  \
                      channel_prefix_matrix,                             \
                      num_tokens,                                        \
                      num_worst_tokens,                                  \
                      hidden_int4,                                       \
                      num_topk,                                          \
                      num_experts,                                       \
                      num_scales,                                        \
                      scale_token_stride,                                \
                      scale_hidden_stride,                               \
                      buffer_ptrs,                                       \
                      rank,                                              \
                      num_max_send_tokens,                               \
                      num_recv_buffer_tokens);                           \
    }                                                                    \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_combine(
    void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens, int num_memset_int, int** barrier_signal_ptrs, int rank) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    if (sm_id == 0) {
        // Barrier before cleaning
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        // Clean
        auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
        auto ptr = static_cast<int*>(buffer_ptrs[rank]);
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)
            ptr[i] = 0;

        // Barrier after cleaning
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {
        const auto channel_id = sm_id - 1;
        const auto thread_id = static_cast<int>(threadIdx.x);
        const auto rank_id = thread_id / 32;
        const auto lane_id = thread_id % 32;
        if (rank_id >= kNumRanks)
            return;

        int token_start_idx, token_end_idx;
        get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // NOTES: `1 << 25` is a heuristic large number
        int last_head = 1 << 25;
        #pragma unroll
        for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
            int token_idx = token_idx_tail - lane_id, expected_head = 0;
            auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;
            for (int i = 0; i < min(32, token_idx_tail - token_start_idx + 1); ++i) {
                const int head = __shfl_sync(0xffffffff, current_head, i);
                if (head < 0) {
                    if (lane_id == i)
                        expected_head = -last_head - 1;
                } else {
                    last_head = head;
                }
            }
            if (current_head < 0 and token_idx >= token_start_idx)
                send_head[token_idx * kNumRanks + rank_id] = expected_head;
        }
    }
}

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           cudaStream_t stream) {
#define CACHED_NOTIFY_COMBINE(ranks)            \
    LAUNCH_KERNEL(&cfg,                         \
                  cached_notify_combine<ranks>, \
                  buffer_ptrs,                  \
                  send_head,                    \
                  num_channels,                 \
                  num_recv_tokens,              \
                  num_memset_int,               \
                  barrier_signal_ptrs,          \
                  rank);                        \
    break

    const int num_threads = std::max(128, 32 * num_ranks);
    EP_HOST_ASSERT(num_ranks <= num_threads);
    EP_HOST_ASSERT(num_threads <= 1024);
    EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);
    SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
    SWITCH_RANKS(CACHED_NOTIFY_COMBINE);
#undef CACHED_NOTIFY_COMBINE
}

template <typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) combine(dtype_t* recv_x,
                                                          float* recv_topk_weights,
                                                          const dtype_t* x,
                                                          const float* topk_weights,
                                                          const dtype_t* bias_0,
                                                          const dtype_t* bias_1,
                                                          const int* src_idx,
                                                          const int* rank_prefix_matrix,
                                                          const int* channel_prefix_matrix,
                                                          int* send_head,
                                                          int num_tokens,
                                                          int num_recv_tokens,
                                                          int hidden,
                                                          int num_topk,
                                                          void** buffer_ptrs,
                                                          int rank,
                                                          int num_max_send_tokens,
                                                          int num_recv_buffer_tokens) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto sm_id = static_cast<int>(blockIdx.x), lane_id = get_lane_id();
    const auto num_channels = num_sms / 2;
    const bool is_sender = sm_id % 2 == 0;
    const int responsible_channel = sm_id / 2;
    EP_DEVICE_ASSERT(num_topk <= 32);

    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
    int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
    int hidden_int4_aligned = align_down(hidden_int4, 32);
    auto x_int4 = reinterpret_cast<const int4*>(x);
    auto bias_0_int4 = reinterpret_cast<const int4*>(bias_0);
    auto bias_1_int4 = reinterpret_cast<const int4*>(bias_1);
    auto recv_int4 = reinterpret_cast<int4*>(recv_x);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
#endif

    if (is_sender) {
        // Workers for sending
        // Several warps are responsible for a single rank
        constexpr int num_send_warps_per_rank = (kNumThreads / 32) / kNumRanks;
        constexpr int num_send_warps = num_send_warps_per_rank * kNumRanks;
        const auto num_threads_per_rank = num_send_warps_per_rank * 32;
        const auto send_thread_id = thread_id;
        const auto send_warp_id = send_thread_id / 32;
        const auto send_rank_id = (responsible_channel + send_warp_id) % kNumRanks;
        const auto send_warp_id_in_rank = send_warp_id / kNumRanks;
        EP_STATIC_ASSERT(num_send_warps * 32 == kNumThreads, "Invalid warp count");

        // Calculate pointers by the specific layout
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[send_rank_id]));
        auto num_channels_total = num_channels * kNumRanks;
        auto channel_rank_offset = responsible_channel * kNumRanks + rank;

        // Channel meta data
        // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_x_buffers = Buffer<int4>(
            ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
        auto channel_src_idx_buffers =
            Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
        auto channel_topk_weights_buffers = Buffer<float>(
            ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);

        // Get tasks
        // NOTES: `channel_offset` is already shifted
        int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;
        int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
        int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];
        int num_channel_tokens =
            (responsible_channel == num_channels - 1 ? num_rank_tokens
                                                     : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1]) -
            channel_offset;
        int token_start_idx = rank_offset + channel_offset, token_end_idx = rank_offset + channel_offset + num_channel_tokens;

        // Iterate over all tokens and send by chunks
        int current_channel_tail_idx = 0;
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            auto start_time = clock64();
            int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            // Send by chunk
            #pragma unroll
            for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
                // Get an empty slot
                int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;

                // Copy data
                auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                auto shifted_x = x_int4 + (token_idx + i) * hidden_int4;
                UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

                // Send source index
                if (elect_one_sync())
                    channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + token_idx + i);

                // Send `topk_weights`
                if (num_topk > 0 and lane_id < num_topk)
                    channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
                        __ldg(topk_weights + (token_idx + i) * num_topk + lane_id);
            }
            token_idx += num_round_tokens;
            current_channel_tail_idx += num_round_tokens;

            // Move tail index
            asm volatile("bar.sync %0, %1;" ::"r"(send_rank_id), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx);
        }
    } else {
        // Workers for receiving
        // One warp for moving the queue head, others for reduction
        constexpr int num_recv_warps = kNumThreads / 32;
        const auto recv_warp_id = thread_id / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32 and kNumThreads > 32);
        EP_DEVICE_ASSERT(thread_id >= 0 and kNumThreads % 32 == 0);

        // Shared head, tail and retired flags for receiver warps
        __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
        __shared__ volatile int channel_tail_idx[kNumRanks];
        __shared__ volatile bool warp_retired[num_recv_warps];
        if (thread_id < num_recv_warps)
            warp_retired[thread_id] = false;
        if (lane_id < kNumRanks)
            warp_channel_head_idx[recv_warp_id][lane_id] = 0;
        if (thread_id < kNumRanks)
            channel_tail_idx[thread_id] = 0;
        asm volatile("bar.sync 0, %0;" ::"r"(kNumThreads));

        if (thread_id < 32) {
            int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + lane_id;
            int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;

            // Queue head updater
            int last_head = 0;
            while (lane_id < kNumRanks) {
                // Check retired
                bool retired = true;
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    retired = retired and warp_retired[i];
                if (retired)
                    break;

                // Update queue tail
                channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);

                // Update minimum head
                int min_head = std::numeric_limits<int>::max();
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    if (not warp_retired[i])
                        min_head = min(min_head, warp_channel_head_idx[i][lane_id]);
                if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
                    st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);
            }
        } else {
            // Receivers
            // Channel metadata
            // All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`
            Buffer<int4> channel_x_buffers[kNumRanks];
            Buffer<float> channel_topk_weights_buffers[kNumRanks];

            // Calculate pointers by the specific layout
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i) {
                auto channel_rank_offset = responsible_channel * kNumRanks + i;
                auto num_channels_total = num_channels * kNumRanks;
                // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
                auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) + 2 * num_channels * kNumRanks * sizeof(int));

                // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
                channel_x_buffers[i] = Buffer<int4>(ptr,
                                                    num_channels_total * num_recv_buffer_tokens * hidden_int4,
                                                    channel_rank_offset * num_recv_buffer_tokens * hidden_int4);

                // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
                ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) + num_channels_total * num_recv_buffer_tokens * sizeof(int));

                // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
                channel_topk_weights_buffers[i] = Buffer<float>(
                    ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
            }

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            for (int64_t token_idx = token_start_idx + recv_warp_id - 1; token_idx < token_end_idx; token_idx += num_recv_warps - 1) {
                // Read expected head
                int expected_head = -1;
                if (lane_id < kNumRanks)
                    expected_head = ld_nc_global(send_head + token_idx * kNumRanks + lane_id);

                auto start_time = clock64();
                while (__any_sync(0xffffffff, channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) {
                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine receivers, rank %d, responsible_channel = %d, expect = %d\n",
                               rank,
                               responsible_channel,
                               expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Broadcast current heads
                int num_topk_ranks = 0, topk_ranks[kNumRanks], slot_indices[kNumRanks];
                #pragma unroll
                for (int i = 0; i < kNumRanks; ++i) {
                    auto expected_head_i = __shfl_sync(0xffffffff, expected_head, i);
                    if (expected_head_i >= 0) {
                        slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens;
                        topk_ranks[num_topk_ranks++] = i;
                    }
                }

                // Wait shared memory release
#ifndef DISABLE_SM90_FEATURES
                tma_store_wait<0>();
                __syncwarp();
#endif

                // Reduce data with pipeline
                constexpr int kNumStages = 8;
                EP_STATIC_ASSERT(kNumStages * 32 * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid count");
                #pragma unroll
                for (int i = lane_id; i < hidden_int4; i += 32) {
                    // Read bias
                    // TODO: make it as a template
                    int4 bias_0_value_int4 =
                        bias_0_int4 != nullptr ? __ldg(bias_0_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
                    int4 bias_1_value_int4 =
                        bias_1_int4 != nullptr ? __ldg(bias_1_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);

                    // Read buffers
                    int4 recv_value_int4[kNumRanks];
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j)
                        recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);

                    // Reduce bias
                    float values[kDtypePerInt4];
                    auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
                    auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);

                    // Reduce all-to-all results
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j) {
                        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                        #pragma unroll
                        for (int k = 0; k < kDtypePerInt4; ++k)
                            values[k] += static_cast<float>(recv_value_dtypes[k]);
                    }

                    // Cast back to `dtype_t`
                    int4 out_int4;
                    auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        out_dtypes[j] = static_cast<dtype_t>(values[j]);

#ifndef DISABLE_SM90_FEATURES
                    if (i < hidden_int4_aligned) {
                        // Wait TMA arrival
                        tma_store_wait<kNumStages - 1>();
                        __syncwarp();

                        // Write into TMA buffer
                        auto tma_stage_idx = (i / 32) % kNumStages;
                        reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * 32 + lane_id] = out_int4;

                        // Issue TMA
                        tma_store_fence();
                        __syncwarp();
                        if (elect_one_sync()) {
                            auto tma_bytes = min(32, hidden_int4 - i) * static_cast<int>(sizeof(int4));
                            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * 32,
                                         recv_int4 + token_idx * hidden_int4 + i,
                                         tma_bytes,
                                         false);
                        }
                        __syncwarp();
                    } else {
#endif
                        recv_int4[token_idx * hidden_int4 + i] = out_int4;
#ifndef DISABLE_SM90_FEATURES
                    }
#endif
                }

                // Reduce `topk_weights`
                if (lane_id < num_topk) {
                    float value = 0;
                    #pragma unroll
                    for (int i = 0; i < num_topk_ranks; ++i)
                        value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
                    recv_topk_weights[token_idx * num_topk + lane_id] = value;
                }

                // Update head
                if (lane_id < kNumRanks)
                    warp_channel_head_idx[recv_warp_id][lane_id] = (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
            }

            // Retired
            __syncwarp();
            if (elect_one_sync())
                warp_retired[recv_warp_id] = true;
        }
    }
}

void combine(cudaDataType_t type,
             void* recv_x,
             float* recv_topk_weights,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* src_idx,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 4096;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

#define COMBINE_LAUNCH_CASE(dtype, ranks)                                      \
    {                                                                          \
        auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                     \
        LAUNCH_KERNEL(&cfg,                                                    \
                      kernel,                                                  \
                      reinterpret_cast<dtype*>(recv_x),                        \
                      recv_topk_weights,                                       \
                      reinterpret_cast<const dtype*>(x),                       \
                      topk_weights,                                            \
                      reinterpret_cast<const dtype*>(bias_0),                  \
                      reinterpret_cast<const dtype*>(bias_1),                  \
                      src_idx,                                                 \
                      rank_prefix_matrix,                                      \
                      channel_prefix_matrix,                                   \
                      send_head,                                               \
                      num_tokens,                                              \
                      num_recv_tokens,                                         \
                      hidden,                                                  \
                      num_topk,                                                \
                      buffer_ptrs,                                             \
                      rank,                                                    \
                      num_max_send_tokens,                                     \
                      num_recv_buffer_tokens);                                 \
    }                                                                          \
    break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)                 \
    SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving
    EP_HOST_ASSERT(num_sms % 2 == 0);
    EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

}  // namespace intranode

}  // namespace deep_ep
