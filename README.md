# MoE Baseline for Thesis (A100 Single-Node Multi-GPU)

这个仓库提供一个适合毕设的 **baseline 起点**：

- 第一阶段（推荐先做）：自建轻量 MoE + 合成数据，快速验证吞吐、延迟、路由负载均衡
- 第二阶段：迁移到现成框架（Megatron/DeepSpeed）做工程化对照

## 为什么先用自建 + 合成数据

对于你的题目（DeepEP 通信 + Triton persistent grouped GEMM overlap），论文/答辩更需要：

- 能清楚拆分变量：通信、计算、路由分布、batch/token 形状
- 能做可重复的 ablation：无 overlap vs overlap，普通 GEMM vs grouped persistent
- 能快速迭代内核与调度，不被大框架训练栈绑定

所以 baseline 最好先是“可控实验平台”，再补“真实框架对照”。

## 当前提供的 baseline

`src/train_moe_baseline.py` 提供：

- Top-k router（默认 top-2）
- 简化版 Expert MLP
- 合成 token 输入
- 单机多卡 DDP 训练（可选）
- 每 step 的吞吐、router 负载统计

> 注意：当前版本是研究基线，不含 DeepEP 和 Triton kernel，仅用于建立对照性能与正确性指标。

`src/triton_grouped_gemm.py` 提供：

- Triton `persistent grouped GEMM` 前向 kernel（按 expert 分组）
- **Striped**：`num_sms` 个 CTA 按步长遍历预建 tile 队列（默认）
- **Atomic queue**（`use_atomic_queue=True`）：全局 `atomic_add` 取下一个 tile，便于负载不均时动态分配（论文里常泛称 work stealing；与 DeepEP overlap 搭配时在 **另一 stream** 上发通信）
- `expert_offsets_from_recv_counts`：从 DeepEP 风格的每 expert token 数构造 `expert_offsets`
- PyTorch 参考实现用于 correctness 对比

## 目录

- `src/train_moe_baseline.py`：主训练脚本
- `requirements.txt`：最小依赖

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行示例

### 单卡快速跑通

```bash
python src/train_moe_baseline.py \
  --steps 50 \
  --batch-size 8 \
  --seq-len 1024 \
  --hidden-size 1024 \
  --num-experts 8 \
  --top-k 2
```

### 多卡 DDP（4 卡示例）

```bash
torchrun --nproc_per_node=4 src/train_moe_baseline.py \
  --steps 200 \
  --batch-size 8 \
  --seq-len 2048 \
  --hidden-size 2048 \
  --num-experts 16 \
  --top-k 2 \
  --dtype bf16
```

### 使用 DeepEP 作为通信后端（intranode）

先在项目内编译 DeepEP（你已经放到 `thirdParty/DeepEP`）：

```bash
cd thirdParty/DeepEP
NVSHMEM_DIR=/path/to/nvshmem python setup.py build
```

然后回到项目根目录运行：

```bash
torchrun --nproc_per_node=8 src/train_moe_baseline.py \
  --steps 100 \
  --batch-size 8 \
  --seq-len 1024 \
  --hidden-size 2048 \
  --num-experts 16 \
  --top-k 2 \
  --dtype bf16 \
  --use-deepep \
  --deepep-num-sms 24
```

> `--use-deepep` 需要多卡分布式环境（`torchrun`）。

### Triton persistent grouped GEMM 基准

```bash
python src/bench_triton_grouped_gemm.py \
  --num-experts 16 \
  --tokens-per-expert 512 \
  --hidden-size 2048 \
  --ffn-size 4096 \
  --num-sms 24 \
  --dtype bf16
```

加 `--atomic-queue` 可测全局原子 dequeue 版本：

```bash
python src/bench_triton_grouped_gemm.py \
  --num-experts 8 \
  --tokens-per-expert 128 \
  --hidden-size 512 \
  --ffn-size 1024 \
  --num-sms 16 \
  --dtype fp16 \
  --atomic-queue
```

输出会包含：

- `max_abs_diff`：与 PyTorch 参考实现的误差
- `triton_ms` 与 `torch_loop_ms`：平均耗时对比

## 建议的毕设路线

1. 跑通当前 baseline，记录 step time / tokens per second / expert load std
2. 加入 MoE dispatch/combine 计时与 profile（Nsight Systems）
3. 替换通信路径为 DeepEP（先 correctness 再性能）
4. 替换 expert 计算为 Triton grouped GEMM（先 non-persistent，再 persistent）
5. 做 overlap 调度（多 stream + event）并和 baseline 对比

## 建议记录的核心指标

- `step_time_ms`
- `tokens_per_sec`
- `router_entropy`
- `expert_load_std` / `expert_load_cv`
- `dispatch_time_ms`、`expert_compute_time_ms`、`combine_time_ms`
- overlap 后的 `communication_hidden_ratio`

