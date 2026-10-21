# InScope HiddenLayer + Probing Tests

本目录存放 InScope 的本地研究型测试设计和运行入口。当前不属于 CI：需要本地 Qwen3 checkpoint、GPU、HF/SGLang 对照环境，且可能需要较长时间训练 probe。

## 目标

验证同一个 context 在两类位置选择方式下，模型指定 decoder block 的 representation 是否包含与目标假设有关的可读出信号：

1. 直接指定完整 input token 序列中的位置；
2. 追加 gauge suffix，并使用 suffix 最后一个 token 的位置。

测试链路：

```text
context
  + optional gauge suffix(hypothesis)
  -> tokenizer/chat template
  -> resolve absolute token positions
  -> SGLang prefill-only + HiddenLayer capture
  -> 指定位置的多层 representation
  -> 按 layer / hypothesis 训练 probe
  -> held-out test 上报告 AUROC 等指标
```

这里的 probe 是分析工具，不是安全决策器。测试结果只能说明表征中存在可利用信号，不能证明模型以人类意义理解了该属性，也不能证明生成时实际使用了该属性。

## 与 SafeGauge 的关系

SafeGauge 当前的核心测试链路是：

```text
context + assistant gauge suffix
  -> suffix token prompt log-probabilities
  -> BinaryMlp probe
```

InScope 的第一版测试保留它的实验思想、数据划分和 suffix 元数据，但把 feature 换成：

```text
[layer_id, suffix_token_position, hidden_dimension]
```

同一个样本可以：

- 不使用 suffix，直接指定 `positions`；
- 使用一个或多个 hypothesis suffix，并将每个 suffix 的最后 token 作为位置。

每个实验记录必须明确 `position_mode`：`explicit` 或 `suffix_last`。使用 suffix 时必须保存 `suffix_id`、token ids 和 span；不使用 suffix 时不得伪造这些字段。不同 hypothesis/suffix 的向量不能无标注地混在一起。

## 目录约定

```text
feature_capture/
├── README.md                         # 本说明
├── DESIGN.md                         # 完整实验设计、指标、验收条件
├── schemas/
│   ├── sample.schema.json            # context 数据格式
│   ├── feature_record.schema.json    # hidden feature 格式
│   └── probe_manifest.schema.json    # probe 与实验配置绑定
├── hypotheses/
│   └── README.md                     # hypothesis/suffix 注册规则
├── scripts/                          # 后续添加可复用提取/训练脚本
└── fixtures/                         # 极小、无敏感信息的 smoke 数据
```

契约、smoke test 和真实模型验收已实现；后续扩展新模型时继续复用
`DESIGN.md` 中的请求与数值对照契约，不要写模型专用的旁路。

## 真实模型 smoke / 数值对照

`test_qwen3_hidden_layer.py` 是当前 Qwen3 后端的手工验收入口。它默认关闭
radix cache 和 chunked prefill，执行 prefill-only 请求，并检查 explicit 与
suffix_last 两种位置选择返回相同向量：

```bash
INSCOPE_QWEN3_MODEL=/DATA/disk1/models/Qwen3-8B \
  /opt/miniconda3/envs/inscope/bin/python \
  test/manual/feature_capture/test_qwen3_hidden_layer.py --gpu 0
```

加入 `--compare-hf` 会在 SGLang Engine 关闭、显存释放后加载同一 checkpoint，
对每个目标层报告 cosine similarity 和 relative L2，并应用 `DESIGN.md` 中的
BF16 阈值（cosine ≥ 0.9999、relative L2 ≤ 2%）。FP8 可通过
`--model "$INSCOPE_QWEN3_FP8_MODEL"` 运行 SGLang smoke，
但不能把 FP8 输出当作 BF16 golden。运行前应先用 `nvidia-smi` 确认所选 GPU
空闲，不要抢占或终止其他用户的进程。

为了避免 GPU 被抢或 HF 阶段失败后重跑 SGLang，建议先原子保存 capture：

```bash
.../test_qwen3_hidden_layer.py --gpu 7 --sglang-only \
  --save-captures /tmp/qwen3-8b-captures.pt --benchmark-iterations 20

# 之后只运行 HF 对照；模型路径默认从 capture 文件恢复
.../test_qwen3_hidden_layer.py --gpu 7 \
  --hf-only /tmp/qwen3-8b-captures.pt
```

输出计时包括 Engine startup、explicit/suffix 请求；启用 benchmark 后另报告
普通 prefill、单层 capture、全部目标层 capture 的 mean/P50/P95。

FP8 的首次请求可能触发多轮 DeepGEMM kernel tuning，单轮可持续数分钟。脚本
默认将 scheduler watchdog 放宽到 1200 秒，避免默认 300 秒在调优尚未完成时
误判为 hang；稳定运行不应出现这段开销。

只验证功能而不做 DeepGEMM 性能调优时，使用 Triton 后端：

```bash
.../test_qwen3_hidden_layer.py --gpu 7 --sglang-only \
  --fp8-gemm-backend triton
```

P3 手工矩阵可分别启用：

```bash
# 乱序多层 × 多位置，并与逐位置请求对照
.../test_qwen3_hidden_layer.py --fp8-gemm-backend triton \
  --layers 36 0 20 1 35 --p3-multi-position

# 无 capture / 单层 / 多层多位置的混合 batch
.../test_qwen3_hidden_layer.py --fp8-gemm-backend triton \
  --layers 36 0 20 1 35 --p3-mixed-batch

# 84-token 输入按 16 token 分块，目标横跨多个 chunk
.../test_qwen3_hidden_layer.py --fp8-gemm-backend triton \
  --layers 36 0 20 1 35 --repeat-input 4 \
  --chunked-prefill-size 16 --p3-multi-position

# radix cache 冷/热请求逐元素一致；目标之前的 KV 仍可复用
.../test_qwen3_hidden_layer.py --fp8-gemm-backend triton \
  --layers 36 0 20 1 35 --enable-radix-cache --p3-radix-cache

# 三个独立异步请求与各自串行基准对照
.../test_qwen3_hidden_layer.py --fp8-gemm-backend triton \
  --layers 36 0 20 1 35 --p3-concurrent

# 普通生成回归，以及生成同时返回 prompt features
.../test_qwen3_hidden_layer.py --fp8-gemm-backend triton \
  --p3-ordinary-regression --p4-generate-features

# Qwen3.5-9B Dense text tower（checkpoint 顶层含视觉塔）
INSCOPE_QWEN3_MODEL=/DATA/disk1/models/Qwen3.5-9B \
.../test_qwen3_hidden_layer.py --language-model-only \
  --layers 32 0 16 1 31 --p3-multi-position --p4-generate-features
```

层号是 residual-stream block boundary：`L<N` 与 HF `hidden_states[L]` 对齐；
`L=N` 是最后一个 decoder block 的 pre-final-norm 输出，HF 参考由最后 block
forward hook 获取，不能直接使用 post-final-norm 的 `hidden_states[N]`。

## 当前范围

当前已验收范围：

- 模型：Qwen3-8B BF16、Qwen3-8B-FP8 和 Qwen3.5-9B Dense text tower
- 文本输入；不测试视觉输入
- TP=1、PP=1
- prefill-only 以及 generate + features
- 目标位置：explicit、last、suffix_last 和多位置
- 表示：decoder block output / residual stream，未经 final norm
- 多层：同一次 prefill 按请求顺序捕获多个 layer
- 运行时：mixed batch、chunked prefill、radix cache 和并发请求
- dtype：服务端可以是 BF16；落盘格式固定记录实际 dtype
- 每个风险类别/上下文属性使用独立 hypothesis suffix

BF16 版本用于数值参考；FP8 版本是第一优先的运行目标，但不能把 FP8 结果当作 BF16 golden。两者应使用相同的样本、tokenizer、chat template、suffix 和 layer 集合，分别报告结果。

## 不在本目录做的事

- 不把 SafeGauge 的风险类别写进 SGLang 生产代码
- 不把具体数据集、probe 权重或 checkpoint 路径提交进仓库
- 不把 probe 训练当作 InScope 后端正确性的替代验收
- 不在 CI 中默认启动 8B 模型并训练 MLP
- 不把不同 hypothesis 的 suffix 结果混成一个无元数据的向量
