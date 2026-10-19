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

后续正式实现前，先完成 `DESIGN.md` 中的契约和 smoke test；不要直接写一份只适配 Qwen3-8B 的脚本。

## 当前范围

首版测试固定：

- 模型：Qwen3-8B BF16 和 Qwen3-8B-FP8
- 文本输入；不测试视觉输入
- TP=1、PP=1
- prefill-only
- 目标位置：完整拼接输入最后一个 token，即 suffix 最后一个 token
- 表示：decoder block output / residual stream，未经 final norm
- 多层：同一次 prefill 捕获多个 layer
- dtype：服务端可以是 BF16；落盘格式固定记录实际 dtype
- 每个风险类别/上下文属性使用独立 hypothesis suffix

BF16 版本用于数值参考；FP8 版本是第一优先的运行目标，但不能把 FP8 结果当作 BF16 golden。两者应使用相同的样本、tokenizer、chat template、suffix 和 layer 集合，分别报告结果。

## 不在本目录做的事

- 不把 SafeGauge 的风险类别写进 SGLang 生产代码
- 不把具体数据集、probe 权重或 checkpoint 路径提交进仓库
- 不把 probe 训练当作 InScope 后端正确性的替代验收
- 不在 CI 中默认启动 8B 模型并训练 MLP
- 不把不同 hypothesis 的 suffix 结果混成一个无元数据的向量
