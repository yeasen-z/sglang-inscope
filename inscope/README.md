# InScope（观心）

在 SGLang fork 上加一条**一等公民的特征提取通路**：一次 prefill，
在指定 token 位置、指定多层上取 block 输出残差流（final norm 之前的表示），
用于安全探测（SafeGauge 式：同段落上下文，对多个风险假设分别做 last-token 探测）。

基线 `sglang-inscope@main` = `92580581da`，上游 `yeasen-z/sglang-inscope`。

## 目录

| 路径 | 内容 |
|---|---|
| [`design/implementation-design.md`](design/implementation-design.md) | **实现设计 v2**：15 条设计决策及被否方案、层号语义、请求契约、行映射、缓存边界、文件清单、阶段验收 |
| [`design/porting-map.md`](design/porting-map.md) | **核心实现移植对照**：前作每项机制在本基线对应什么、能搬什么、能删什么 |
| [`design/dev-workflow.md`](design/dev-workflow.md) | **开发流程**：fork 分支为真源、patch 只作发布产物、提交纪律与上游跟进成本 |
| [`design/prior-art.md`](design/prior-art.md) | 前作调研：本机四套已有的 SGLang 隐藏层采集实现及其代价 |
| `../test/manual/feature_capture/` | 测试设计、样例 schema、后续探测实验的落点 |

代码落点（尚未创建）：
- 通用运行时 `python/sglang/srt/feature_capture/`——模型无关
- 模型注入点 `models/*.py` 各 2 行

## 三种执行模式

| 模式 | 请求形态 | 产出 |
|---|---|---|
| `generate` | 普通请求 | 文本（无采集，零额外开销） |
| `prefill_only` | `max_new_tokens=0` + `feature_capture` | 仅特征，跳过 decode/采样/logits |
| `generate+features` | 普通请求 + `feature_capture` | 文本 + 特征 |

## 当前状态

- [x] 前作调研（四套实现，全部在本机，不在本仓）
- [x] 实现设计 v2 定稿
- [x] 测试设计与三份 JSON schema（`test/manual/feature_capture/`）
- [x] **评审决策 R1/R2/R3**（见设计 §15）：交付面只做 Engine、P5 只做 9B、probe 在外部仓
- [ ] **P0 环境**：conda env + `torch==2.14.1` + `flashinfer_python[cu13]==0.7.2` + sgl-kernel + `pip install -e python`（**进行中**）
- [~] P1 契约与校验骨架：`spec.py`、`plan.py` 已落盘，40 个纯 CPU 单测通过；待补 `capabilities.py`、`bridge.py`
- [ ] P2 Qwen3-8B-FP8 `prefill_only` 最小闭环（`Engine.generate()`）
- [ ] P3 多层 / 多位置 / chunked / 缓存命中 / 并发
- [ ] P4 `generate+features`（仍只采 prompt，不做 decode 采集）
- [ ] P5 第二架构 `Qwen3.5-9B`（Dense）证明解耦
- [ ] P6 SafeGauge 式探测实验（外部仓）
- [ ] P7（推迟）OpenAI 兼容层

## 两条不能妥协的约束

1. **模型适配与通用逻辑解耦**：`feature_capture/` 里不出现任何模型名；
   模型文件只有 2 行（`layer_capture(...)` + `final_norm(capture=...)`）和一个类属性
   `feature_capture_sites`。加第二个架构只允许改这 3 行。
2. **数值与位置不能靠"看起来对"**：层号语义统一为 HF `outputs.hidden_states[L]`
   （内部零偏移）；缓存命中重放、FP8 与 BF16 的关系都有专门用例；
   FP8 永不与 BF16 断言逐位相等。

## 知识边界

探测 AUROC 高只说明表示里存在**可利用的信号**，
不等于模型「理解」了该属性，也不证明生成时真的使用了它。
任何结论文档必须保留这条边界。
