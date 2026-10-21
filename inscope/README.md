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

代码落点：
- 通用运行时 `python/sglang/srt/feature_capture/`——模型无关
- 模型注入点 `models/*.py`——仅保留薄适配层

## 三种执行模式

| 模式 | 请求形态 | 产出 |
|---|---|---|
| `generate` | 普通请求 | 文本（不创建 feature plan/缓冲或 D2H 传输） |
| `prefill_only` | `max_new_tokens=0` + `feature_capture` | 仅特征，跳过 decode/采样/logits |
| `generate+features` | 普通请求 + `feature_capture` | 文本 + 特征 |

## 当前状态

- [x] 前作调研（四套实现，全部在本机，不在本仓）
- [x] 实现设计 v2 定稿
- [x] 测试设计与三份 JSON schema（`test/manual/feature_capture/`）
- [x] **评审决策 R1/R2/R3**（见设计 §15）：交付面只做 Engine、P5 只做 9B、probe 在外部仓
- [x] **P0 环境**：Conda `inscope`（`/opt/miniconda3/envs/inscope`），`torch==2.14.1+cu130`、`flashinfer_python[cu13]==0.7.2`、`sglang-kernel==0.4.10`，当前源码以 editable 模式安装
- [x] P1 契约与校验骨架：`spec.py`、`plan.py`、`capabilities.py`、`bridge.py` 已落盘
- [x] P2 请求透传：`GenerateReqInput → TokenizedGenerateReqInput → Req → ForwardBatch` 已打通并建立 extend slot plan
- [x] P2 CPU/设备通路骨架：recorder、Qwen3 layer callback、异步 auxiliary output、`meta_info["features"]` 已接通
- [x] P2 Engine API：同步/异步 Engine 与 HTTP adapter 均透传 `feature_capture`
- [x] P2 手工验收入口：Qwen3 explicit/suffix smoke、base64 解码、HF BF16 数值对照工具
- [x] CPU 回归：feature-capture 套件加 output-streamer 相关测试 87 passed（另有 35 个 subtests）
- [x] P2 GPU 验证：Qwen3-8B BF16 五层 HF 对照通过；Qwen3-8B-FP8 使用 Triton 后端完成真实 `prefill_only` 闭环
- [x] P2 Qwen3-8B-FP8 `prefill_only` 最小闭环（真实 GPU）
- [x] P3 多层 / 多位置 / 混合 batch / chunked / 缓存命中 / 并发 / 普通生成回归
- [x] P4 `generate+features`：生成 token/text 与普通生成一致，仍只采 prompt
- [x] P5 第二架构 `Qwen3.5-9B`（Dense text tower）证明解耦：单/多位置与 generate+features 真实 GPU 通过
- [ ] P6 SafeGauge 式探测实验（外部仓）
- [ ] P7（推迟）OpenAI 兼容层

## 两条不能妥协的约束

1. **模型适配与通用逻辑解耦**：`feature_capture/` 里不出现任何模型名；
   模型文件只声明 `feature_capture_sites`，并在已有 layer boundary 上
   调用 `layer_capture(...)` / `final_capture(...)`；位置、编码和传输逻辑不得进入模型文件。
2. **数值与位置不能靠"看起来对"**：`L < N` 与 HF `outputs.hidden_states[L]`
   对齐，`L=N` 是最后 block 的 pre-final-norm 输出，必须用 block hook 作 HF
   参考；缓存命中重放、FP8 与 BF16 的关系都有专门用例；
   FP8 永不与 BF16 断言逐位相等。

## 知识边界

探测 AUROC 高只说明表示里存在**可利用的信号**，
不等于模型「理解」了该属性，也不证明生成时真的使用了它。
任何结论文档必须保留这条边界。
