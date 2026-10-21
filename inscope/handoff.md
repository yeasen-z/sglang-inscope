# InScope（观心）开发交接文档

> 本文是当前项目的唯一快速交接入口。新接手者应先读本文，再按链接进入详细设计。
>
> 更新时间：2026-10-20（Asia/Shanghai）

---

## 1. 项目目标

在 fork 的 SGLang 中增加一条**一等公民的 feature-capture 通路**：

- 一次 prompt prefill；
- 在指定 token 位置、指定多个层上捕获 block 输出残差流；
- 捕获点位于 final norm 之前；
- 支持仅特征、普通生成加特征两种主要使用方式；
- 供外部 SafeGauge/HiddenLayer/Probing 实验使用。

SafeGauge 的典型调用方式是：给定 context 和 assistant gauge suffix，对每个风险假设拼出一段 prompt，在指定位置（通常是 suffix 最后一个 token）读取多个层的 hidden representation，再交给外部 probe。

### 重要知识边界

probe 的高 AUROC 只能说明表示中存在可利用信号，不等于模型“理解”了该属性，也不证明生成时模型实际使用了该属性。后续实验报告必须保留这条边界。

---

## 2. 仓库、基线与环境

### 仓库

- 远端 workspace：`/DATA/disk2/workspace/zmosg/sglang-inscope`
- Git remote：`https://github.com/yeasen-z/sglang-inscope.git`
- 当前基线：`main@92580581da6fc8542a997a00c1a8b71b03497680`
- 当前状态：生产推理链路尚未接入 feature capture；设计、移植分析、P0 环境和 P1 纯 Python 骨架已完成。

### 模型

模型位于 `/DATA/disk1/models/`：

- `Qwen3-8B`：BF16 数值基准，36 层，hidden size 4096；
- `Qwen3-8B-FP8`：主要开发目标，FP8 权重；
- `Qwen3.5-9B`：P5 第二架构，Dense，32 层。

代码中禁止硬编码模型路径，使用环境变量，例如：

```bash
export INSCOPE_QWEN3_MODEL=/DATA/disk1/models/Qwen3-8B
export INSCOPE_QWEN3_FP8_MODEL=/DATA/disk1/models/Qwen3-8B-FP8
export INSCOPE_QWEN35_9B_MODEL=/DATA/disk1/models/Qwen3.5-9B
```

### P0 环境

当前开发环境：

```text
环境：/opt/miniconda3/envs/inscope（Conda env：inscope）
Python：3.12
torch：2.14.1+cu130
flashinfer：flashinfer_python[cu13]==0.7.2
sglang-kernel：0.4.10
sglang：当前仓库 python/ 的 editable 安装
```

editable 安装必须设置 `SGLANG_BUILD_RUST_EXTS=none`，否则 `setup.py` 会发现
仓库内 Rust workspace 并要求安装 cargo。

验证：

```bash
SGLANG_CACHE_DIR="$PWD/.cache/sglang" \
FLASHINFER_WORKSPACE_BASE="$PWD/.cache" \
/opt/miniconda3/envs/inscope/bin/python -c 'import sglang; print(sglang.__file__)'
```

当前沙箱看不到 GPU（`torch.cuda.device_count() == 0`）；GPU E2E 需要在有设备权限的
运行环境中执行，但这不影响 CPU contract/plan 测试。

---

## 3. 已经确定的架构决策

### 3.1 层号语义：residual-stream block boundary

用户侧层号 `L` 表示 block boundary：

- 范围是 `[0, N]`，其中 `N` 是 decoder layer 数；
- `0` 是 embedding 输出；
- `L=N` 是最后一个 decoder block 输出、final norm 之前；
- `-1` 是 `N` 的别名；
- 内部不加 block offset，不使用“第几个 block”的替代语义。

其中 `L < N` 与 Hugging Face `outputs.hidden_states[L]` 对齐。当前
Transformers Qwen3 的 `outputs.hidden_states[N]` 已经过 final norm，因此
`L=N` 的 HF golden 必须从最后一个 decoder block 的 forward hook 获取。

对 Qwen3-8B：`0..36` 都是合法层号，`36` 和 `-1` 指向同一位置。

这是项目最重要的数值/位置约束，禁止重新改回前作的 block-index 语义。

### 3.2 采集点

复用基线已有的 layer-boundary callback：

- `L ∈ [0, N-1]`：在 layer-loop 的 attention entry/block boundary 捕获；
- `L=N`：通过 `final_norm(..., capture=callback)` 捕获 final norm 前的 residual；
- 不手写 `hidden + residual`；
- 不关闭 MLP all-reduce fusion；
- 不改模型对象上的全局 `layers_to_capture` 或 `capture_aux_hidden_states` 状态。

模型适配必须保持极小：模型文件只增加一个通用 `layer_capture(...)` 调用、一个 `final_capture(...)` 调用以及 `feature_capture_sites = ("block_output",)` 类属性。

### 3.3 请求字段与位置

使用一等请求字段 `feature_capture`，不把 JSON 塞进 `extra_key`。

位置模式：

1. `last`：prompt 最后一个 token；
2. `explicit`：显式 token 位置列表，支持负索引；
3. `suffix_last`：查找 suffix 的最后一次出现，取 suffix 最后一个 token；找不到时必须报错，不能静默退回 prompt 最后一个 token；
4. `all`：prompt 全部 token。

suffix 是可选能力；显式位置必须是一等公民。

### 3.4 传输通路

复用基线 auxiliary output 通道：

```text
LogitsProcessorOutput.feature_device_output
    → append_auxiliary_output(...)
    → DeviceAuxiliaryOutput.copy_to_host()
    → HostAuxiliaryOutput.consume(batch, commits)
    → tokenizer manager / Engine.generate() 返回
```

不能直接使用 `auxiliary_device_output`，因为 `ModelRunner.sample()` 会清理它。feature capture 使用独立字段，之后通过官方 `append_auxiliary_output` 合并。

### 3.5 consume 的三条防御

实现 `consume()` 时必须保留：

- 跳过 `req.is_retracted`；
- 根据 `finished_len` 过滤 stop-string/truncation 后不可见的尾部；
- 去重键使用 `(layer, position)`，不能使用 target index。

chunked prefill 的中间块也要被消费，不能只在最终完成时收集。

### 3.6 执行范围

- `prefill_only`：`max_new_tokens=0`，跳过 decode 和采样；
- `generate+features`：正常生成，同时采集 prompt 表示；
- v1 **不采 decode/生成 token**；
- 有 feature capture 的 batch 先使用 eager，优先保证正确性；
- v1 拒绝 MoE、多模态、PP、DP attention、speculative decoding、TBO 等复杂路径。

### 3.7 对外接口

R1 已确认：只交付 Python `Engine.generate()`。

原生 HTTP `/generate` 会免费继承结果，因为它直接返回 `tokenizer_manager.generate_request(...)` 的结果。

不实现 OpenAI 兼容层。原因不是“以后可能做”，而是部署形态决定 feature extractor 是独立进程，不与服务侧链接，OpenAI 层没有调用方。OpenAI 层列为 P7 deferred，不预留半成品字段。

### 3.8 模型范围

- P2/P3/P4：Qwen3-8B-FP8，Qwen3-8B 作为 BF16 数值参考；
- P5：只做 Qwen3.5-9B Dense；
- 不做 Qwen3.5-27B。

### 3.9 probe 归属

probe 训练与评估放外部仓库。本仓只保留运行时、契约、手工测试和数值验证，不存数据集、probe checkpoint 或训练输出。

---

## 4. 已完成的文件

### 设计文档

- [`design/implementation-design.md`](design/implementation-design.md)：完整实现设计 v2，包含 D1-D15、层号语义、请求契约、缓存边界、文件清单、P0-P7 计划和测试矩阵；
- [`design/porting-map.md`](design/porting-map.md)：前作 djs/zyt 核心实现逐项如何映射到当前基线，哪些可复用、哪些必须删除、哪些防御必须保留；
- [`design/dev-workflow.md`](design/dev-workflow.md)：为什么本项目使用 fork 分支作为真源，patch 只作为导出产物，而不采用 djs 的 patch-first 或 zyt 的 shadow-source workflow；
- [`design/prior-art.md`](design/prior-art.md)：四套既有实现的调研和债务对比。

### 手工测试契约

目录：`../test/manual/feature_capture/`

包含：

- `DESIGN.md`：SafeGauge/HiddenLayer/Probing 测试设计；
- `schemas/sample.schema.json`；
- `schemas/feature_record.schema.json`；
- `schemas/probe_manifest.schema.json`。
- `test_qwen3_hidden_layer.py`：真实 Qwen3 Engine smoke、响应解码，以及可选 HF BF16 数值对照。

feature record 的 `layer_ids` 必须解释为上述 block boundary；legacy
block-index 数据必须带 `metadata.index_convention = "block"`。

### P1 纯 Python 骨架

- `python/sglang/srt/feature_capture/spec.py`
  - 请求校验；
  - layer 解析；
  - 四种 position mode；
  - suffix 失败不降级；
  - raw-byte budget 检查。
- `python/sglang/srt/feature_capture/plan.py`
  - packed token row mapping；
  - 不采集的请求仍推进 flat-row offset；
  - layer slot layout；
  - 优先使用 `extend_prefix_lens_cpu`。
- `python/sglang/srt/feature_capture/capabilities.py`
  - `SGLANG_ENABLE_FEATURE_CAPTURE` 默认关闭；
  - v1 未验证配置逐项 fail-fast；
  - 模型适配通过 `feature_capture_sites` 类属性探测。
- `python/sglang/srt/feature_capture/bridge.py`
  - 合并 feature hook 与既有 auxiliary hook；
  - 不使用模块级运行时状态；
  - 不信任生产者的 `owned=True`，feature 路径固定复制。
- `test/registered/unit/feature_capture/test_feature_capture_spec.py`；
- `test/registered/unit/feature_capture/test_feature_capture_plan.py`。

最新 CPU 回归结果：feature-capture 套件加 output-streamer 相关测试
**87 passed**（另有 35 个 subtests）。

运行真实测试环境完成后执行：

```bash
cd /DATA/disk2/workspace/zmosg/sglang-inscope
SGLANG_CACHE_DIR="$PWD/.cache/sglang" \
FLASHINFER_WORKSPACE_BASE="$PWD/.cache" \
/opt/miniconda3/envs/inscope/bin/python -m pytest test/registered/unit/feature_capture/ -q
```

---

## 5. 前作核心实现的移植结论

### djs FeaturePrefill

它在 qwen2/qwen3 层循环里手算 residual、在 streamer 中新增 FeatureBatchOutput、在 batch result processor 中特殊收集 chunked payload，并用 `extra_key` 识别 feature 请求。

可借鉴：

- feature record 需要 layer ids、positions、dtype、raw bytes；
- chunked prefill 中间块不能丢；
- output 应用模型 dtype 的原始字节传输，而不是 `.tolist()`。

不搬：

- `extra_key` 协议；
-手工 `hidden + residual`；
- `break` 早停；
- 自建 FeatureBatchOutput/streamer 通道；
- 固定 512-token cache margin。

### zyt CoolWatch

它的 `activation_capture/runtime.py` 提供了最接近当前需求的三段式实现：`make_plan`、GPU capture/DeviceOutput、HostOutput.consume，并处理 retraction、finished_len 和重复位置。

可借鉴：

- `row_start` 必须覆盖 batch 中所有请求；
- `consume` 的三条防御；
- 精确 cache boundary；
- 后续若要做 decode graph，GPU selection table 是参考实现。

不搬：

- shadow `.source` 双源码工作流；
- 自建 CompositeDevice/CompositeHost；
- 当前 v1 的 decode capture 和 Triton capture kernel。

完整映射见 [`design/porting-map.md`](design/porting-map.md)。

---

## 6. 推荐开发流程

### 真源

建立开发分支：

```bash
cd /DATA/disk2/workspace/zmosg/sglang-inscope
git checkout -b feat/inscope
```

直接修改 fork，并使用 `pip install -e python`。不维护 `.source` 影子副本，不把 patch 当真源。

### 分阶段提交

推荐 commit 顺序：

1. `feature_capture: add request contract and row plan`；
2. `feature_capture: add capability gate and bridge`；
3. `feature_capture: capture at layer boundaries`；
4. `feature_capture: expose meta_info features`；
5. `qwen3: declare feature-capture adapter`；
6. `qwen35: declare feature-capture adapter`。

模型适配单独提交，避免上游模型循环重构时影响通用运行时。

### patch

只有需要让 stock SGLang 复现时才导出：

```bash
git diff main..feat/inscope -- python/ > inscope/patches/feature-capture-v1.patch
```

patch 是导出物，禁止手改。

---

## 7. 当前下一步（按顺序执行）

### 已完成：P2 GPU 验证

请求透传、recorder、Qwen3 callback、auxiliary output 和 response encoding
已经接通并通过真实模型验证：

- Qwen3-8B BF16 五层与 HF 对应 block boundary 数值对齐；最大 relative L2 约 1.4%；
- Qwen3-8B-FP8 在 Triton FP8 GEMM 后端完成五层真实 capture；
- explicit 与 suffix_last 逐元素一致，shape/dtype/finite 检查通过；
- DeepGEMM 首次自动调优与 feature capture 正确性分离，不把调优时间计作 capture 开销。

### 已完成：P3

- 乱序多层 `[36, 0, 20, 1, 35]` + 多位置 `[0, 10, 20]` 与独立请求逐元素一致；
- 混合 batch（无 capture / 单层 / 多层多位置）与独立请求一致；
- 84-token 输入、16-token chunk 的首/中/末位置跨块结果完整；
- radix cache 热请求从最早目标位置开始重算，保留此前 KV 命中，冷/热结果逐元素一致。
- 三个独立并发请求与各自串行基准逐元素一致，rid/层/位置未串线；
- 开启 capability 但不传 capture 的普通确定性生成 token/text 一致，且无非空 features。

### 已完成：P4 最小闭环

普通生成与 `generate+features` 均生成 8 token，token ids/text 完全一致；
feature 正常返回且只对应 prompt 位置，decode 阶段没有重复采集。

### 已完成：P5

Qwen3.5-9B checkpoint 顶层包含视觉塔，使用 SGLang 的
`--language-model-only`/`language_model_only=True` 只加载 Dense text tower。
通用 bridge 增加 accumulator-style boundary 适配器，模型侧仍保持一个 layer
collector、一个 final norm hook 和一个 `feature_capture_sites` 声明。真实 GPU 验证：

- 层 `[0,1,16,31,32]` explicit/suffix 逐元素一致；
- 乱序层 `[32,0,16,1,31]` × 位置 `[0,11,22]` 与独立请求一致；
- `generate+features` 的 8 个生成 token/text 与普通生成一致；
- 稳定 capture 请求约 41–42 ms（首次 GDN/Triton 编译不计）。

### 下一步：交付整理 / P6

先整理提交边界、上游跟随说明和完整验收命令；probe 训练与 SafeGauge 式实验
仍按 R3 放在外部仓库，本仓不引入数据集或训练依赖。

### 第五步：P5

将同一通用 runtime 接到 Qwen3.5-9B Dense，只允许模型文件增加三处最小适配，不复制 Qwen3 专用逻辑。

---

## 8. 风险和检查清单

实现前每次都检查：

- [ ] 是否把 `L=N` 错拿 HF post-final-norm 的 `hidden_states[N]` 作参考？
- [ ] 是否手动 `hidden + residual`？如果是，停止并改用 callback value；
- [ ] 是否把结果放进会被 `sample()` 清空的 `auxiliary_device_output`？
- [ ] 是否漏掉 chunked prefill 中间块？
- [ ] 是否忘记跳过 `is_retracted`？
- [ ] 是否用 `finished_len` 裁掉不可见尾部？
- [ ] 是否用 `(layer, position)` 去重？
- [ ] 是否让不采集请求的 flat rows 仍推进 offset？
- [ ] 是否把 suffix 缺失静默降级为 last？
- [ ] 是否在模型文件中写入模型专用 capture 逻辑？
- [ ] 是否把 probe 数据、checkpoint 或训练代码放进本仓？
- [ ] 是否误加 OpenAI 兼容层？当前明确不做。

---

## 9. 不要做的事情

1. 不要直接照抄 89KB 的 djs patch；它是部署快照，不是当前基线的正确实现。
2. 不要照抄 zyt 的 shadow-source 目录；我们有 fork 和 editable install。
3. 不要先做 decode graph、Triton kernel、MoE、OpenAI 层或 27B。
4. 不要为了适配 Qwen3-8B 而把逻辑写死在 Qwen3 模型文件中。
5. 不要把“探测有效”表述成“模型理解”或“模型使用了该属性”。

---

## 10. 关键入口索引

| 内容 | 路径 |
|---|---|
| 本交接文档 | `inscope/handoff.md` |
| 总体实现设计 | `inscope/design/implementation-design.md` |
| 前作核心实现移植对照 | `inscope/design/porting-map.md` |
| fork/patch 开发流程 | `inscope/design/dev-workflow.md` |
| 前作调研 | `inscope/design/prior-art.md` |
| 环境脚本 | `inscope/scripts/setup_env.sh` |
| 请求契约骨架 | `python/sglang/srt/feature_capture/spec.py` |
| 行规划骨架 | `python/sglang/srt/feature_capture/plan.py` |
| P1 契约测试 | `test/registered/unit/feature_capture/test_feature_capture_spec.py` |
| P1 行规划测试 | `test/registered/unit/feature_capture/test_feature_capture_plan.py` |
| 手工测试设计 | `test/manual/feature_capture/DESIGN.md` |
| 真实模型验收入口 | `test/manual/feature_capture/test_qwen3_hidden_layer.py` |

---

## 11. 交接结论

当前项目不是“还没有开始”，而是处于：

```text
前作调研完成
→ 架构设计和关键取舍完成
→ fork/patch 开发逻辑完成
→ P0 editable 开发环境完成
→ P1 请求契约与行规划完成
→ P2 运行时通路、公开 Engine API 与启动 capability gate 完成
→ P3 多层多位置、混合 batch、chunked prefill、radix cache、并发与普通生成回归通过
→ P4 generate + features 通过，开启 capture 不改变生成 token
→ P5 Qwen3.5-9B Dense text tower 的显式/后缀位置、乱序层与 generate + features 通过
→ 当前进入代码整理、静态检查和交付拆分阶段
```

最重要的实现原则只有一句：

> **通用运行时负责请求、位置、规划、传输和结果；模型文件只负责在正确的 layer boundary 调用通用 callback。**
