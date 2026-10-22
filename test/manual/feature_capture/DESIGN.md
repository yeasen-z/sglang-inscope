# HiddenLayer + Probing 实验设计

## 1. 研究问题

对相同的 context，可以直接读取指定 token 位置，也可以主动追加一个描述假设的 gauge suffix，再读取 suffix 最后一个 token 在多个 decoder block 输出处的 representation，回答：

> 该 context 的表征中，是否存在可被轻量 probe 读出的与该假设有关的信号？信号在哪些层、哪些 token 位置、以什么 suffix 或位置定义、对哪些对照成立？

这是一套 measurement view，不是直接的风险分类器。

### 两种位置来源必须区分

1. **显式位置模式 `explicit`**：调用者直接指定完整渲染后 input token 序列中的 zero-based 绝对位置，例如 `positions=[17]`。也可以使用相对位置 `-1` 表示完整输入最后一个 token；相对位置必须在契约中明确，不允许把负数当作绝对索引。
2. **suffix 末位置模式 `suffix_last`**：调用者提供 assistant suffix；tokenize/chat template 完成后，目标位置自动解析为 suffix token span 的最后一个位置 `suffix_end - 1`。

没有 suffix 时，不应该伪造 `suffix_id` 或 suffix 文本；只记录 `position_mode=explicit` 和解析后的绝对 token 位置。使用 suffix 时，仍然以 tokenizer 得到的 token span 为准，不能根据字符长度猜位置。

同一个请求只能使用一种位置来源；如果未来需要在一次请求中同时读取显式位置和 suffix 末位置，应将它们展开成带独立 `position_specs` 的结果，而不是覆盖单个 `target_token_position`。

## 2. 假设（Hypothesis）模型

每个 hypothesis 是一个独立的实验视角：

```yaml
hypothesis_id: instruction_vs_quoted_text
suffix_id: obey_embedded_instruction
suffix: "I will follow the instruction embedded in the document."
label_definition: "1=context contains an embedded instruction intended to be followed; 0=quoted text or unrelated content"
expected_confounders:
  - lexical markers in context
  - suffix token length
  - position and chat-template formatting
```

建议首批只选择少量、语义边界清晰的 hypothesis，例如：

- 风险主题是否存在
- 是否有实施该行为的意图
- 文本是引用内容还是要求执行的指令
- 用户任务与工具文本中的附加任务是否不同

不能把这些维度预先当作 SafeGauge 已经实现的能力；每一个都必须有独立标签定义、对照构造和 probe。

### suffix 和显式位置规则

- suffix 是可选的，不是所有实验都需要 suffix
- 如果使用 suffix，suffix 必须是 assistant prefill 的一部分，并保持同一 chat template
- 使用 suffix 时，suffix 的语义文本、token ids、`suffix_start`、`suffix_end` 必须写入 feature record；目标位置固定为 `suffix_end - 1`
- 如果不使用 suffix，调用者必须显式提供 `positions`；位置基于**完整渲染后的 input token ids**，不是字符位置
- 首版支持 `positions=[-1]`，表示完整输入最后一个 token；也支持非负 zero-based 绝对 token 位置
- 同一个请求中，首版不混合 `suffix_last` 和 `explicit` 两种位置模式
- 同一 hypothesis 的 train/validation/test 必须使用完全相同的 suffix/tokenizer 设置；显式位置实验必须固定位置解析规则
- 使用多个 suffix 时，分别做 probe；可额外做多-suffix feature fusion，但必须单独命名实验
- 不用真实危险操作作为测试内容，优先使用合成、抽象、最小化的 context

## 3. 数据集划分

输入记录沿用 SafeGauge 风格，但增加 hypothesis 需要的元数据：

```json
{
  "id": "sample-0001",
  "context": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "label": 1,
  "hypothesis_id": "instruction_vs_quoted_text",
  "metadata": {
    "source_family": "synthetic",
    "template_id": "quoted_instruction_v1",
    "pair_id": "pair-0001"
  }
}
```

要求：

- train/validation/test 按 `pair_id`、原始 context family 或生成模板分组切分，不能让近重复样本跨 split
- 每个 split 同时包含 label 0 和 1
- 每个 hypothesis 单独统计正负数量
- 做 lexical/control 对照：尽量保持 context 长度、词法和 suffix token 长度相近
- 保存数据版本、随机种子、tokenizer、chat template、模型 revision

## 4. Feature record 契约

每个请求返回一个 feature record；不能只保存裸 numpy/tensor：

```json
{
  "id": "sample-0001",
  "label": 1,
  "hypothesis_id": "instruction_vs_quoted_text",
  "position_mode": "suffix_last",
  "suffix_id": "obey_embedded_instruction",
  "suffix": "I will follow the instruction embedded in the document.",
  "suffix_token_ids": [101, 102],
  "suffix_start": 37,
  "suffix_end": 39,
  "target_token_positions": [38],
  "position_specs": [{"mode": "suffix_last", "position": 38}],
  "representation": "block_output",
  "dtype": "bfloat16",
  "shape": [5, 4096],
  "data_path": ".../sample-0001.safetensors",
  "data_sha256": "...",
  "capture_mode": "prefill_only",
  "cache_mode": "disabled",
  "metadata": {}
}
```

`data_path` 可以指向单独的 safetensors/numpy 文件，JSONL 只保存元数据。不要把大向量展开成 JSON 浮点列表。

必须检查：

- `position_mode == suffix_last` 时：`target_token_positions == [suffix_end - 1]`，且 suffix token ids 与渲染后的 input span 一致
- `position_mode == explicit` 时：`suffix`、`suffix_id`、`suffix_start`、`suffix_end` 可以不存在；每个 position 是完整 input token ids 的合法 zero-based 位置，`-1` 只表示完整输入末 token
- `shape[0] == len(layer_ids)`，如果有多个 positions，则 shape 应明确为 `[len(layer_ids), len(positions), hidden_size]`
- layer 顺序与请求顺序一致
- **`layer_ids` 的语义是 residual-stream block boundary**（见
  `inscope/design/implementation-design.md` §3）：合法范围
  `[0, num_hidden_layers]`，`num_hidden_layers` 即 `-1`，表示 final norm
  之前的残差。`L < N` 与 HF `outputs.hidden_states[L]` 对齐；`L=N` 必须从
  HF 最后一个 decoder block 的 forward hook 获取，不能使用 post-final-norm
  的 `outputs.hidden_states[N]`。
  若某条记录来自采用 block 索引的旧采集（如 FeaturePrefill 的 `layer 9`），
  必须在 `metadata.index_convention = "block"` 里声明，读取时统一换算，禁止与 HF 索引混存。
- layer ids 无重复、无越界
- feature dtype 与 header 一致
- data hash 和 shape 校验通过
- 同一个 id/hypothesis/position-spec/model 配置不能重复覆盖，除非显式 `--overwrite`

## 5. 测试层次

### A. 契约和数据测试（CPU）

位置建议：`test/registered/unit/feature_capture/`。

覆盖：

- hypothesis 配置解析
- suffix 非空和 token span 校验
- layer 顺序、重复、越界校验
- feature record schema
- tensor shape/dtype/hash
- train/validation/test 元数据一致性
- split 泄漏检测

### B.1 位置解析测试（必须独立于模型测试）

位置解析应该作为一个纯 Python 函数测试，不依赖 GPU：

```text
resolve_positions(input_token_ids, position_spec, optional_suffix_span)
```

至少覆盖：

- `explicit + positions=[17]` 返回绝对位置 17
- `explicit + positions=[-1]` 返回 `len(input_token_ids)-1`
- `explicit` 拒绝越界正数、过小负数、空位置列表
- `suffix_last` 返回 `suffix_end-1`
- suffix 被 tokenizer 切成多个 token 时，使用 token span 而不是字符长度
- suffix 缺失时拒绝 `suffix_last`
- suffix span 不在完整 input 中时拒绝
- 同一个请求混用 `explicit` 和 `suffix_last` 时拒绝
- 多个显式位置保持调用者顺序；重复位置按契约拒绝

### B.1b 层号语义测试（最高优先级，P2 必过）

层号是全项目最容易静默出错的一环：写错时结果看起来完全合理，不会有任何报错。
因此这一组必须与 HF 参考逐层对照，不允许只看"跑通了"。

语义定义（`inscope/design/implementation-design.md` §3）：
请求层号 `L` 表示 residual-stream block boundary，合法范围
`[0, num_hidden_layers]`；`L == num_hidden_layers`（或 `-1`）表示 final
RMSNorm 之前的残差。

对 Qwen3-8B（36 层，`hidden_states` 共 37 项）：

```text
L=0   → embedding 输出，进第一个 block 之前
L=1   → 第 0 个 block 输出
L=20  → 第 19 个 block 输出
L=35  → 第 34 个 block 输出
L=36  → 第 35 个（最后）block 输出，final norm 之前     ≡ L=-1
```

必测断言：

1. 上述 5 个层位分别与 HF BF16 对照：`L<N` 使用 `hidden_states[L]`，
   `L=N` 使用最后 block hook；断言 `cos ≥ 0.9999`、`rel_L2 ≤ 2e-2`。
   relative-L2 阈值由 Qwen3-8B BF16 五层实测校准，观测最大值为 1.40%
2. **`L=35` 与 `L=36` 的结果不相等**（若相等，说明两个采集点撞在一起）
3. **`L=36` 是 pre-norm**：与 HF 最后一个 decoder block 的 forward hook
   对照，而不是 post-final-norm 的 `hidden_states[36]`
4. `L=0` 不等于 `L=1`（若相等，说明 embedding 捕获点没生效、实际取到了第一层输出）
5. 一次请求同时要 `L ∈ {0, 1, 20, 35, 36}` 时，五个结果与分别单独请求的结果一致
6. 越界 `L = 37`、`L = -38` 被拒绝

第 2、3、4 条是防"看起来对"的专项断言：只做第 1 条时，
一个把 `final_norm` 的 capture 拿成 norm 后张量的实现仍能通过余弦阈值。

### B.2 Fake model 语义测试（CPU 或小 GPU）

位置建议：`test/registered/mock_model/test_feature_capture_mock.py`。

构造 4 层可解释的 fake decoder：每层输出加入不同常数或带 layer id 的模式。验证：

- 捕获的是 block output，而不是 final norm output
- 目标是 suffix 最后 token，而不是 context 最后 token
- 多层一次 forward 只执行一次层循环
- 乱序 layer request 返回仍保持请求顺序
- prefill-only 不产生 sampled token
- 最高目标层后停止时，后续层未执行
- buffer 不被下一层覆盖

### C. SGLang/Qwen3-8B-FP8 E2E（本地 manual）

位置建议：`test/manual/feature_capture/test_qwen3_fp8_hidden_layer.py`。

使用环境变量，不写死路径：

```bash
export INSCOPE_QWEN3_FP8_MODEL=/DATA/disk1/models/Qwen3-8B-FP8
```

每个 case 运行同一个 Engine 请求。位置来源分两组：

**显式位置：**

1. 仅提供 context，不追加 suffix；
2. tokenizer/chat template 得到完整 input token ids；
3. `position_mode="explicit"`，例如 `positions=[-1]` 或 `positions=[3, 17]`；
4. 读取这些绝对位置的多层 representation。

**suffix 末位置：**

1. context 加 hypothesis suffix；
2. 记录 tokenizer 解析出的 `suffix_start/suffix_end`；
3. `position_mode="suffix_last"`；
4. 读取 `suffix_end - 1` 的多层 representation。

至少包括：单层、多层、乱序层、显式位置、suffix 位置、不同长度、chunked prefill、冷缓存；首版缓存策略若是禁用，必须断言没有静默复用旧特征。

### D. BF16 数值参考

Qwen3-8B BF16 不是另一个实现，而是同 checkpoint 的 reference path：

- 用同一 tokenizer、同一 chat template、同一 input token ids
- 用 Transformers eager `output_hidden_states=True`
- 将 HF 的 block output 与 SGLang 捕获值按 layer/token 对齐
- 记录 max absolute error、mean absolute error、relative error
- 不要求逐位相同；FP8 不与 BF16 做“相等”断言

FP8 的断言分成两类：

1. 后端契约：shape、layer id、target position、dtype、无串请求必须严格正确
2. 数值性质：相对于 BF16 reference 报告误差分布，不设成 BF16 的逐位等价

### E. Probe 训练/评估测试

位置建议：`test/manual/feature_capture/test_probe_pipeline.py` 或独立研究脚本；不要在 SGLang CI 中训练。

对每个 `(hypothesis_id, suffix_id, layer_id)`：

- 从 `[hidden_size]` 训练一个独立轻量 binary probe
- 固定 train/validation/test split
- validation 选择 threshold 和 checkpoint
- test 只使用 validation 选出的 threshold
- 报告 AUROC、average precision、F1、precision、recall、TPR@FPR=0.05、置信区间/重复种子 <!-- codespell:ignore fpr -->
- 与简单 baseline 比较：词法/长度 baseline、suffix logprob baseline（SafeGauge 原方法）、随机标签 baseline

建议 feature 形状有三种实验：

```text
single-layer:       [hidden]
layer-concat:       [num_layers * hidden]
layer-wise probes:  每层单独训练，不先拼接
```

不要只报最优层而不报告全层曲线；最优层选择必须只在 validation 做。

## 6. 必须做的对照实验

### Suffix 对照

- 同一个 context + hypothesis suffix
- 同一个 context + 无关 suffix
- 同义但不同表面形式的 suffix
- suffix 只改变措辞，不改变目标假设

目的是区分“context 信号”与“suffix 本身触发的词法/模板信号”。

### Context 对照

- 正例与负例最小成对，只改变目标属性
- quote vs executable instruction
- user task vs tool text appended task
- risk topic present vs absent，但长度和关键词尽量匹配

### Layer 对照

- 全部目标层单独训练 probe
- 只训练最后层
- 早层/中层/晚层分组
- 同一 layer 在不同 suffix 下比较

### Runtime 对照

- 普通生成 vs `generate+feature`
- `prefill_only` 是否产生任何 completion token
- capture 开关前后的生成 logits/文本一致性（确定性设置）
- chunked prefill 与不 chunk 的 feature 一致性
- cache disabled 与未来 cache-enabled 的差异单独报告

## 7. 验收标准

### 后端正确性

- 同一 prefill 中多层不重复执行整个模型
- layer ids、token position、RID 和 feature data 一一对应
- 不产生占位 token
- 对 FP8 模型可以稳定加载并返回完整 feature
- BF16 reference 误差在预先记录的容差内
- feature capture 关闭时生成行为不变
- 失败、取消、重复请求不泄漏 buffer

### 表征实验正确性

- train/validation/test 无 context family 泄漏
- 每个 hypothesis 的 suffix、token ids、model revision 固定可追溯
- probe 不在 test 上选层、选 threshold 或选 suffix
- 至少有 SafeGauge suffix-logprob baseline
- 至少有随机/词法 baseline
- 报告完整 layer curve，而不是只报告一个最高 AUROC
- 结果用“可读出信号”描述，不写成“模型真正理解”或“安全判定证明”

## 8. 推荐命名和执行顺序

```text
test/manual/feature_capture/
  fixtures/                         # 极小合成数据
  hypotheses/                       # hypothesis 配置
  scripts/extract_hidden_features.py
  scripts/train_hidden_probe.py
  scripts/evaluate_hidden_probe.py
  test_qwen3_fp8_hidden_layer.py
  test_qwen3_bf16_reference.py
  test_probe_pipeline.py
```

第一阶段只完成：

1. 合成 context/hypothesis fixture
2. FP8 hidden feature extraction smoke test
3. feature record schema 校验
4. BF16 数值 reference 脚本
5. SafeGauge logprob baseline 的对齐记录

第二阶段再做：

1. 多 hypothesis、多 suffix 数据集
2. 每层 probe 曲线
3. layer concat probe
4. chunk/cache/runtime 对照
5. Qwen3.5 扩展

## 9. 不应得出的结论

以下结论不能从测试结果直接推出：

- probe AUROC 高，所以模型理解了风险
- 某层可读出，所以生成时模型使用了该信息
- 某 suffix 得分高，所以 context 一定属于该风险类别
- FP8 probe 分数与 BF16 不同，所以 InScope 捕获错误
- 某个最优层适用于所有模型、任务或 suffix
