# InScope 实现设计 v2

状态：设计定稿，待评审
基线：`sglang-inscope@main` = `92580581da`（2026-10-16）
目标模型：`Qwen3-8B`（BF16 数值基准）、`Qwen3-8B-FP8`（主要开发目标）
前作调研：[prior-art.md](prior-art.md)

> v2 相对 v1 的实质变化见 §0.2。最重要的一条是**层号语义改成 residual-stream block boundary**，
> 消除了整个项目风险最高的一项。

---

## 0. 一页结论

### 0.1 架构一句话

把采集拆成三块：**通用运行时**（`feature_capture/`，模型无关）+ **一行模型注入点** +
**复用官方传输协议**。模型文件里只出现一个函数名，不认识层号、位置、dtype、编码。

### 0.2 决策表（含被否方案）

每条都是「有别的做法，为什么不做」。

| # | 决策点 | 选择 | 被否的方案 | 否掉的理由 |
|---|---|---|---|---|
| D1 | 层内容从哪取 | 挂基线已有的 `capture` 回调 | 手算 `hidden + residual`（前作 A） | 基线已把 residual 合并好；手算会绕开融合 kernel 的 FP32 累加 |
| D1 | 同上 | 同上 | 取 FFN 输出再自己加残差（前作 D） | 为拿完整 FFN 输出必须关掉 MLP all-reduce 延迟融合，是性能损失 |
| D2 | **层号语义** | residual-stream block boundary，L ∈ [0, N]；`L < N` 与 HF `outputs.hidden_states[L]` 对齐 | block 索引（前作 D：layer 9 = 第 10 个 block） | embedding 是 0，随后每越过一个 block 加 1；避免使用方手写 `+1` |
| D2 | 最后一层 | `L = N`（也接受 `-1`）走 `final_norm(capture=)`，表示最后 block 的 pre-final-norm 输出 | 直接用 HF `hidden_states[N]` 作 golden | 当前 Transformers Qwen3 的最后一项已过 final norm；HF golden 必须挂最后 block hook |
| D3 | 结果回传 | 复用 `DeviceAuxiliaryOutput` → `copy_to_host` → `consume` | 自己写 D2H / 落盘中转（前作 A、C） | 官方已有 pinned + `record_stream` + 事件 + 进程通信，重写必踩同样的坑 |
| D3 | 挂在哪个字段 | `LogitsProcessorOutput.feature_device_output`，scheduler 里 `append_auxiliary_output` 合并 | 直接塞 `auxiliary_device_output` | `sample()` 开头会把它清成 `None`（`model_runner.py:2086`） |
| D3 | 合并方式 | 用基线 `append_auxiliary_output` | 手写 `CompositeDevice/CompositeHost`（前作 D） | 基线已有该协议，手写等于重复实现且漏掉观察者组合 |
| D3 | `consume` 的防御 | 跳过 `is_retracted`、按 `finished_len` 裁尾、键用 `(layer, position)` | 不做 | 三条都是前作 D 的实测经验，见 [porting-map.md](porting-map.md) §3.4 |
| D4 | 选行方式 | `torch.index_select` 到预分配输出 | Triton 融合 kernel（前作 D） | `capture` 给的 `value` 本来就在显存里，`index_select` 只产出小结果，不存在"物化整个 prompt"；前作 D 需要 fused kernel 是因为它还要 fused 加残差 |
| D5 | decode 阶段采集 | **v1 不做** | GPU 选择表 + graph 复用（前作 D） | SafeGauge 是 last-token **prompt** 探测，只采 prompt；为此改造 decode graph 是纯负担 |
| D5 | decode 的执行模式 | 有采集行 → eager | 固定容量 prefill graph（前作 A） | 先保正确；profiling 证明有收益再做 |
| D6 | 模型能力声明 | 模型类属性 `feature_capture_sites` | 字符串注册表 | 子类自动继承，不会漏注册；加模型 = 1 行调用 + 1 行类属性 |
| D6b | 不改模型属性 | 完全不碰 `capture_aux_hidden_states` / `layers_to_capture` | 开关 + `finally` 恢复（前作 A、D） | 开关会在包装层与内层之间产生 tuple 解包不一致；EAGLE 采集必须不受影响 |
| D7 | 缓存边界 | 精确到「最早未采集位置」+ 已采位置集合 | 固定 margin 512（前作 A） | 512 会白算最多 512 token；且它不处理重算导致的重复采集 |
| D8 | 请求契约 | 一等字段 `feature_capture` | `extra_key` 字符串塞 JSON（前作 A/B） | 无法类型化校验；校验时机被迫推迟到模型里；进程级环境变量（前作 B）连请求级都做不到 |
| D9 | 位置模式 | 四档：`last` / `explicit` / `suffix_last` / `all` | 只有 `last` | 用户明确要求显式位置为一等公民；`suffix_last` 是**校验器**不是别名 |
| D10 | 结果编码 | 模型 dtype 原始字节 + base64 | `.tolist()` 走 JSON（基线 `return_hidden_states` 的做法） | 4096 维 × BF16 转浮点文本会膨胀一个数量级且丢精度 |
| D11 | 是否支持 MoE | v1 拒绝 | 支持 | 前作 D 的 BF16 Qwen3.5 MoE 对照 9216 元素里 54 个超差，原因未明 |
| D12 | 进程开关 | `SGLANG_ENABLE_FEATURE_CAPTURE=0` 默认关 | 总是开启 | 关闭时不创建 plan、recorder、feature tensor 或 D2H 传输 |
| D13 | 交付面（R1） | **只做 `Engine.generate()`**；原生 `/generate` 免费 | 同时做 OpenAI 兼容层 | 见 §15.1/§15.2：原生 `/generate` 原样序列化 `meta_info`；OpenAI 层要逐字段改 3 文件、4 类位置，且部署形态是独立进程，没有调用方 |
| D14 | 第二架构（R2） | **`Qwen3.5-9B`（Dense）** | 同时上 `27B` | 27B 不增加验证覆盖面，只增加显存与时间成本 |
| D15 | probe 代码归属（R3） | **外部仓** | 放本仓 | 本仓保持「运行时 + 测试」，探测实验的依赖与数据不进模型服务仓 |

### 0.3 规模约束

通用逻辑集中在 `feature_capture/`；上游运行时文件只负责传递、规划与输出接线；
模型文件只声明 capability 并复用已有 layer-boundary callback。行数不作为
验收条件，以“模型中不出现位置、编码、传输逻辑”为准。

---

## 1. 为什么能比前作少改这么多

前作 A 改 17 文件 +887 行（v0.5.18）；前作 D 改 18 文件 +424 行运行时（0.5.19）。
我们的基线（`main@92580581da`）里已经有它们当时没有的地基：

| 需要的能力 | 基线已有 | 位置 |
|---|---|---|
| 数值正确的采集点（层入口） | `attn_boundary.prepare(..., capture=cb)` | `layers/layer_boundary/stage.py:165` |
| 数值正确的采集点（最后一层） | `final_norm(..., capture=cb)` | `layers/layer_boundary/residual/batch.py:135` |
| 所有权契约 | `capture(value, *, owned: bool)` | `layers/aux_hidden_states.py:20` |
| 设备→主机异步传输 | `DeviceAuxiliaryOutput.copy_to_host()` / `HostAuxiliaryOutput.consume()` | `managers/auxiliary_output.py` |
| 多来源输出合并 | `append_auxiliary_output()` | `managers/auxiliary_output.py:74` |
| 纯 prefill 完成路径 | `ScheduleBatch.is_prefill_only` + `tp_worker` 跳过采样 | `schedule_batch.py:1469`、`tp_worker.py:714` |
| 结果透传到 `meta_info` | `routed_experts` 的成熟模式 | `tokenizer_manager.py:2615` |

所以工作性质从「重建一条通路」变成「在既有接缝上插入一个消费者」。

---

## 2. 分层

```
┌──────────────────────────────────────────────────────────────┐
│ python/sglang/srt/feature_capture/          （模型无关）       │
│   spec.py         请求契约、校验、四种位置模式解析（纯 CPU）    │
│   plan.py         ForwardBatch → 本轮采集计划（行映射）        │
│   recorder.py     设备缓冲、每层 hook、结果分段                │
│   recorder.py     Device/Host auxiliary output 与 consume      │
│   encoding.py     base64 / float_list + 预算                  │
│   capabilities.py 启动拒绝 + 进程开关 + 能力探测               │
│   bridge.py       ★ 模型唯一入口                               │
└──────────────────────────────────────────────────────────────┘
        ▲                                          ▲
        │ 薄适配层                                  │ 复用既有 auxiliary 协议
┌───────┴──────────────┐              ┌────────────┴───────────┐
│ models/*.py          │              │ auxiliary_output.py    │
│  layer_capture(...)  │              │ managers/utils.py      │
│  final_norm(capture) │              │ scheduler.py(1 行合并)  │
└──────────────────────┘              └────────────────────────┘
```

**解耦的验收标准（由源码契约测试强制）**：

- `feature_capture/` 内不出现任何模型名、层数字面量、`qwen`/`llama` 字符串；
- 模型文件里不出现 `position`、`dtype`、`base64`、`layer_ids` 等采集语义，
  只出现 `feature_capture.layer_capture(...)` 与一个类属性。

---

## 3. 层号语义（本项目最高风险项，因此在这里定死）

### 3.1 定义

**用户写的 `L` 是 residual-stream block boundary**：embedding 为 0，越过每个
decoder block 后加 1，范围 `[0, N]`。

HF 的 `hidden_states` 是长度 `N+1` 的元组（Qwen3-8B：N=36 → 37 个元素）：

| HF 索引 | 含义 |
|---|---|
| `0` | embedding 输出（还没进任何 decoder block） |
| `L` (1 ≤ L ≤ N-1) | 第 `L-1` 个 block 的输出 = 进入第 `L` 个 block 之前的残差流 |
| `N`（即 `-1`） | Transformers 当前返回 final RMSNorm **之后**的输出，不能直接作为 InScope `L=N` golden |

### 3.2 内部映射：零偏移

基线的采集点正好与 HF 索引重合，**不需要任何 `+1`**：

| 请求的 `L` | 采集位置 | 数值 |
|---|---|---|
| `0 … N-1` | 层循环第 `L` 次迭代的注意力入口 | HF `hidden_states[L]` |
| `N`（或 `-1`） | `final_norm(..., capture=cb)` 的 `capture(residual)` | 最后 decoder block forward hook 的输出（pre-final-norm） |

两条都已核对基线源码：

- `stage.py:_prepare_attention`：`L=0` 时 `stream.residual is None` →
  `capture_before_read=True` → `capture(value)`，`value` 就是 embedding 输出；
  `L≥1` 时走 `capture(stream.residual)`，即上一层 block 输出（pre-norm）。
- `residual/access.py:final_norm_pair`：`residual` 非空时先
  `hidden_states, residual = norm(...)` 再 `capture(residual)`——
  拿到的是**更新后的 residual**，final norm 之前。当前 Transformers Qwen3
  的 `hidden_states[N]` 已经过 final norm，因此参考实现必须 hook 最后 block。

> **与前作 D 的换算**：前作 D 的 `layer: 9` 是"第 10 个 block 的输出"，
> 对应我们的 `L = 10`。文档必须写这句，否则移植已有 probe 会整体错一层。

### 3.3 支持负索引

`L = -1` → `N`，`L = -2` → `N-1`，采用 Python 式负索引。`-1` 表示最后一个
block boundary；不要把它等同于当前 Transformers 的
`outputs.hidden_states[-1]`（post-final-norm）。

### 3.4 为什么不用 block 索引

block 索引（"第 k 层 block 输出"，范围 `[0, N-1]`）本身没问题，但它与 HF 的数组差 1。
整个项目的使用方是 probe 训练代码，而那些代码无一例外从 HF 取向量。
选 block 索引等于强迫每一处使用方手写 `+1`——**而 `+1` 写错时结果看起来完全合理**，
不会有任何报错。这类错误一旦进入训练集，事后极难发现，只能全部重采。

选 HF 索引的代价只是文档里多一句与前作的换算说明。

### 3.5 必须有的测试

`test/manual/feature_capture/` 的 B.1 组，P2 就要过，不进 P3：

1. Qwen3-8B FP8，同一段输入，`L ∈ {0, 1, 20, 35, 36}` 全部采集；
2. 同一输入走 HF `transformers` BF16；`L<N` 读 `hidden_states[L]`，`L=N`
   从最后一个 decoder block forward hook 读取；
3. 逐层断言 `cos ≥ 0.9999`、`rel_L2 ≤ 2e-2`；该阈值由 Qwen3-8B BF16
   五层实测校准（观测最大 `rel_L2=1.40%`），保留约 0.6 个百分点余量；
4. 单独断言 `L=35` 与 `L=36` **不相等**（若相等说明两个采集点撞了）；
5. 单独断言 `L=36` 在 `final_norm` 之前（与 HF 最后 block hook 对照，不与
   post-final-norm 的 `hidden_states[36]` 对照）。

第 4、5 条是防"看起来对"的：如果 `final_norm` 的 capture 拿成了 norm 之后的张量，
`cos` 仍可能很高，但那是错的。

---

## 4. 接缝一：采集（模型侧薄适配）

### 4.1 模型改动

`models/qwen3.py`（其它模型同构）：

```python
 aux_hidden_states = AuxHiddenStateList()
 for i in range(self.start_layer, self.end_layer):
     hidden_states = self.layers[i](
         positions, hidden_states, forward_batch,
-        capture_output=aux_hidden_states.capture if i in self.layers_to_capture else None,
+        capture_output=feature_capture.layer_capture(i, forward_batch, aux_hidden_states, self.layers_to_capture),
     )
 ...
 hidden_states = residual_batch.final_norm(
-    hidden_states, forward_batch, self.norm, skip_empty=True
+    hidden_states, forward_batch, self.norm,
+    capture=feature_capture.final_capture(forward_batch), skip_empty=True,
 )
```

外加一行类属性：

```python
class Qwen3ForCausalLM(nn.Module):
    feature_capture_sites = ("block_output",)
```

### 4.2 `bridge.py`

```python
def layer_capture(*, layer, recorder, auxiliary_hook=None):
    feature_hook = recorder.capture_hook(layer=layer) if recorder else None
    return _combine(feature_hook=feature_hook, auxiliary_hook=auxiliary_hook)


def final_capture(*, num_layers, recorder):
    return None if recorder is None else recorder.capture_hook(layer=num_layers)
```

**三个刻意的性质**

1. **不改模型属性。** 前作 A/B 都靠 `model.layers_to_capture = [...]` +
   `capture_aux_hidden_states = True` 再在 `finally` 里恢复，还要处理
   "包装层以为要解包、内层没返回 tuple" 的畸形分支
   （前作 A 的补丁里 `isinstance(hidden_states, tuple) and len(hidden_states)==2 and isinstance(hidden_states[1], (list, tuple))` 就是为此）。
   我们完全不碰这两个状态，EAGLE/DFLASH 采集永远照常工作。
2. **可与既有 aux 采集共存。** 同槽位有两个消费者时，`_combine`
   依次调用 feature hook 和既有 auxiliary hook。
3. **`owned` 不被信任。** v1 一律复制进自己的缓冲。放弃一点性能换确定性——
   误判 `owned` 会读到被后续层复用的存储，产生随机复现的数值错误。

### 4.3 recorder `_capture()`

```python
def _capture(self, *, slot, value):
    if value.ndim != 2:
        raise RuntimeError("feature capture expects a 2D hidden tensor")
    torch.index_select(
        value, 0, slot.rows,
        out=self._buffer[slot.out_start:slot.out_stop],
    )
```

slot plan 在 forward 前就已经固定输入行与输出切片。recorder 会检查二维形状、
同一层不得重复采集，以及所有计划层都确实执行。

---

## 5. 接缝二：传输（复用官方协议）

### 5.1 `sample()` 会清空 aux 输出——精确分析

```python
# model_runner.py:2086  sample() 第一件事
logits_output.auxiliary_device_output = None
```

但注意调用链（`tp_worker.py:700-722`）：

```python
if not forward_batch.is_prefill_only:
    batch_result.next_token_ids = self.model_runner.sample(logits_output, forward_batch)
else:
    batch_result.next_token_ids = torch.zeros(...)      # 不调 sample()
```

所以：

| 路径 | `sample()` 是否调用 | 在 `_forward_raw` 里塞 `auxiliary_device_output` 会怎样 |
|---|---|---|
| `prefill_only`（SafeGauge 主路径） | **否** | 能保住 |
| `generate+features` | 是 | **被清成 None** |
| overlap 延迟采样 | 是（闭包里） | 被清成 None |

结论：**不能直接塞 `auxiliary_device_output`**，否则只有 prefill-only 能用。
这正是前作 D 新增 `activation_device_output` 字段的原因。

### 5.2 方案

```python
# layers/logits_processor.py —— 1 行（紧挨 auxiliary_device_output）
feature_device_output: Optional[DeviceAuxiliaryOutput] = None
```

```python
# model_executor/model_runner.py:_forward_raw 返回前 —— 1 行
ret.feature_device_output = feature_capture.finish(forward_batch)
```

```python
# managers/scheduler.py:_copy_auxiliary_output_to_cpu —— 3 行
device_output = append_auxiliary_output(
    self.logits_output.auxiliary_device_output,
    self.logits_output.feature_device_output,
)
```

之后全部由官方代码完成：

```
feature_device_output
  → append_auxiliary_output 合并
  → copy_to_host(_async_d2h)        # pinned + record_stream + 事件
  → consume(batch, commits)        # 采样已提交、响应之前
  → 每请求暂存
  → output_streamer 写 BatchStrOutput.features
  → detokenizer 透传
  → meta_info["features"]
```

这个合并对三条路径都成立：prefill-only 时 `auxiliary_device_output` 是 `None`，
`append` 直接返回我们的；有 observer 时两者组合成 `CompositeDeviceAuxiliaryOutput`。

### 5.3 `consume()` 做什么

```python
def consume(self, batch, commits):
    """Split this forward's captured rows back to their owning requests."""
    reqs = {req.rid: req for req in batch.reqs}
    for seg in self.segments:
        req = reqs.get(seg.rid)
        if req is None or req.is_retracted:
            continue                                  # ① 撤回的请求行已无效
        seen = req._feature_seen_positions
        # ② 只保留真正可见的位置：stop-string 命中或截断后的尾部不是有效向量。
        visible_end = len(req.origin_input_ids) + (
            req.finished_len if req.finished_len is not None else len(req.output_ids)
        )
        chosen = [
            i for i, p in enumerate(seg.positions)
            if p < visible_end and (seg.layer, p) not in seen
        ]
        if not chosen:
            continue
        positions = [seg.positions[i] for i in chosen]
        values = self.host_tensor[seg.start:seg.end]
        if len(chosen) != len(seg.positions):
            values = values[chosen]
        req._feature_chunks.append({
            "layer": seg.layer,
            "positions": positions,
            "token_ids": [req.token_ids_at(p) for p in positions],
            "shape": [len(positions), self.width],
            "dtype": self.dtype_name,
            # 一次整块字节拷贝，绝不产生「每个激活一个 Python float」
            "data": values.contiguous().view(torch.uint8).numpy().tobytes(),
        })
        seen.update((seg.layer, p) for p in positions)
```

三条防御都来自前作 D 的实测经验（详见 [porting-map.md](porting-map.md) §3.4）：

| # | 防御 | 不做的后果 |
|---|---|---|
| ① | 跳过 `req.is_retracted` | 被撤回的请求还在 batch 里，它的行写进训练集就是脏数据 |
| ② | `p < visible_end`（用 `finished_len`） | stop-string 截断后，采集到的行比实际产出长，尾部被当成有效向量 |
| ③ | 去重键用 `(layer, position)` 而不是 `(target_index, position)` | 两请求用同一 target 下标但不同层时会串 |

`consume` 的既有语义正好匹配：它在「采样 token 已提交到每个请求之后、
响应流式发送之前」运行，并且 **prefill 的 chunked 中间块也会调用它**
（`batch_result_processor.py:448`），所以逐块累积不需要额外分支——
这正是前作 A 必须专门打 `_maybe_collect_customized_info` 补丁才能做到的事。

---

## 6. 请求契约

### 6.1 Schema

```python
feature_capture = {
    "layers": [-1],                  # 必填。block boundary，支持负索引
    "site": "block_output",          # 可选，v1 仅此一项
    "positions": {"mode": "last"},   # 可选，见 6.2
    "encoding": "base64",            # 可选，base64 | float_list
}
```

校验（TokenizerManager 里，分词后）：

- `layers` 非空、去重、每项 `-N-1 ≤ L ≤ N`、解析负索引、个数 ≤ `INSCOPE_MAX_TARGETS`（默认 8）；
- `site` 不是 `block_output` → 拒绝（不是静默忽略）；
- 预算：`len(layers) × len(positions) × hidden × 4 ≤ INSCOPE_MAX_BYTES`（默认 128 MiB，按 FP32 保守估）。

### 6.2 位置四种模式

用户明确要求：**suffix 不一定存在，显式位置必须是一等公民；用 suffix 时取它的最后一个位置。**

| mode | 字段 | 语义 | 用途 |
|---|---|---|---|
| `last` | — | prompt 最后一个 token（默认） | SafeGauge 常规 |
| `explicit` | `index: [int, ...]` | 显式位置，**支持负索引**（`-1` = 末 token） | 指定 token 探测 |
| `suffix_last` | `suffix_token_ids: [int]` 或 `suffix_text: str` | 定位该子序列**最后一次出现**，取其最后一个 token | 校验模板未变 |
| `all` | — | 全部 prompt token | 训练集采集 |

```json
{"mode": "explicit", "index": [12, 13, -1]}
{"mode": "suffix_last", "suffix_token_ids": [151657, 198, 32313]}
```

**`suffix_last` 不是 `last` 的别名，它是校验器。**
模板变了、suffix 被截断、后面又被追加了内容时，`last` 会在错误位置**静默**采集，
`suffix_last` 会直接失败。探测实验最怕这种静默错位，所以：

- 找不到 → 抛 `FeatureCaptureSuffixNotFound`，**不降级成 `last`**；
- 出现多次 → 取最后一次，并把出现次数记进响应元信息（便于发现异常）；
- `suffix_text` 需要 `tokenizer.encode(..., add_special_tokens=False)`，也在 TokenizerManager 做。

### 6.3 解析时机

TokenizerManager 分词之后（那里才有含 chat template 与特殊 token 的 `origin_input_ids`），
`spec.validate_and_resolve()` 把模式解析成**绝对位置元组**，写进
`TokenizedGenerateReqInput.feature_capture`。之后调度与执行阶段只认绝对位置，
不需要 tokenizer，`plan.py` 也因此能脱离 tokenizer 做纯 CPU 单测。

### 6.4 不对齐的语义明确排除

v1 **不实现** `generated: processed`。只采 prompt token。

理由：SafeGauge 的探测对象是「上下文 + suffix」这类完整输入，
需要的正是 prompt 末 token 的表示；生成过程中的状态不在主用例里，
而实现它要动 decode 的 CUDA Graph（见 §10）。

文档里要写清楚：**不返回**「用于预测某 token 的前一个位置」这种语义，
也**不为补齐最后一个生成 token 的状态而额外跑一次 forward**。

---

## 7. 计划与行映射

### 7.1 数据结构

```python
@dataclass(frozen=True)
class FeaturePlanEntry:
    rid: str
    layers: tuple[int, ...]        # 解析后的 block boundary
    row_start: int                 # 该请求本轮在 batch 里的起始行
    position_start: int            # 该行对应的起始绝对位置
    positions: tuple[int, ...]     # 本轮真正要采的绝对位置（已求交）

# ForwardBatch 新增
feature_specs: Optional[List[Optional[FeatureRequestSpec]]] = None   # 与 rids 对齐
feature_plan: Optional[tuple[FeaturePlanEntry, ...]] = None
```

`ForwardBatch` **不持有** `reqs`，只保存从 `ScheduleBatch.reqs` 派生的逐请求列表
（`rids`、`lora_ids` 都是这么来的，`forward_batch_info.py:1074`）。
`feature_specs` 同样在 `init_new` 派生，`plan.build()` 只读
`rids` / `feature_specs` / `extend_seq_lens_cpu` / `seq_lens_cpu`。

### 7.2 行映射（把不采集的请求也计入）

前作 A 和 D 都特别强调过这一点，保持一致：

> 某次 prefill 中 A/B/C 分别占 5/4/3 行，A 和 C 取本轮最后 token，
> 应读第 4、11 行，**不能因为 B 不采集就跳过它的行数**。

```python
def build(forward_batch):
    entries, offset = [], 0
    extend_lens = forward_batch.extend_seq_lens_cpu
    seq_lens = forward_batch.seq_lens_cpu
    specs = forward_batch.feature_specs or ()
    for i, rid in enumerate(forward_batch.rids):
        n = int(extend_lens[i])
        spec = specs[i] if i < len(specs) else None
        if spec is not None:
            position_start = int(seq_lens[i]) - n
            wanted = tuple(p for p in spec.positions
                           if position_start <= p < position_start + n)
            if wanted:
                entries.append(FeaturePlanEntry(rid, spec.layers, offset,
                                                position_start, wanted))
        offset += n                      # ← 无条件推进
    return tuple(entries) or None
```

### 7.3 按槽位归组

一个 batch 里不同请求选不同层，所以槽位是多层集合的**并集**：

```python
by_slot = {}                       # slot -> [(entry, positions), ...]
for entry in plan:
    for L in entry.layers:
        by_slot.setdefault(L, []).append(entry)
```

每个槽位分到输出缓冲的一段连续区间；`rows` 张量就是
`entry.row_start + p - entry.position_start` 的拼接。
未被任何请求选中的槽位不产生 `rows`，`bridge.layer_capture` 返回既有
auxiliary hook 或 `None`，不创建 feature tensor，也不执行 feature D2H。

### 7.4 chunked prefill

逐块累积，只在包含目标位置的块里采。去重键是
`(layer, site, absolute_position)`——不能只用 row，因为重算会让同一位置出现两次。

### 7.5 prefill-only 模式

直接复用 `ScheduleBatch.is_prefill_only`（`max_new_tokens == 0 and spec_alg is None`）。
已验证该路径在 `tp_worker.py:714` 会跳过 `sample()` 并造 CPU dummy token ids，
**不需要新建完成路径**（前作 A 为此写了 `FeatureBatchOutput` 和新的 scheduler 分支）。

| 请求形态 | 行为 |
|---|---|
| `max_new_tokens=0` + `feature_capture` | `prefill_only`：只出特征，跳过 decode/采样/logits |
| 普通请求 + `feature_capture` | `generate+features`：正常生成 + 采集 prompt 特征 |
| 普通请求 | 完全不变 |
| `max_new_tokens=0` 且**无** `feature_capture` | 拒绝：否则请求什么都不产出 |

---

## 8. 缓存边界（正确性关键）

Prefix cache 只存 KV，不存中间层向量。目标 token 若完全命中缓存、没有重新 forward，
就没有新向量——会**静默返回错误的（旧的或缺失的）结果**。

```python
def feature_prefix_limit(req, limit):
    spec = getattr(req, "feature_capture", None)
    if spec is None:
        return limit
    seen = getattr(req, "_feature_seen_positions", set())   # 已采过的绝对位置
    missing = [p for p in spec.positions if p not in seen]
    if missing:
        return min(limit, missing[0]) if limit is not None else missing[0]
    return limit
```

调用点：`schedule_batch.py` 算出 `key_limit` 之后一行
`key_limit = feature_prefix_limit(self, key_limit)`。

与两个前作的差别：

- 前作 A 用固定 margin 512，保守但白算最多 512 token；
- 前作 D 用精确的「最早未采集位置」，我们不额外引入静态集合也会漏掉
  **重算/撤回（retraction）导致的重复采集**，所以额外维护 `_feature_seen_positions`。

**效果**：`positions = {"mode": "last"}` 时，`missing[0] = prompt_len - 1`，
前缀可以复用到倒数第二个 token，只重算最后 1 个 token。
这是 SafeGauge 的常态，也是为什么不能粗暴关掉 radix cache。

**必须有的测试（C.4）**：同一 prompt 连发两次（第二次全命中），
两次返回的所有 `(layer, position)` 向量必须 `torch.testing.assert_close` 相等。
不测这条，缓存命中时的静默错位几乎必然漏掉。

---

## 9. 数值正确性

| 项 | 要求 |
|---|---|
| 采集点定义 | 见 §3.1；`L<N` 与 HF `hidden_states[L]` 同义，`L=N` 是 pre-final-norm block output |
| 参考基准 | HF `transformers` BF16；普通层读 `hidden_states`，末层用 decoder block hook |
| 断言 | `cos ≥ 0.9999`、`rel_L2 ≤ 2e-2`；不要求逐位相等 |
| FP8 | **禁止**与 BF16 断言逐位相等；只断言契约（shape/layer/position/dtype/无串扰）+ 报告误差分布 |
| 跨请求污染 | 同批不同请求选不同层/位置，结果逐请求独立 |
| 空批次 | 无采集请求的 batch 不产生额外张量，`feature_device_output is None` |
| 行数自检 | `value.shape[0] == sum(extend_seq_lens)`，不符即抛错（§4.3） |

前作 D 在真实 Qwen3-8B 上做到最低余弦 0.9995，可作为阈值现实性的旁证。
但那是两个引擎之间的差异，我们的自对照应更紧；**5e-3 是 betweeen-engine 的量级，
同引擎不同层位对照应能到 1e-3**，实现时按实测收紧。

---

## 10. 执行模式与 graph

| 阶段 | prefill | decode | 说明 |
|---|---|---|---|
| **v1 (P2–P3)** | 有采集行 → eager；无采集行 → 官方 prefill graph | 不采集 | 先保正确 |
| P4（可选） | 固定行容量 prefill graph | — | 仅当 profiling 证明收益显著 |
| 不做 | — | GPU 选择表 + graph 复用 | 前作 D 的方案；SafeGauge 用不到，等有真实需求再评估 |

强制 eager 的条件只有一条：

```python
if forward_batch.feature_plan is not None:   # 本批有真实采集行
    # 跳过 prefill / decode cuda graph，走 eager
```

`feature_plan is None` 时行为与上游**逐字节相同**，这是"不影响普通请求"的硬保证。

---

## 11. 能力矩阵与启动拒绝

`capabilities.initialize(runner)` 在 ModelRunner 初始化时一次性检查，**启动即失败**。
理由：采集时机与并行方式的耦合无法在运行期安全降级，
静默降级的后果是返回"看起来合理但语义错"的向量。

| 拒绝项 | 原因 |
|---|---|
| 非 CUDA | 未验证 |
| `pp_size != 1` | PP 下 residual 跨 rank，语义未验证（且 `_copy_auxiliary_output_to_cpu` 在 PP 下直接 return） |
| `enable_dp_attention` | 行映射假设单 DP |
| `attn_cp_size != 1` / `dcp_size != 1` | 序列被切分，行不再是完整 feature row |
| `enable_two_batch_overlap` | TBO 拆分 forward，采集时机不确定 |
| `enable_attn_tp_input_scattered` / `enable_layernorm_sp` | 同上 |
| spec decoding 启用 | 草稿模型并行 forward，语义不明 |
| `disaggregation_mode != "null"` | PD 分离 |
| 模型类无 `feature_capture_sites` | 未适配 |
| 请求带多模态输入 | v1 只做文本 |
| `is_split_prefill` | 该路径的 `final_norm` 是另一处调用，未验证 |

**能力探测用类属性，不用字符串表**：

```python
sites = getattr(model, "feature_capture_sites", ())
if "block_output" not in sites:
    raise ValueError(
        f"{type(model).__name__} has no verified feature-capture adapter; "
        f"add a layer_capture() call and this class attribute")
```

子类自动继承，加模型忘写调用会立刻暴露（因为层数对不上或 hooks 从不触发）。

层数 / 隐藏维一律从 `hf_config`（有 `text_config` 时取其）读取，不硬编码。

---

## 12. 文件改动清单

### 12.1 新增（模型无关，共 ~760 行）

| 文件 | 行数 | 内容 |
|---|---:|---|
| `feature_capture/__init__.py` | 稳定公共导出 |
| `feature_capture/spec.py` | 契约、校验、四种位置模式、suffix 定位 |
| `feature_capture/plan.py` | 行映射与槽位归组 |
| `feature_capture/recorder.py` | 设备缓冲、Device/Host 输出和 `consume` |
| `feature_capture/encoding.py` | base64 / float_list 编码 |
| `feature_capture/capabilities.py` | 启动拒绝、进程开关与能力探测 |
| `feature_capture/bridge.py` | 模型边界 callback 合并 |

无 `kernels.py`（见 D4）。

### 12.2 上游接线

| 区域 | 职责 |
|---|---|
| `entrypoints/` + `managers/io_struct.py` | Engine API 与 IPC 字段 |
| `tokenizer_manager.py` | 分词后解析 suffix/绝对位置并校验 spec |
| `schedule_batch.py` + `forward_batch_info.py` | Req 状态、radix prefix 上界与 extend slot plan |
| `model_runner.py` + `logits_processor.py` + `managers/utils.py` | capability gate、graph 准入与 auxiliary D2H 合并 |
| `output_streamer.py` | 将 `features` 作为 request-level metadata 发送 |
| `models/qwen3.py` + `models/qwen3_5.py` | capability 声明与 layer-boundary 薄适配 |

**对照**：前作 A 17 文件 +887 行且无测试；前作 D 18 文件 +424 行运行时 + 自写 D2H。
我们的模型侧改动从「改计算逻辑」降到「换一个函数名」。

---

## 13. 阶段与验收

前置 P0 已完成：Conda `inscope` 环境使用 editable 方式安装当前源码。

| 阶段 | 内容 | 验收（全过才进下一阶段） |
|---|---|---|
| **P0** | 建环境，`import sglang` | `test/srt` 里与 `aux_hidden_states` 相关的既有单测通过 |
| **P1** | `spec.py`/`capabilities.py`/`plan.py` + 纯 CPU 单测 | 四种位置模式、负索引、suffix 缺失报错、行映射（含"中间请求不采集"）、chunked 去重 |
| **P2** | Qwen3-8B-FP8 `prefill_only`，**交付面 = `Engine.generate()`**，`L ∈ {0,1,20,35,36}` | **B.1b 层号语义全过**（最关键）；无采集 batch 无额外张量；schema 符合 |
| **P3** | 多层 + 多位置 + chunked + 缓存命中 + 并发混合 | C 组全过；**缓存二次采集逐元素相等**；不采集请求不受影响 |
| **P4** | `generate+features`（仍只采 prompt） | 文本与纯生成一致；eager 与无采集路径结果一致；`feature_plan is None` 时零影响 |
| **P5** | 第二架构 **`Qwen3.5-9B`（Dense）** | 只补 capability 和 layer-boundary 薄适配，全流程通过 |
| **P6** | SafeGauge 式探测实验（**外部仓**） | 见 `test/manual/feature_capture/DESIGN.md` |
| **P7**（推迟） | OpenAI 兼容层（`/v1/completions`、`/v1/chat/completions`） | 见 §15.2；SafeGauge 不需要，等有真实调用方再做 |

**P2 的交付面已定为 `Engine.generate()`**（R1）。原生 `/generate` 顺带可用，
不需额外代码（§15.1）；OpenAI 层不在本阶段。

**P5 是解耦的验收关口**：若加第二个架构需要动 `feature_capture/` 任何一行，
说明分层没做对，**回退重构而不是加特例**。

---

## 14. 测试矩阵

细节见 `test/manual/feature_capture/DESIGN.md`，此处只列与实现强绑定的项。

| 组 | 用例 | 保护的不变量 |
|---|---|---|
| A | 契约单测（无 GPU） | spec 校验、位置解析、行映射、chunked 去重 |
| A | **源码契约测试** | `feature_capture/` 内无模型名；有 `feature_capture_sites` 的模型文件必含 `layer_capture(` |
| A | 能力拒绝 | §11 每项都被拒绝且错误可读 |
| B.1 | **层号映射** | `L=0/1/20/35/36` 对 HF；`35≠36`；`36` 是 pre-norm |
| B.2 | 空批次 / 单请求 / 全采集 | 无额外分配、无串扰 |
| C.1 | 多层多位置 E2E（FP8） | schema + 契约 |
| C.2 | chunked 跨块 | 位置去重、结果完整 |
| C.3 | 并发混合（采/不采/不同层） | 逐请求独立 |
| C.4 | **缓存命中重放** | 二次采集与首次逐元素相等 |
| D | BF16 参考对照 | `cos`/`rel_L2` |
| E | probe 训练 | 见 test DESIGN |

**禁止**：FP8 与 BF16 断言相等；不同 `position_mode` 的记录混在一起比较。

---

## 15. 已定决策（2026-10-19 评审确认）

| # | 决策 | 结论 |
|---|---|---|
| **R1** | 首版交付面 | **只做 Python `Engine.generate()`**。原生 HTTP `/generate` 顺带免费获得（同一 dict，见 §15.1）；OpenAI 兼容层推迟到 P7 |
| **R2** | P5 第二架构 | **只做 `Qwen3.5-9B`（Dense，32 层，混合注意力）**，`27B` 不列入本阶段 |
| **R3** | probe 代码归属 | **在外部仓**。本仓只保留 `test/manual/feature_capture/` 作为测试与契约 |

R1/R2/R3 生效后，§12 的上游接线不包含 `protocol.py`/`serving_*`。
文档显式记录哪些接口已可用、哪些被推迟，避免后来的人误以为遗漏。

### 15.1 为什么原生 HTTP `/generate` 是免费的

`http_server.py:968` 与 `Engine` 走的是**同一个** `tokenizer_manager.generate_request(...)`，
而且它把结果整个丢给 `orjson_response(ret)`：

```python
ret = await _global_state.tokenizer_manager.generate_request(obj, request).__anext__()
return orjson_response(ret)          # meta_info 原样序列化
```

所以 `meta_info["features"]` 一旦在 `TokenizerManager` 里写好，
`/generate` 不做任何改动就能返回它。数据是 base64 字符串，`orjson` 直接可序列化。

### 15.2 为什么 OpenAI 层不是免费的（推迟到 P7）

除了实现成本，**部署形态本身就是独立的**：特征提取是单独拉起的一个进程，
不与线上服务侧链接，调用方是 Python。因此 OpenAI 兼容层在这个形态里没有调用方。

实现成本方面：`serving_completions.py` / `serving_chat.py` **不原样透传 `meta_info`**，
而是逐字段挑选、构造 `CompletionResponseChoice` / `ChatCompletionResponseChoice`：

```python
hidden_states[index] = content["meta_info"].get("hidden_states", None)
routed_experts[index] = content["meta_info"].get("routed_experts", None)
```

要接入 `features` 需要动 4 类位置：

| 文件 | 改动 |
|---|---|
| `openai/protocol.py` | choice 类加 `features` 字段 + 空值 `data.pop("features")`（比照 `hidden_states` 的写法） |
| `openai/serving_completions.py` | 非流式映射 + 流式 chunk 映射 |
| `openai/serving_chat.py` | 非流式映射 + 流式 delta 映射（chat 的 61 处 `meta_info` 引用说明这里分支很多） |

工作量与「一次完整功能」相当，而 SafeGauge 的调用方是 Python，
收益为零。因此明确推迟，且**不预留半成品代码**（避免协议里出现永远为 `None` 的字段）。

### 15.3 技术风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 层号映射错（norm 前后或差一层） | 结果"看起来合理"地整体错位 | B.1b（`test/manual/feature_capture/DESIGN.md`），P2 必过；`35≠36`、`36` 是 pre-norm、`0≠1` 三条专项断言 |
| 缓存命中静默返回旧/错向量 | 训练集被污染，事后极难发现 | `_feature_seen_positions` + C.4 |
| `owned` 误判导致存储被覆写 | 偶发数值错误，随机复现 | v1 一律复制 |
| 行数与预期不符（DP padding 等漏进来） | 静默按错行取 | §4.3 的 `expected_rows` 硬断言 |
| `final_norm` 的 capture 拿到 norm 后张量 | `cos` 仍很高但是错的 | B.1b 第 3 条 |
| MoE 数值未对齐 | 前作 D 已失败一次 | v1 直接拒绝 |

### 15.4 明确不做（避免范围蔓延）

- OpenAI 兼容层（P7，见 15.2）
- decode 阶段采集（GPU 选择表 / graph 动态换层）
- `site = ffn_output`（要逐模型声明，且要处理 all-reduce 融合）
- no-KV 池（djs canary 的设计稿；省几 GB 显存但风险高，P3 通了再作独立实验）
- MoE / PP / DP attention / CP / TBO / 多模态
- 结果落 safetensors 到磁盘（走内存回传，落盘是客户端责任）
