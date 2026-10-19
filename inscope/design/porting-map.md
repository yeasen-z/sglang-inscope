# 核心实现移植对照表

本文回答一个问题：**前作（djs FeaturePrefill、zyt CoolWatch）的核心机制，
逐项落到本基线时对应什么？哪些能直接搬，哪些必须重写，哪些可以删掉。**

前作原始调研见 [prior-art.md](prior-art.md)；本基线设计见
[implementation-design.md](implementation-design.md)。

---

## 0. 一句话总结

前作各自手写了一条「采集 → 传输 → 回传」的完整通路，因为当时的 sglang 没有抽象。
本基线在 `layers/layer_boundary/` 和 `managers/auxiliary_output.py` 里已经有这套抽象，
**所以移植的工作性质不是「搬代码」，而是「把他们的三段落点重新对齐到基线的三个契约上」**。

| 前作的三段 | 他们各自实现 | 本基线对应契约 |
|---|---|---|
| 在哪取、取哪几行 | 改模型层循环内的计算 | `capture(value, *, owned)` 回调 |
| 怎么传回主机 | 自建输出结构 + 自写 D2H | `DeviceAuxiliaryOutput.copy_to_host` |
| 怎么拆给请求 | 自建 streamer 分支 + 自写 IPC | `HostAuxiliaryOutput.consume(batch, commits)` |

---

## 1. 采集点：从「改计算」到「挂回调」

### 1.1 前作怎么做

**djs**（`models/qwen2.py`，Qwen3 继承它）——直接在层循环里改计算：

```python
for i in range(self.start_layer, self.end_layer):
    if not feature_only and i in self.layers_to_capture:
        aux_hidden_states.append(
            hidden_states + residual if residual is not None else hidden_states
        )
    hidden_states, residual = self.layers[i](...)
    if feature_only and i + 1 in self.layers_to_capture:
        selected = hidden_states + residual if residual is not None else hidden_states
        row_indices = getattr(forward_batch, "fyh_feature_row_indices", None)
        aux_hidden_states.append(
            selected.index_select(0, row_indices) if row_indices is not None else selected
        )
        if i + 1 == capture_stop_layer:
            break
```

**zyt**（`models/qwen2.py`）——插一个 `tap()`：

```python
activation_capture.tap(layer, hidden, residual, forward_batch)
```

两者都必须**手工 `hidden + residual`**。zyt 为了拿到「完整 FFN 输出」，
还必须在 `layers/communicator.py` 里**关掉 MLP all-reduce 的延迟融合**，
否则 tap 到的 FFN 输出还没完成跨卡通信。

### 1.2 本基线对应

`layers/layer_boundary/stage.py:_prepare_attention` 已经把这个点封成回调：

```python
capture(
    value,                      # 已完成通信的 residual stream，[tokens, hidden]
    owned=entry.capture_move_allocates or (...),
)
```

`value` 就是**上一层 block 输出、pre-norm 残差流**，residual 已经合好。
并且——这是关键的——合并发生在**融合 kernel 内部，保留 FP32 累加**。
手工 `hidden + residual` 会绕开它，得到的是「先把两项舍入到 bf16 再相加」，
与 `access.py` 注释里明确警告的错误做法相同：

> Keep add and norm together: normalizing a separately rounded residual is not
> numerically equivalent to the fused kernel's FP32 accumulation.

### 1.3 结论

| | djs | zyt | 我们 |
|---|---|---|---|
| 插桩位置 | 层循环内改计算 | 层循环内加 `tap()` | 换一个 `capture=` 实参 |
| residual 合并 | 手写 `h + r` | 手写 `h + r` | 基线融合 kernel |
| 是否要关 all-reduce 融合 | 否（它取 block 输出） | **是**（它要 ffn_output） | **否** |
| 模型改动 | qwen2 +59, qwen3 +163 | qwen2 数十行 | **2 行** |

**可以删掉的前作机制**：`capture_stop_layer` 早停、`fyh_feature_row_indices` 侧信道、
`set_capture_enabled` 遍历模块、`_ftd_saved_layers_to_capture` 存取。

早停那条要单独说：djs 用 `break` 跳出层循环，是想省掉后面层的计算。
但我们**不能这么做**——层循环是整个 batch 共享的，
batch 里还有不采集的请求和别的请求要的更高层。
zyt 也没做早停。所以 djs 的早停只在「整批都是同一个 feature-only 请求」时成立，
这正是它必须独占 batch（`feature requests with different capture specs cannot share a forward batch`）的原因。
我们不需要这个限制，代价是最高目标层之后的层照常算——
而它们本来就要算（有别的请求在用）。

---

## 2. 行选择：从「自己算偏移」到「槽位布局」

### 2.1 前作怎么做

**djs**：`row_indices` 从 `forward_batch` 侧信道读，在模型层里 `index_select`。
偏移计算散在 `output_streamer` 里，靠 `FeatureRecord.row_start/row_count` 事后拼。

**zyt**：`make_plan` 算 `row_start`，`recorder.begin` 按层聚组：

```python
for i, req in enumerate(batch.reqs):
    length = 1 if decode else int(batch.extend_lens[i])
    start = int(batch.seq_lens_cpu[i]) - 1 if decode else int(batch.prefix_lens[i])
    ...
    row += length          # ← 无条件推进，这是关键
```

### 2.2 本基线对应

同一套算术，已经实现在 `feature_capture/plan.py`，39 个纯 CPU 测试覆盖，
其中最关键的一条用例就是「中间请求不采集也不能跳过它的行数」。

**一处改进**：zyt 用 `batch.prefix_lens[i]` 作 `position_start`。
本基线 `ForwardBatch` 上已经有 `extend_prefix_lens_cpu`（`forward_batch_info.py:661`），
比 `seq_lens - extend_len` 更直接、不用假设两者一致。设计文档里改成优先用它。

### 2.3 结论

可直接搬**算法**，不可搬**位置**——他们写在 `runtime.py` 的 `make_plan` 里，
我们写在独立的 `plan.py`，因为 `ForwardBatch` 不持有 `reqs`，
只能读到 `init_new` 派生出来的逐请求列表。这个差异让 `plan.py` 能脱离调度器单测。

---

## 3. 传输：从「自建结构 + 自写 D2H」到「复用官方协议」

这是省得最多的一段，也是两份前作踩坑最集中的一段。

### 3.1 djs 自建了整套输出通道

`FeatureBatchOutput`（`io_struct.py` 新增）+ `FeatureRecord`，
`output_streamer` 新增 105 行 `_stream_output_features`：

```python
feature_reqs = [req for req in reqs if req.extra_key.startswith("fyh-feature:")]
if feature_reqs:
    self._stream_output_features(feature_reqs)
reqs = [req for req in reqs if req not in feature_reqs]     # 从正常流里摘出来
```

然后在这 105 行里它必须自己处理：跨块 payload 收集、`layer_ids` 一致性校验、
`dtype` 一致性校验、按 position 重组（`selected_rows.update(...)`）、
`torch.cat`、`.view(torch.uint8).numpy().tobytes()`。
还额外要改 `batch_result_processor`，否则**chunked prefill 的中间块 payload 会丢**：

```python
else:
    # being chunked reqs' prefill is not finished
    if isinstance(req.extra_key, str) and req.extra_key.startswith("fyh-feature:"):
        self._maybe_collect_customized_info(i, req, logits_output)
```

### 3.2 zyt 自建了 Device/Host 输出

`runtime.py` 的 `DeviceOutput` / `HostOutput`，加 `CompositeDevice/CompositeHost` 做组合：

```python
@dataclass
class DeviceOutput:
    def copy_to_host(self, copy):
        return HostOutput(copy(self.tensor), self.segments, self.execution)
```

还有一条重要的踩坑记录：

> `ModelRunner.sample()` 会清空原有的 `auxiliary_device_output`。
> 如果直接把采集结果放进去，到了采样阶段就会丢失。
> 因此先用 `activation_device_output` 保存，在 D2H 拷贝阶段再与原有辅助输出合并。

它为此新增字段 + 手写 `CompositeDevice.copy_to_host` / `CompositeHost.consume`。

### 3.3 本基线对应

**这两件事基线都已经做了，而且做完了**：

| 他们手写的 | 基线已有 | 位置 |
|---|---|---|
| `DeviceOutput.copy_to_host` | `DeviceAuxiliaryOutput` Protocol + `_async_d2h` | `auxiliary_output.py:20`、`utils.py:225` |
| `CompositeDevice/CompositeHost` | `append_auxiliary_output` + `CompositeDeviceAuxiliaryOutput` | `auxiliary_output.py:62-93` |
| 「怕被 sample 清空」的规避字段 | 同样需要，但**只加一个字段，不手写组合** | `logits_processor.py` +1 字段 |
| chunked 中间块收集 | `consume` 在 prefill 中间块也会被调用 | `batch_result_processor.py:448` |
| `torch.cat` 拼批 | 不需要拼，每个 slot 一段独立缓冲 | — |
| retraction 去重 | `consume(batch, commits)` 提供 `commits`，已采集合由我们维护 | — |

已核对基线的调用顺序，正好满足我们的需求：

```python
# batch_result_processor.py:448（prefill，含 chunked 中间块）
# batch_result_processor.py:1083（decode）
if auxiliary_output is not None:
    self.consume_auxiliary_output(batch, auxiliary_output, auxiliary_output_starts)
...
self.output_streamer.stream_output(batch.reqs, batch.return_logprob)   # ← consume 之后
```

这与 `auxiliary_output.py` 里的契约文档一字不差：

> ``consume`` runs after sampled tokens have been committed to each request
> and immediately before response streaming.

### 3.4 zyt 的 `consume` 有两个我没写进去的细节，要采纳

读它的实现（`runtime.py:225`）发现两处值得直接搬：

**① 过滤掉「超出已提交范围」的位置**

```python
end = len(req.origin_input_ids) + (
    req.finished_len if req.finished_len is not None else len(req.output_ids))
chosen = [i for i, p in enumerate(seg.positions) if p < end and (idx, p) not in seen]
```

`finished_len` 是 stop-string 命中等情况下**实际可见**的产出长度
（`schedule_batch.py:1832` 设置）。采集到的行可能比它长，
不过滤就会把「被截断的尾部」当成有效向量写进训练集。

**② 跳过已撤回的请求**

```python
if req is None or req.is_retracted:
    continue
```

撤回的请求还在 batch 里，但它的行已经不有效。

**③ tap 时检查连续性和 stride**

```python
if hidden.ndim != 2 or hidden.shape[1] != self.width or hidden.stride(1) != 1:
    raise RuntimeError("activation adapter expected complete contiguous feature rows")
if residual is not None and (residual.shape != hidden.shape or residual.stride(1) != 1):
    raise RuntimeError("activation adapter received a sharded or incompatible residual")
```

`stride(1) != 1` 这条比只看 shape 强：一个被切过的视图 shape 可能对，但取行会拿到错数据。

结论：**结构复用基线，`consume` 内部逻辑照搬 zyt 的三条防御**。

---

## 4. 缓存边界：两套策略合起来

### 4.1 前作怎么做

**djs**：固定 margin，压 `key_limit`：

```python
if extra_key.startswith('fyh-feature:'):
    first = min(p if p >= 0 else input_len + p for p in spec['selected_positions'])
    first = max(0, first - FEATURE_CACHE_RECOMPUTE_TOKENS + 1)   # 512
    key_limit = min(key_limit, first)
```

它自己也知道这个不精确，所以留了个 canary 补丁
`feature-cache-tail-margin-1.patch` 想把 512 收到 1，且要求对比 radix 关闭的参考。

**zyt**：精确到最早未采集位置：

```python
def prefix_limit(req, limit):
    seen = getattr(req, "_activation_seen", set())
    wanted = selected_positions(spec, 0, len(req.origin_input_ids), len(req.origin_input_ids))
    missing = [p for p in wanted if (0, p) not in seen]
    if missing:
        return min(limit, missing[0]) if limit is not None else missing[0]
    return limit
```

### 4.2 本基线对应

取 zyt 的精确策略（djs 的 512 会白算最多 512 token），
但**去重键要带上层号**：`(layer, position)`，不是只有 `(0, position)`。
zyt 的 `_activation_seen` 是 `(target_index, p)`——
`target_index` 是 target 列表里的下标，不是层号，
所以两个请求用同一个 target 下标但不同层时会串。我们的键用真实层号。

### 4.3 结论

| | djs | zyt | 我们 |
|---|---|---|---|
| 策略 | 固定 margin 512 | 精确最早未采集 | 精确 |
| 去重键 | 无 | `(target_index, p)` | `(layer, position)` |
| 白算 token | 最多 512 | 0 | 0 |

---

## 5. 完成路径：三种模式怎么落

### 5.1 前作怎么做

**djs**：为 feature-only 新建了一条完成路径——`FeatureBatchOutput` +
`scheduler` 分支 + `prefill_only` 时 `LogitsProcessorOutput(next_token_logits=None)`：

```python
if feature_only:
    output = LogitsProcessorOutput(next_token_logits=None)   # 不产生 logits
else:
    output = self.logits_processor(...)
```

**zyt**：不新建路径，复用 `max_new_tokens=1` 的正常生成入口，分类只消费 prompt 向量：

> 当前请求仍使用正常生成入口，设置 `max_new_tokens=1`；分类只消费 prompt 向量，不使用这个生成 token。

代价是每次探测都白算一个 decode 步。

### 5.2 本基线对应

基线**已经有** `is_prefill_only`（`max_new_tokens == 0 and spec_alg is None`），
且 `tp_worker.py:714` 已经为它跳过采样、造 CPU dummy token ids：

```python
if not forward_batch.is_prefill_only:
    batch_result.next_token_ids = self.model_runner.sample(logits_output, forward_batch)
else:
    batch_result.next_token_ids = torch.zeros(...)
```

所以 djs 新建的那条路**基线已有**，zyt 的「白算一个 token」也**不必付**。

### 5.3 结论

| 模式 | djs | zyt | 我们 |
|---|---|---|---|
| 仅特征 | 新建 `FeatureBatchOutput` 通路 | 用 `max_new_tokens=1` 白算 | `is_prefill_only`，已有 |
| 生成+特征 | 同一通路 | 同一条路 | 走正常生成，`feature_plan` 非空时强制 eager |

---

## 6. GPU Graph：三种策略的取舍

| | djs | zyt | 我们 v1 |
|---|---|---|---|
| decode | `if can_run_graph and not capture_requested` → 强制 eager | GPU 选择表 `int32[层数, 2, 容量]`，graph 内动态换层 | 不采 decode |
| prefill | 固定行容量 graph（`max_bs * len(positions)`），spec 匹配才 replay | 有采集行 → eager | 有采集行 → eager |
| 换层是否要重捕 | 是（除非 spec 匹配） | 否 | 不涉及 |

zyt 的 decode 选择表是四套里唯一真正解决「graph 内动态换层」的方案，
但它为此付出的代价写在它自己的 README 里：

> 未选中的层只检查表项，不读取整个隐藏向量。不过，检查 kernel 仍然存在，
> 所以开启采集能力的 decode graph 有固定开销。普通 chat 实例关闭采集开关，避免承担这部分开销。

而我们 **v1 不采 decode**，所以这块成本我们完全不付。
如果以后要采生成 token 的表示，zyt 的表结构是现成的参考，不是现在的工作。

---

## 7. 移植后我们能删掉的前作代码

| 前作代码 | 为什么可以删 |
|---|---|
| `ftd_activation.set_capture_enabled` 遍历 modules | 我们不碰 `capture_aux_hidden_states` |
| `_ftd_saved_layers_to_capture` 存取 | 同上 |
| `configure_model()` 的 prev_enabled 快照/恢复 | 同上 |
| djs 的 tuple 解包畸形分支 | 我们不让内层返回 tuple |
| `FeatureBatchOutput` / `FeatureRecord` / `_stream_output_features` | 基线有 `consume` + `meta_info` |
| `_maybe_collect_customized_info` 的特例 | 基线 `consume` 在中间块就调 |
| `CompositeDevice.copy_to_host` / `CompositeHost.consume` | 基线有 `append_auxiliary_output` |
| Triton `capture_extend` / `capture_decode` | `index_select` 够用（我们的 `value` 已在显存） |
| 关闭 MLP all-reduce 延迟融合 | 我们取 block 输出，不取 ffn_output |
| decode GPU 选择表 | v1 不采 decode |
| `extra_key` 前缀解析 | 一等请求字段 |

---

## 8. 唯一必须新增、两个前作都没有的

**`suffix_last` 位置模式。** 前作都只有「prompt 最后 token」和「显式位置列表」。
用户要求 suffix 可选、但用 suffix 时取其最后一个位置，
且显式位置必须是一等公民——这个组合前作没有，属于新增能力。
它同时是**校验器**（找不到就失败，不降级），见 `spec.py` 的
`FeatureCaptureSuffixNotFound` 与对应测试。

---

## 9. 移植风险清单

| 风险 | 来源 | 我们的处理 |
|---|---|---|
| 手工加 residual 的数值偏差 | djs/zyt 都这么做 | 用基线 `capture`，不手工加 |
| `sample()` 清空 aux 输出 | zyt 踩过 | 独立 `feature_device_output` + `append_auxiliary_output` |
| chunked 中间块 payload 丢失 | djs 踩过 | 基线 `consume` 在中间块就调，已验证 |
| 裁掉的尾部被当成有效向量 | zyt 处理过 | 采用 `p < finished_len` 过滤 |
| 撤回请求的行 | zyt 处理过 | 采用 `req.is_retracted` 跳过 |
| 非连续视图 | zyt 处理过 | 采用 `stride(1) != 1` 检查 |
| 缓存命中静默错位 | djs 用 margin 绕过 | 精确 `(layer, position)` 去重 + C.4 用例 |
| 模型属性被全局改写 | djs/zyt 都这么干 | 完全不碰，纯 batch 级状态 |
