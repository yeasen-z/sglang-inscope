# 前作调研：SGLang 侧隐藏层采集的四套实现

调研日期 2026-10-19。本文只记录**别人已经做过什么、代价是什么**，结论用于
[implementation-design.md](implementation-design.md)。

四套实现都在本机，都不在 `yeasen-z/sglang-inscope` 里。

| # | 位置 | 基线 | 规模 | 作者 |
|---|---|---|---|---|
| A | `/mnt/kpfs_qyang/djs/from-zhisuanyun-g03/follow-your-heart/patches/sglang-0.5.18-fyh-direct/feature-prefill-v0.5.18.patch` | v0.5.18 `71de97b2` | 17 文件 / +887 | djs（FYH FeaturePrefill） |
| B | 同上目录 `sglang-0.5.18-activation-prefill/activation-prefill.patch` | v0.5.18 | 5 文件 / 176 行 | djs（更早的 FTD 版） |
| C | 同上目录 `sglang-0.5.16-kimi-k3-ipi-capture/sglang-kimi-k3-ipi-capture.patch` | v0.5.16 | 3 文件 / 542 行 | djs（Kimi K3 / DSPARK） |
| D | `/mnt/kpfs_qyang/yutongz/CoolWatch/sglang-activation-runtime/` | 官方 wheel **0.5.19** | 18 文件 / 补丁 516 行 + 核心 424 行 | yutongz（zyt，CoolWatch） |

---

## A. djs FeaturePrefill（v0.5.18，17 文件）

**做法**：`extra_key` 字符串前缀携带 JSON spec（`fyh-feature:{...}`），在
`models/qwen2.py` 的层循环里手动 `index_select` 选行，`io_struct` 新增
`FeatureBatchOutput`，`hidden_bytes` 传原始字节。

改动分布：

| 文件 | 增/删 | 作用 |
|---|---|---|
| `models/qwen3.py` | +163/-12 | 请求级 spec 解析、早停、payload 组装 |
| `models/qwen2.py` | +59/-2 | 真正的层边界采集（Qwen3 继承它） |
| `managers/tokenizer_manager.py` | +101/-69 | 特征结果通道 |
| `managers/scheduler.py` | +103/-2 | 传 spec、走 feature 完成路径 |
| `managers/scheduler_components/output_streamer.py` | +105/-0 | 新建：特征流式回传 |
| `srt/ftd_activation.py` | +73/-0 | 新建：前缀解析 |
| `managers/schedule_batch.py` | +60/-3 | 缓存 tail 重算边界 |
| `runner/prefill_cuda_graph_runner.py` | +47/-2 | graph 固定行容量 + 实时索引 |
| `model_executor/model_runner.py` | +34/-1 | capture 开关 + graph 准入 |
| `managers/io_struct.py` | +24/-0 | `FeatureBatchOutput` |
| 其余 7 个文件 | 少量 | engine / forward_batch / policy / tp_worker / detokenizer 等 |

**值得抄的两点**：

1. 捕获时就地选行，不存整段：
   ```python
   if feature_only and i + 1 in self.layers_to_capture:
       selected = hidden_states + residual if residual is not None else hidden_states
       aux_hidden_states.append(selected.index_select(0, row_indices))
       if i + 1 == capture_stop_layer:
           break                        # 早停：后面 norm / lm_head / 采样全不做
   ```
2. 缓存边界不粗暴关 radix，而是压 `key_limit`：
   ```python
   if extra_key.startswith('fyh-feature:'):
       first = min(p if p >= 0 else input_len + p for p in spec['selected_positions'])
       key_limit = min(key_limit, max(0, first - CACHE_MARGIN + 1))   # margin 默认 512
   ```
   后续 `feature-cache-tail-margin-1.patch` 想把 512 收到 1，生产仍留 512 等对比验证。

**代价 / 技术债**（补丁 README 自陈）：
- 用 `extra_key` 塞 JSON，不是类型化请求对象；
- `qwen2.py` 残留 legacy `fyh-capture:` 落盘代码；
- `tokenizer_manager.py` 夹带与采集无关的配置改动；
- 全部命名 `fyh-*`；
- 为了拿到 tuple 形式的捕获结果，包装层得做畸形解包：
  ```python
  if isinstance(hidden_states, tuple) and len(hidden_states) == 2 and isinstance(hidden_states[1], (list, tuple)):
      hidden_states, aux_hidden_states = hidden_states
  elif isinstance(hidden_states, (list, tuple)):
      hidden_states, *aux_hidden_states = hidden_states
  ```
- 缺 GPU 回归：多位置、多层、chunked、radix 命中、graph vs eager、KV 释放。

---

## B. djs activation-prefill（v0.5.18，5 文件 176 行）

最轻的一版：`extra_key` 前缀 `ftd-activation:`，`eager_runner.execute` 里
按批开关 `set_capture_enabled`，复用既有 aux-hidden 通路，捕获请求强制 eager。

价值在于它证明了：**"复用 aux capture + 强制 eager" 能在 176 行内跑通**。
代价是层集合来自进程级环境变量 `FTD_ACTIVATION_LAYERS`，不是请求级——所以 A 才重写成请求级。

---

## C. djs Kimi K3 / DSPARK（v0.5.16，3 文件）

不是新通路，而是**配置已有通路**：Kimi K3 为 DSPARK 已经物化了
post-layer / pre-norm 残差流，这版只是配置它、按 prompt 行挑选、原子写 safetensors。
自带单测 `test_ipi_aware_capture.py`。

结论：如果模型本身已有等价的 aux 输出，采集可以是纯配置；但这对通用性没帮助，
因为多数模型没有。

---

## D. yutongz / zyt CoolWatch（0.5.19）——本文档的主要参照

**这是四套里架构最好的一个**，理由见下。它不是打补丁改模型，而是：

- 新增独立包 `src/sglang/srt/activation_capture/`：
  | 文件 | 行数 | 内容 |
  |---|---:|---|
  | `spec.py` | 85 | 请求校验、token 选择、前缀缓存限制（无 CUDA import，可跑在 HTTP 进程） |
  | `runtime.py` | 301 | 每轮采集计划、缓冲区、结果拆分、编码 |
  | `kernels.py` | 37 | prefill / decode 的 Triton 融合 gather |
- 对官方源码的改动集中在一个可审阅补丁（516 行），`upstream.json` 钉死 wheel SHA-256，
  `tools/bootstrap.py` 从哈希校验过的 wheel 重新提取源码再打补丁，`tools/export_patch.py` 反向导出。
- 验证留档：18 个原始文件 byte-identical 未改、8 个单测、HTTP 测试、
  真实 Qwen3-8B 对照 HF BF16 最低余弦 0.9995 / 最大相对 L2 误差 3.15%。

### D 的关键设计决定（逐条对我们有直接价值）

**D1. 请求参数是一等字段，不是字符串前缀 hack。**

```json
"activation_capture": {
  "targets": [{"layer": 9, "site": "block_output"},
              {"layer": 19, "site": "ffn_output"}],
  "tokens": {"prompt": "last", "generated": "none"},
  "encoding": "base64"
}
```
`site` 区分 `block_output`（加回残差后、下一层 norm 前）和 `ffn_output`（FFN 分支输出，
跨卡通信已完成、未加残差）。

**D2. 三层分离的采集运行时。**

```
ForwardBatch.activation_plan  →  Recorder.begin()  →  tap(layer,...)  →  Recorder.finish()
```
`tap()` 是模型层循环里唯一的注入点，且签名固定为 `tap(layer, hidden, residual, forward_batch)`。
`initialize(runner)` 时一次性做能力拒绝（PP / DP attention / sequence parallel /
context parallel / two-batch-overlap / speculative decoding / PD 分离 / MoE），
**启动即失败**而不是运行中出错。

**D3. Prefill：按层聚组，一个融合 kernel 完成选行 + 可选加残差 + 复制。**

```python
by_layer.setdefault(target["layer"], []).append((entry, index, target))
...
rows.extend(entry["row_start"] + p - entry["position_start"] for p in positions)
```
只取最后一 token 时不物化整个 prompt 的向量。行偏移计算把**不采集的请求也计入**：

> 某次 prefill A/B/C 各占 5/4/3 行，A、C 取末 token，则应读第 4、11 行，
> 不能因为 B 不采集就跳过它的行数。

**D4. Decode：用 GPU 选择表保持 CUDA Graph 可复用。**

这是四套里唯一真正解决 graph 动态层选择的做法。表形状：

```
[模型层数, 2 种 site, batch 容量]     # -1 表示本行不采
```
每次 forward 前只改表项内容，不改 Python 分支，所以**换层不需要重捕 graph**，
并发请求选不同层也各自成立。未选中层只读表项、不读向量；但检查 kernel 本身有固定开销，
所以普通 chat 实例关掉 `SGLANG_ACTIVATION_CAPTURE` 来免掉这部分成本。

**D5. 回传前先快照。**

decode 输出缓冲会被下一次 replay 复用，异步 D2H 若直接引用会被覆盖。
`Recorder.finish()` 先在 forward stream 上复制本轮有效结果，再走官方 pinned-memory
异步 D2H + `record_stream` + 完成事件。全程不在每层 `.cpu()` / `.tolist()`。

**D6. 结果走独立字段，因为 `sample()` 会清空 aux 输出。**

> `ModelRunner.sample()` 会清空原有的 `auxiliary_device_output`。如果直接把采集结果放进去，
> 到了采样阶段就会丢失。因此先用 `activation_device_output` 保存，在 D2H 拷贝阶段再与原有辅助输出合并。

这条是踩坑记录，非常有价值（我们的基线里 `model_runner.py:2086` 仍在做同样的清空）。

**D7. 缓存：`prefix_limit()` 精确到最早未采集 token，不用固定 margin。**

```python
wanted = selected_positions(spec, 0, len(req.origin_input_ids), len(req.origin_input_ids))
missing = [p for p in wanted if (0, p) not in seen]
if missing:
    return min(limit, missing[0]) if limit is not None else missing[0]
```
比 A 的固定 512 margin 更准：取最后一 token 时前面的缓存照常复用，取全 prompt 时自然退化为整段重算。

**D8. token 对齐语义写清楚了。**

`generated: processed` 返回的是**真正作为模型输入执行过 forward 的生成 token**，
不是"预测该 token 的前一个位置"。正常生成结束时最后一个输出 token 没有再次 forward，
所以不返回它；实现明确拒绝为补这一个状态多加一次 forward。

### D 也有明确的边界，不要照抄它的限制

- 只支持 CUDA、文本、普通自回归、chunked prefill、TP；其余启动拒绝；
- **MoE 明确不开放**：BF16 Qwen3.5 MoE 对照 9216 个元素里 54 个超差
  （最大绝对误差 0.0108，atol=0.003 / rtol=0.04），原因未明；
- 有采集行的 prefill batch 走 eager，无采集行的 batch 才走 prefill graph；
- 为拿到完整 FFN 输出，关闭了 MLP all-reduce 延迟融合（性能代价）；
- 采集内核在 decode graph 里有固定开销。

---

## 横向对比

| 维度 | A (djs FeaturePrefill) | B (djs FTD) | C (Kimi) | D (zyt CoolWatch) |
|---|---|---|---|---|
| 请求契约 | `extra_key` 前缀 + JSON | 进程级环境变量 | 配置文件 | **一等请求字段** |
| 层集合粒度 | 请求级 | 全局 | 全局 | **请求级** |
| 模型改动 | 改 `qwen2/qwen3` 计算 | 改 runner | 无 | **`tap()` 单点注入** |
| 通用/专用解耦 | 差 | 中 | 好但无通用性 | **好** |
| Prefill 选行 | `index_select` | 存整段 | safetensors | **融合 kernel** |
| Decode graph | 强制 eager | 强制 eager | 不涉及 | **GPU 选择表，可复用** |
| 缓存边界 | 固定 margin 512 | 不涉及 | 不涉及 | **精确 position** |
| 结果字段 | 新增 `FeatureBatchOutput` | `meta_info` | 落盘 | 新增 `activation_device_output` |
| 能力拒绝 | 无 | 无 | 无 | **启动 fail-fast 表** |
| 验证留档 | 无 | 有单测 | 有单测 | **完整（含 HF 数值对照）** |

**结论**：以 D 为蓝本，改三处——(1) 用请求字段 + 类型化 spec（D 已做，A/B 的问题）；
(2) 层内容采集挂到基线已有的 `layer_boundary` `capture` 契约上，模型改动降到 1 行/模型，
且**不需要手动加残差**（见设计文档 §3）；(3) 结果合并走基线已有的
`append_auxiliary_output` 协议，而不是自己写 D2H 逻辑。
