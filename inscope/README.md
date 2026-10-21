# InScope（观心）

InScope 是一个 SGLang fork：在正常生成接口上增加请求级 `feature_capture`，
可在一次 prompt prefill 中提取指定 token 位置、指定多层的 residual-stream
block output。主要用于 SafeGauge 式的表征分析和 probe 实验。

普通请求不会创建 feature plan、特征缓冲或 D2H 传输；该功能默认关闭，
必须通过环境变量显式开启。

基线 `sglang-inscope@main` = `92580581da`，上游 `yeasen-z/sglang-inscope`。

## 快速开始

### 1. 安装当前 fork

推荐以 editable 模式安装，修改源码后无需重新打包：

```bash
cd /path/to/sglang-inscope
bash inscope/scripts/setup_env.sh
```

脚本默认使用 `/opt/miniconda3/envs/inscope`。已有环境也可以直接安装：

```bash
SGLANG_BUILD_RUST_EXTS=none \
  python -m pip install --no-build-isolation -e ./python
```

### 2. 提取最后一个 prompt token 的多层特征

`SGLANG_ENABLE_FEATURE_CAPTURE` 要在 import/Engine 启动前设置，因为 scheduler
子进程会在启动时读取它。

```python
import os

os.environ["SGLANG_ENABLE_FEATURE_CAPTURE"] = "1"

import sglang as sgl

engine = sgl.Engine(
    model_path="/path/to/Qwen3-8B",
    tp_size=1,
    disable_cuda_graph=True,
)

try:
    result = engine.generate(
        prompt="Explain why the sky is blue.",
        sampling_params={"max_new_tokens": 0},
        feature_capture={
            "layers": [0, 20, -1],
            "positions": {"mode": "last"},
            "encoding": "float_list",
        },
    )
    records = result["meta_info"]["features"]
    for record in records:
        print(record["layer"], record["positions"], record["shape"])
finally:
    engine.shutdown()
```

`max_new_tokens=0` 表示只做 prefill 并返回特征。若设为正整数，同一个请求会
同时返回正常生成结果和 prompt 特征。

### 3. input + gauge suffix

把 gauge 语句拼到输入末尾，再用 `suffix_last` 锁定它的最后一个 token。
如果分词后找不到 suffix，请求会报错，不会静默退化到 prompt 末尾。

```python
context = "The user asks for instructions that may be unsafe."
gauge = "\nRisk assessment:"

result = engine.generate(
    prompt=context + gauge,
    sampling_params={"max_new_tokens": 0},
    feature_capture={
        "layers": [8, 16, -1],
        "positions": {
            "mode": "suffix_last",
            "suffix_text": gauge,
        },
        "encoding": "base64",
    },
)
```

### 4. 原生 HTTP `/generate`

```bash
SGLANG_ENABLE_FEATURE_CAPTURE=1 \
python -m sglang.launch_server \
  --model-path /path/to/Qwen3-8B \
  --tp-size 1 \
  --disable-cuda-graph

curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Explain why the sky is blue.",
    "sampling_params": {"max_new_tokens": 0},
    "feature_capture": {
      "layers": [0, 20, -1],
      "positions": {"mode": "last"},
      "encoding": "float_list"
    }
  }'
```

### 5. 请求契约

| 字段 | 含义 |
|---|---|
| `layers` | 必填，最多 8 层；`-1` 是最后 block 的 pre-final-norm 输出 |
| `site` | 可选，v1 仅支持 `block_output` |
| `positions.mode` | `last`、`explicit`、`suffix_last` 或 `all` |
| `positions.index` | `explicit` 模式的 token 位置列表，支持 Python 式负索引 |
| `positions.suffix_text` | `suffix_last` 模式由 SGLang tokenizer 编码的文本 |
| `positions.suffix_token_ids` | 调用方已分词时可直接传 token ids |
| `encoding` | `base64`（默认，保留原 dtype 字节）或 `float_list`（方便调试） |

结果位于 `result["meta_info"]["features"]`，每层一条 record，包含
`schema_version` / `layer` / `site` / `positions` / `dtype` / `shape` /
`encoding` / `data`。默认 base64 的解码参考
[`test_qwen3_hidden_layer.py`](../test/manual/feature_capture/test_qwen3_hidden_layer.py)。

## 已验证范围与限制

- 已验证：Qwen3-8B BF16、Qwen3-8B-FP8、Qwen3.5-9B Dense text tower。
- Qwen3.5 顶层 checkpoint 需用 `language_model_only=True`；v1 不接受视觉输入。
- 已验证配置为 CUDA、TP=1、PP=1；DP attention、CP、TBO、speculative decoding、
  disaggregation、multimodal 和 split prefill 会 fail fast。
- 默认 capture 请求走 eager；显式配置 `feature_capture_graph_layers` 可使用专用 prefill CUDA graph，见下节。chunked prefill 和 radix cache 保留原生请求语义。
- 当前交付面是 Python `Engine.generate()` 和原生 `/generate`；不包含 OpenAI
  `/v1/chat/completions` 或 `/v1/completions` 的 features 字段。

## 测试

CPU 契约和运行时回归：

```bash
SGLANG_CACHE_DIR=/tmp/sglang-inscope-cache \
  /opt/miniconda3/envs/inscope/bin/python -m pytest -q \
  test/registered/unit/feature_capture \
  test/registered/unit/managers/test_output_streamer_customized_info.py
```

真实 GPU/HF 数值对照和 P3–P5 验收参见
[`test/manual/feature_capture/README.md`](../test/manual/feature_capture/README.md)。

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

## 专用 feature-only prefill

`max_new_tokens=0` 的纯特征请求跳过 vocabulary projection；请求 logprob、
scoring、grammar 或 hidden-state 输出时保留原有 logits 路径。所有 decoder
层仍正常执行，层边界定义不变。

启用专用图时，在 `SGLANG_ENABLE_FEATURE_CAPTURE=1` 下创建 Engine：

```python
engine = sglang.Engine(
    model_path=model_path,
    feature_capture_graph_layers=[8, 16, 24],
    cuda_graph_config={
        "decode": {"backend": "disabled"},
        "prefill": {"backend": "breakable", "bs": [64, 128, 256], "max_bs": 256},
    },
)
```

仅层集合固定；位置、RID 和编码仍取自每次请求的原生 `feature_capture`。
静态索引在 replay 前更新，返回张量在 replay 后复制到独立的紧凑缓冲，
不暴露 padding 或下一次 replay 可覆盖的存储。当前仅开放经过真实模型数值验证的 breakable backend；full 路径的长输入
特征数值尚不一致，因此拒绝启用。
普通请求、生成请求及层集合之外的请求走 eager。默认服务不配置这些图层时，
普通请求保留原有 graph 路径，不分配特征缓冲。

Qwen3 FA3/FA4 还可使用已有 `prefill_only_disable_kv_cache=True`，同时设置
`disable_radix_cache=True, chunked_prefill_size=-1`。生成模型的此模式要求开启
原生 feature capture，只接受显式 `feature_capture`、`max_new_tokens=0`。
现有 embedding 模式及 dtype/pool/backend 限制不变；不支持 hybrid/Mamba。
专用 no-KV HTTP 服务应跳过通用生成 warmup，并使用不生成 token 的健康检查。

输入长度仍严格校验；输出预算由 scheduler 截至剩余上下文，输入不截断。

CPU 回归位于 `test/registered/unit/feature_capture/`；独立 CUDA 索引重放与
输出所有权检查：

```bash
CUDA_VISIBLE_DEVICES=0 python test/manual/feature_capture/test_graph_recorder_cuda.py
```

完整模型数值验证仍需运行手工验收。上述小型 CUDA 检查不代替模型验收。
