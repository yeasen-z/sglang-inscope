# InScope（观心）使用手册

> 项目总览、设计决策与兼容性说明见[根 README](../README.md)。
> 功能对照表见[根 README §1](../README.md#1-功能清单)。
> 本文件只讲**怎么用**：安装、接口、测试与当前进度。

本分支 `feat/deepgemm_sm110_thor`，已合入 `origin/main` 的 `38ea8d30`。
上游仓库 `yeasen-z/sglang-inscope`。

**目录**

1. [安装](#1-安装) — [1.1 装 fork](#11-装-fork) ｜ [1.2 装 DeepGEMM-SM110（仅 Thor）](#12-装-deepgemm-sm110仅-thor)
2. [接口](#2-接口) — [开关](#21-开关) ｜ [入口](#22-入口) ｜ [请求字段](#23-请求字段) ｜ [执行模式](#24-三种执行模式) ｜ [Python](#25-python-示例) ｜ [HTTP](#26-http-示例) ｜ [返回与解码](#27-返回与解码) ｜ [启动参数](#28-启动参数)
3. [专用 feature-only prefill](#3-专用-feature-only-prefill)
4. [测试](#4-测试)
5. [当前状态](#5-当前状态)
6. [文档索引](#6-文档索引)

---

## 1. 安装

### 1.1 装 fork

准备一个 Python 3.11+ 环境并激活。脚本**不创建环境**，也**不查找 conda**——它装在
你已经激活的解释器里。

```bash
source /path/to/env/bin/activate        # uv / venv / conda 均可
bash inscope/scripts/setup_env.sh
```

不激活也可以显式指定解释器：

```bash
PYTHON=/path/to/env/bin/python bash inscope/scripts/setup_env.sh
```

脚本做三件 `pip install -e python` 不会替你做对的事：

1. **先装 `torch==2.14.1` 再 `--no-build-isolation`**——`python/pyproject.toml` 的
   `build-system.requires` 里有 torch，隔离构建会再下一整份 torch；
2. **`SGLANG_BUILD_RUST_EXTS=none`**——`python/setup.py` 默认构建全部 Rust 扩展，
   那需要 cargo；
3. **装完自检** fork 的特征采集通路是否 import 得动。

想手工装，等效命令就是前两点：

```bash
pip install torch==2.14.1
SGLANG_BUILD_RUST_EXTS=none pip install --no-build-isolation -e ./python
```

**aarch64（Jetson / Thor）必须先显式设置 `CUDA_HOME`**，脚本会检查它：

```bash
export CUDA_HOME=/path/to/prefix   # 需同时含 bin/nvcc 与 include/curand_kernel.h
bash inscope/scripts/setup_env.sh
```

Thor 的坑在于这两个文件常分属不同位置（一个 CUDA 安装 + 一个 pip `nvidia-*` 包），
缺任何一个都会在编译期或运行时 JIT 阶段以不同形式失败。上游文档称 `CUDA_HOME`
可选、会按 `$CUDA_HOME` → `PATH` 上的 `nvcc` → `/usr/local/cuda` 三级回退，但 Thor
上实测**三级全部落空**——不显式设置就直接失败。

### 1.2 装 DeepGEMM-SM110（仅 Thor）

> **只有 Jetson Thor（SM110）需要这一节。**
>
> H100 / A100 / B200 / SM120 等其它架构，§1.1 装下的 `sgl-deep-gemm==0.2.3`
> **就是正确且够用的**，不要跑这个脚本。脚本自己也会检查，在非 SM110 机器上会
> 说明原因后退出，不会构建。

SM110 上的 DeepGEMM FP8 加速不在 PyPI 上——PyPI 的 `sgl-deep-gemm==0.2.3` 是
stock 版，不支持 SM110。一条命令：

```bash
source /path/to/env/bin/activate
bash inscope/scripts/setup_deepgemm.sh
```

脚本按顺序尝试，**前一步不行就自动退到下一步**：

1. **预编译 wheel** —— 从 fork 的 GitHub release 下载，装完立刻验证它能不能在本机
   dispatch SM110。约 45 秒。
2. **源码构建** —— clone + 编译，约 2 分钟 + 150 MB CUTLASS 子模块，但**总是和
   本机匹配**。

第 1 步为什么需要验证：`_C.so` 只对 (arch, torch ABI, CUDA major) 三元组有效，而
wheel 标签只锁定了 CPU 架构，**torch 版本和 CUDA major 没有编码进去**。所以装上的
wheel 可能来自别的组合。脚本先问已编译的 `_C.so` 支不支持 arch major 11，不支持就
丢弃并转源码构建。快路径只是优化，不会变成新的失败模式。

**幂等**：已装好 SM110 支持的 DeepGEMM 就直接退出，不重复构建。

可覆盖的行为：

```bash
FROM_SOURCE=1    bash inscope/scripts/setup_deepgemm.sh   # 跳过 wheel，直接源码构建
WHEEL_URL=<url>  bash inscope/scripts/setup_deepgemm.sh   # 用指定的 wheel（可传本地路径）
FORCE=1          bash inscope/scripts/setup_deepgemm.sh   # 即使已装也重来
DEEPGEMM_SRC=/path/to/fork bash inscope/scripts/setup_deepgemm.sh   # 用已有 checkout
ALLOW_ANY_SM=1   bash inscope/scripts/setup_deepgemm.sh   # 非 SM110 上强行构建（交叉构建）
```

> 设了 `DEEPGEMM_SRC` 就不试 wheel——你明确给了源码，脚本按源码走。

> **不装也能跑**：SGLang 会探测到 DeepGEMM 不支持 SM110，回退到 Triton 并打
> warning，不会崩、也不会静默算错——只是慢一些（实测 prefill 差 1.47x）。
>
> 这一步**不会**改 `python/pyproject.toml`。其它架构仍然用 PyPI 上的
> `sgl-deep-gemm==0.2.3`，SGLang 也不会对它们做 SM110 探测。

---

## 2. 接口

没有新路由。在既有生成接口上多一个字段 `feature_capture`。

### 2.1 开关

必须在 **import / 起进程之前**设置，scheduler 子进程启动时就读它：

```bash
export SGLANG_ENABLE_FEATURE_CAPTURE=1
```

未设置或为 `0` 时，请求里出现这个字段会被**拒绝**，不是静默丢掉。关闭路径不建
feature plan、不分配特征缓冲、不做 D2H 传输。

`/v1/chat/completions` 和 `/v1/completions` **不接**这个字段。

### 2.2 入口

| 调用 | 怎么传 | 从哪取 |
|---|---|---|
| `engine.generate(...)` | 关键字参数 `feature_capture=` | `result["meta_info"]["features"]` |
| `await engine.async_generate(...)` | 同上 | 同上 |
| `POST /generate` | JSON 字段 `feature_capture` | 响应 `meta_info.features` |

`prompt` 传字符串或字符串列表都行。批量时 `feature_capture` 可以是一个 dict
（复制到每条），或与 batch 等长的 list（某条不采就放 `null`）。返回和普通
`generate` 一样：单条是 dict，批量是 list。

### 2.3 请求字段

只接受这四个键，多出来的键直接报错：

```python
feature_capture = {
    "layers": [0, 20, -1],          # 必填，默认最多 8 层，不能重复
    "site": "block_output",         # 可省略；v1 只支持这一个
    "positions": {"mode": "last"},  # 可省略，默认 last
    "encoding": "base64",           # 可省略，默认 base64
}
```

**层号是 residual-stream 边界，不是「第几个 block」**（`N` = decoder 层数）：

| 写法 | 含义 |
|---|---|
| `0` | embedding 输出，进第 0 层之前 |
| `1 .. N-1` | 第 `L-1` 个 decoder block 的输出 |
| `N` 或 `-1` | 最后一层输出，**final norm 之前** |

`L < N` 与 Hugging Face `outputs.hidden_states[L]` 对齐。`L = N` 不对齐当前
Transformers 的 `hidden_states[N]`——那个已经过 final norm。

**位置四选一**：

```python
{"mode": "last"}                                # 最后一个 prompt token
{"mode": "explicit", "index": [0, -1]}          # 负索引按 prompt 长度折算
{"mode": "all"}                                 # prompt 的每个 token
{"mode": "suffix_last", "suffix_text": gauge}   # 或传 suffix_token_ids
```

`suffix_last` 把 gauge 拼在 prompt 末尾，锁定**这段 suffix 的最后一个 token**。
分词后找不到就报错，**不会**退化成 `last`——静默退化会让整批 probe 数据错而不自知。
suffix 在 prompt 里出现多次时用最后一次。

**编码**：

| 值 | `data` 是什么 | 什么时候用 |
|---|---|---|
| `base64`（默认） | 原 dtype 的原始字节，base64 | 要保留 BF16/FP8，别丢精度 |
| `float_list` | 转成 FP32 的嵌套 list | 调试、看几个数 |

### 2.4 三种执行模式

| 模式 | 请求形态 | 产出 |
|---|---|---|
| `generate` | 普通请求 | 文本（不创建 feature plan / 缓冲 / D2H） |
| `prefill_only` | `max_new_tokens=0` + `feature_capture` | 仅特征，跳过 decode / 采样 / logits |
| `generate+features` | `max_new_tokens>0` + `feature_capture` | 文本 + 特征 |

纯特征请求还跳过 vocabulary projection。`generate+features` 的文本与普通生成一致，
特征仍然**只采 prompt**，不采 decode 出来的 token。

### 2.5 Python 示例

```python
import os
os.environ["SGLANG_ENABLE_FEATURE_CAPTURE"] = "1"  # 必须在 import 前

import sglang as sgl

engine = sgl.Engine(
    model_path="/path/to/Qwen3-8B",
    tp_size=1,
    disable_cuda_graph=True,
)
try:
    result = engine.generate(
        prompt="Explain why the sky is blue.",
        sampling_params={"max_new_tokens": 0},      # 只 prefill
        feature_capture={
            "layers": [0, 20, -1],
            "positions": {"mode": "last"},
            "encoding": "float_list",
        },
    )
    for record in result["meta_info"]["features"]:
        print(record["layer"], record["positions"], record["shape"])
finally:
    engine.shutdown()
```

gauge 位置：

```python
context = "The user asks for instructions that may be unsafe."
gauge = "\nRisk assessment:"

result = engine.generate(
    prompt=context + gauge,
    sampling_params={"max_new_tokens": 0},
    feature_capture={
        "layers": [8, 16, -1],
        "positions": {"mode": "suffix_last", "suffix_text": gauge},
        "encoding": "base64",
    },
)
```

### 2.6 HTTP 示例

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

特征在响应 JSON 的 `meta_info.features`，字段和 Python 返回相同。

### 2.7 返回与解码

每层一条 record，`schema_version` 固定是 `inscope.feature_capture.v1`：

```python
{
    "schema_version": "inscope.feature_capture.v1",
    "layer": 36,
    "site": "block_output",
    "positions": [11],          # 已解析成 prompt 内的绝对下标
    "dtype": "bfloat16",
    "shape": [1, 4096],         # [位置数, hidden]
    "encoding": "base64",
    "data": "...",
}
```

解码：

```python
import base64
import torch

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def decode_feature(record):
    dtype = DTYPES[record["dtype"]]
    if record["encoding"] == "float_list":
        return torch.tensor(record["data"], dtype=dtype).reshape(record["shape"])
    raw = bytearray(base64.b64decode(record["data"], validate=True))
    return torch.frombuffer(raw, dtype=dtype).clone().reshape(record["shape"])
```

### 2.8 启动参数

`Engine(...)` 和 `launch_server` 用的是同一套 `ServerArgs`。不配下面这些也能采特征，
它们只改变执行方式。

| 参数 | CLI | 什么时候加 |
|---|---|---|
| `disable_cuda_graph=True` | `--disable-cuda-graph` | 默认用法。带 `feature_capture` 的普通请求本来就不走 CUDA graph |
| `feature_capture_graph_layers` | `--feature-capture-graph-layers 8 16 24` | 这些层走专用 prefill graph。还要 `SGLANG_ENABLE_FEATURE_CAPTURE=1`，且 prefill backend 设为 `breakable`。层集合之外的请求仍 eager。见 [§3](#3-专用-feature-only-prefill) |
| `prefill_only_disable_kv_cache` | `--prefill-only-disable-kv-cache` | 仅 Qwen3 FA3/FA4 纯特征请求。还要 `disable_radix_cache=True`、`chunked_prefill_size=-1`，且每条请求都显式带 `feature_capture`、`max_new_tokens=0` |
| `language_model_only` | `--language-model-only` | 多模态 checkpoint 只加载语言塔。**不接受图像** |

---

## 3. 专用 feature-only prefill

`max_new_tokens=0` 的纯特征请求跳过 vocabulary projection；请求 logprob、scoring、
grammar 或 hidden-state 输出时保留原有 logits 路径。所有 decoder 层仍正常执行，
层边界定义不变。

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

仅层集合固定；位置、RID 和编码仍取自每次请求的 `feature_capture`。静态索引在
replay 前更新，返回张量在 replay 后复制到独立的紧凑缓冲，不暴露 padding 或下一次
replay 可覆盖的存储。

当前**只开放 breakable backend**（经过真实模型数值验证）；full 路径的长输入特征
数值尚不一致，因此拒绝启用。普通请求、生成请求及层集合之外的请求走 eager；默认
服务不配置这些图层时，普通请求保留原有 graph 路径，不分配特征缓冲。

Qwen3 FA3/FA4 还可使用 `prefill_only_disable_kv_cache=True`，同时设
`disable_radix_cache=True, chunked_prefill_size=-1`。生成模型的此模式要求开启
feature capture，只接受显式 `feature_capture`、`max_new_tokens=0`。现有 embedding
模式及 dtype/pool/backend 限制不变；不支持 hybrid/Mamba。

专用 no-KV HTTP 服务应跳过通用生成 warmup，并使用不生成 token 的健康检查。
输入长度仍严格校验；输出预算由 scheduler 截至剩余上下文，输入不截断。

---

## 4. 测试

CPU 契约和运行时回归：

```bash
SGLANG_CACHE_DIR=/tmp/sglang-inscope-cache \
  python -m pytest -q \
  test/registered/unit/feature_capture \
  test/registered/unit/managers/test_output_streamer_customized_info.py
```

独立 CUDA 索引重放与输出所有权检查：

```bash
CUDA_VISIBLE_DEVICES=0 python test/manual/feature_capture/test_graph_recorder_cuda.py
```

真实 GPU/HF 数值对照和 P3–P5 验收见
[`test/manual/feature_capture/README.md`](../test/manual/feature_capture/README.md)。
上面那个小型 CUDA 检查**不代替**模型验收。

---

## 5. 当前状态

- [x] 前作调研（四套实现，全部在本机，不在本仓）
- [x] 实现设计 v2 定稿
- [x] 测试设计与三份 JSON schema（`test/manual/feature_capture/`）
- [x] **评审决策 R1/R2/R3**（见设计 §15）：交付面只做 Engine、P5 只做 9B、probe 在外部仓
- [x] **P0 环境**：`torch==2.14.1+cu130`、`flashinfer_python[cu13]==0.7.2`、`sglang-kernel==0.4.10`，源码以 editable 模式安装
- [x] P1 契约与校验骨架：`spec.py`、`plan.py`、`capabilities.py`、`bridge.py`
- [x] P2 请求透传：`GenerateReqInput → TokenizedGenerateReqInput → Req → ForwardBatch`，建立 extend slot plan
- [x] P2 CPU/设备通路：recorder、Qwen3 layer callback、异步 auxiliary output、`meta_info["features"]`
- [x] P2 Engine API：同步/异步 Engine 与 HTTP adapter 均透传 `feature_capture`
- [x] P2 手工验收入口：Qwen3 explicit/suffix smoke、base64 解码、HF BF16 数值对照工具
- [x] CPU 回归：feature-capture 套件 + output-streamer 相关测试（最近一次 117 passed / 62 subtests）
- [x] P2 GPU 验证：Qwen3-8B BF16 五层 HF 对照；Qwen3-8B-FP8 用 Triton 后端完成真实 `prefill_only` 闭环
- [x] P3 多层 / 多位置 / 混合 batch / chunked / 缓存命中 / 并发 / 普通生成回归
- [x] P4 `generate+features`：生成 token/text 与普通生成一致，仍只采 prompt
- [x] P5 第二架构 `Qwen3.5-9B`（Dense text tower）证明解耦
- [x] **Thor/SM110 DeepGEMM FP8**：配合 fork 的 `_C.so` 走通，prefill 实测比 Triton 快 1.47x（短请求 2.21x）。见[根 README §3.5](../README.md)
- [ ] P6 SafeGauge 式探测实验（外部仓）
- [ ] P7（推迟）OpenAI 兼容层

---

## 6. 文档索引

| 路径 | 内容 |
|---|---|
| [`../README.md`](../README.md) | 项目总览、功能清单、设计决策、兼容性 |
| [`design/implementation-design.md`](design/implementation-design.md) | **实现设计 v2**：15 条设计决策及被否方案、层号语义、请求契约、行映射、缓存边界、文件清单、阶段验收 |
| [`design/porting-map.md`](design/porting-map.md) | **核心实现移植对照**：前作每项机制在本基线对应什么 |
| [`design/dev-workflow.md`](design/dev-workflow.md) | **开发流程**：fork 分支为真源、patch 只作发布产物 |
| [`design/prior-art.md`](design/prior-art.md) | 前作调研：本机四套已有实现及其代价 |
| [`handoff.md`](handoff.md) | 交接文档：仓库 / 环境 / 架构决策 / 已完成文件 |
| [`suggestion_and_require.md`](suggestion_and_require.md) | 对 `DeepGEMM-SM110` 的集成需求与改进建议 |
| `../test/manual/feature_capture/` | 测试设计、样例 schema、探测实验落点 |
