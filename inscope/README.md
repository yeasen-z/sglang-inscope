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

在 Jetson / Thor（aarch64）上改用 `setup_env_thor.sh`：那里没有 sudo 和
`/opt` 前缀，且 pip 在该网络下会卡住，所以用 `uv` 解析。所有机器相关路径都是
可覆盖的环境变量（conda 位置自动发现，仓库根从脚本位置推导），不含任何个人路径。

```bash
bash inscope/scripts/setup_env_thor.sh
```

该脚本还会提示一个 Thor 特有的坑：`CUDA_HOME` 必须同时能解析到
`bin/nvcc` 和 `include/curand_kernel.h`——这两个文件在本机常分属不同位置
（一个 CUDA 安装 + 一个 pip `nvidia-*` 包），缺任何一个都会在编译期或
运行时 JIT 阶段以不同形式失败。

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
- capture 请求不走 CUDA graph；chunked prefill 和 radix cache 已有真实 GPU 验证。
- 当前交付面是 Python `Engine.generate()` 和原生 `/generate`；不包含 OpenAI
  `/v1/chat/completions` 或 `/v1/completions` 的 features 字段。

## Thor / SM110 上的 DeepGEMM FP8

分支 `feat/deepgemm_sm110_thor` 让 SM110（NVIDIA Jetson Thor）能用上 DeepGEMM
的 block-FP8 内核，而不是被强制退回 Triton。**需要两半同时到位**：

1. **DeepGEMM 侧**：`DeepGEMM-SM110` 构建出的 `_C.so`——它把 host 侧 dispatch 的
   `arch_major == 10` 扩成 `10 or 11`，使 SM110 落到既有 SM100 tcgen05 内核族。
   设备内核仍由 DeepJIT 在运行时编译（`--gpu-architecture=sm_110f`），
   `jit.hpp` 无需改动。
2. **本仓两处改动**（commit `deepgemm: route SM110 (Thor) through the SM100 FP8 path`）：
   - `deep_gemm_wrapper/configurer.py`：`DEEPGEMM_SCALE_UE8M0` 的判定加入 SM110。
     SM100 族 block-FP8 的 SF 打包内核要求 scale 是非负 2 的幂
     （`deep_gemm/impls/smxx_layout.cuh` 里的 device assert），传寻常 FP32 scale 会
     触发 CUDA 719。SM110 与 SM100 共用该内核，所以必须同样请求 UE8M0 并做权重 requant。
   - `quantization/fp8_utils.py`：原来 `backend == "auto"` 时对 SM110 **无条件**
     降级到 `triton`，使 DeepGEMM 路径不可达。改为仅当 UE8M0 未启用时才降级。

启用：

```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=1
```

**失败模式是响亮的**：若只装本仓改动而 `_C.so` 仍是原版，会直接抛
`RuntimeError: Assertion error (csrc/apis/gemm.hpp:225): Unsupported architecture`，
不会静默算错。

性能（PrefillBench 真实数据，**仅 prefill 阶段、41 条请求 / 1.44 M token**）：
DeepGEMM **5,331 tok/s** vs Triton 3,639 tok/s，即 **1.47x**；
短请求（~4k）2.21x。注意这是截断到 24/36 层的 prefill 测量，绝对值不代表全模型吞吐，
但后端之间的比值有效。

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
- [x] **Thor/SM110 DeepGEMM FP8**（分支 `feat/deepgemm_sm110_thor`）：配合 `DeepGEMM-SM110` 的 `_C.so` 走通，prefill 实测比 Triton 快 1.47x（短请求 2.21x）
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
