# InScope（观心）使用手册

> **项目总览、设计决策与兼容性说明见[根 README](../README.md)。**
> 本文件只讲**怎么用**：安装、API、请求契约、测试与当前进度。

基线 `sglang-inscope@main` = `e898054d`，上游 `yeasen-z/sglang-inscope`。

---

## 快速开始

### 1. 安装当前 fork

先准备一个 Python 3.11+ 环境并激活（脚本**不**创建环境，也**不**查找 conda），
然后以 editable 模式安装：

```bash
source /path/to/env/bin/activate        # uv / venv / conda 均可
bash inscope/scripts/setup_env.sh
```

也可以用 `PYTHON` 显式指定解释器，不激活：

```bash
PYTHON=/path/to/env/bin/python bash inscope/scripts/setup_env.sh
```

脚本做三件 `pip install -e python` 不会替你做对的事：

1. **先装 `torch==2.14.1` 再 `--no-build-isolation`**——`python/pyproject.toml`
   的 `build-system.requires` 里有 torch，隔离构建会再下一整份 torch；
2. **`SGLANG_BUILD_RUST_EXTS=none`**——`python/setup.py` 默认构建全部 Rust 扩展，
   那需要 cargo；
3. **装完自检** fork 的 `aux_hidden_states` / `final_norm` 是否 import 得动。

已有环境想手工装，等效命令就是上面两点：

```bash
pip install torch==2.14.1
SGLANG_BUILD_RUST_EXTS=none pip install --no-build-isolation -e ./python
```

aarch64（Jetson / Thor）上**必须先显式设置 `CUDA_HOME`**，脚本会检查它。

> **Thor 特有的坑**：`CUDA_HOME` 必须同时能解析到 `bin/nvcc` 和
> `include/curand_kernel.h`。这两个文件在 Thor 上常分属不同位置（一个 CUDA 安装
> + 一个 pip `nvidia-*` 包），缺任何一个都会在编译期或运行时 JIT 阶段以不同形式
> 失败。DeepGEMM on SM110 的额外要求见[根 README §3.5](../README.md)。
>
> **为什么不能靠自动发现**：上游文档称 `CUDA_HOME` 可选，会自动按
> `$CUDA_HOME` → `PATH` 上的 `nvcc` → `/usr/local/cuda` 三级回退。但 Thor 的
> CUDA 来自 pip `nvidia-*` 包，实测**三级全部落空**——不显式设置就直接失败，
> 没有"自动"可言。

```bash
export CUDA_HOME=/path/to/prefix   # 需同时含 bin/nvcc 与 include/curand_kernel.h
bash inscope/scripts/setup_env.sh
```

### 1b. 安装 DeepGEMM-SM110（仅 Jetson / Thor 需要）

SM110 上的 DeepGEMM FP8 加速**不在 PyPI 上**——PyPI 上的 `sgl-deep-gemm==0.2.3`
是 stock 版，不支持 SM110。两条路可选：

**A. 直接下预编译 wheel（快，推荐先试）**

```bash
wget https://github.com/yeasen-z/DeepGEMM-thor-sm110/releases/download/v0.2.3-sm110/sgl_deep_gemm-0.2.3%2Bsm110-py3-none-linux_aarch64.whl
pip install --force-reinstall --no-deps sgl_deep_gemm-0.2.3+sm110-py3-none-linux_aarch64.whl
```

> 适用 **aarch64 + PyTorch 2.14.1 + CUDA 13.0**。`_C.so` 的 ABI 与这个三元组绑定，
> 其他组合请走 B。
>
> **不要改文件名**——wheel 文件名本身就是元数据，改名后 pip 会报
> `Invalid wheel filename (wrong number of parts)`。

**B. 从源码构建（通用，约 2 分钟 + 150 MB 子模块）**

`_C.so` 只对一个 (arch, torch ABI, CUDA major) 三元组有效，所以换环境就该重编。
这个脚本把 clone → 构建 → 安装 → 校验串起来：

```bash
source /path/to/env/bin/activate
bash inscope/scripts/setup_deepgemm.sh

# 或指定已有 checkout / 强制重建
DEEPGEMM_SRC=/path/to/fork bash inscope/scripts/setup_deepgemm.sh
FORCE=1 bash inscope/scripts/setup_deepgemm.sh
```

它会下载约 150 MB 的 CUTLASS 子模块（最慢的一步），构建约 2 分钟。
**幂等**：已装好 SM110 支持的 DeepGEMM 就直接退出，不会重复构建。

校验分两级：先问 `_C.so` 自己支持哪些 arch major（最可靠），再做一次真实
`bf16_gemm_nt` 冒烟测试。装错了会以非零退出码和可照做的提示结束。

> **不装也能跑**：SGLang 会探测到 DeepGEMM 不支持 SM110，回退到 Triton 并打
> warning，不会崩、也不会静默算错——只是慢一些（实测 prefill 差 1.47x）。

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
| `layers` | 必填，默认单请求最多 8 层；`-1` 是最后 block 的 pre-final-norm 输出 |
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

---

## 三种执行模式

| 模式 | 请求形态 | 产出 |
|---|---|---|
| `generate` | 普通请求 | 文本（不创建 feature plan/缓冲或 D2H 传输） |
| `prefill_only` | `max_new_tokens=0` + `feature_capture` | 仅特征，跳过 decode/采样/logits |
| `generate+features` | 普通请求 + `feature_capture` | 文本 + 特征 |

---

## 测试

CPU 契约和运行时回归：

```bash
SGLANG_CACHE_DIR=/tmp/sglang-inscope-cache \
  python -m pytest -q \
  test/registered/unit/feature_capture \
  test/registered/unit/managers/test_output_streamer_customized_info.py
```

真实 GPU/HF 数值对照和 P3–P5 验收参见
[`test/manual/feature_capture/README.md`](../test/manual/feature_capture/README.md)。

---

## 设计文档索引

| 路径 | 内容 |
|---|---|
| [`design/implementation-design.md`](design/implementation-design.md) | **实现设计 v2**：15 条设计决策及被否方案、层号语义、请求契约、行映射、缓存边界、文件清单、阶段验收 |
| [`design/porting-map.md`](design/porting-map.md) | **核心实现移植对照**：前作每项机制在本基线对应什么、能搬什么、能删什么 |
| [`design/dev-workflow.md`](design/dev-workflow.md) | **开发流程**：fork 分支为真源、patch 只作发布产物、提交纪律与上游跟进成本 |
| [`design/prior-art.md`](design/prior-art.md) | 前作调研：本机四套已有的 SGLang 隐藏层采集实现及其代价 |
| [`handoff.md`](handoff.md) | 交接文档：仓库/环境/架构决策/已完成文件 |
| [`suggestion_and_require.md`](suggestion_and_require.md) | 对 `DeepGEMM-SM110` 的集成需求与改进建议 |
| `../test/manual/feature_capture/` | 测试设计、样例 schema、后续探测实验的落点 |

---

## 当前状态

- [x] 前作调研（四套实现，全部在本机，不在本仓）
- [x] 实现设计 v2 定稿
- [x] 测试设计与三份 JSON schema（`test/manual/feature_capture/`）
- [x] **评审决策 R1/R2/R3**（见设计 §15）：交付面只做 Engine、P5 只做 9B、probe 在外部仓
- [x] **P0 环境**：Conda `inscope`，`torch==2.14.1+cu130`、`flashinfer_python[cu13]==0.7.2`、`sglang-kernel==0.4.10`，当前源码以 editable 模式安装
- [x] P1 契约与校验骨架：`spec.py`、`plan.py`、`capabilities.py`、`bridge.py` 已落盘
- [x] P2 请求透传：`GenerateReqInput → TokenizedGenerateReqInput → Req → ForwardBatch` 已打通并建立 extend slot plan
- [x] P2 CPU/设备通路骨架：recorder、Qwen3 layer callback、异步 auxiliary output、`meta_info["features"]` 已接通
- [x] P2 Engine API：同步/异步 Engine 与 HTTP adapter 均透传 `feature_capture`
- [x] P2 手工验收入口：Qwen3 explicit/suffix smoke、base64 解码、HF BF16 数值对照工具
- [x] CPU 回归：feature-capture 套件加 output-streamer 相关测试 87 passed（另有 35 个 subtests）
- [x] P2 GPU 验证：Qwen3-8B BF16 五层 HF 对照通过；Qwen3-8B-FP8 使用 Triton 后端完成真实 `prefill_only` 闭环
- [x] P3 多层 / 多位置 / 混合 batch / chunked / 缓存命中 / 并发 / 普通生成回归
- [x] P4 `generate+features`：生成 token/text 与普通生成一致，仍只采 prompt
- [x] P5 第二架构 `Qwen3.5-9B`（Dense text tower）证明解耦：单/多位置与 generate+features 真实 GPU 通过
- [x] **Thor/SM110 DeepGEMM FP8**（分支 `feat/deepgemm_sm110_thor`）：配合 `DeepGEMM-SM110` 的 `_C.so` 走通，prefill 实测比 Triton 快 1.47x（短请求 2.21x）。详见[根 README §3.5](../README.md)
- [ ] P6 SafeGauge 式探测实验（外部仓）
- [ ] P7（推迟）OpenAI 兼容层
