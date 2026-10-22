# InScope（观心）

**在 SGLang 上增加请求级、多层 residual-stream 特征采集的 fork。**

一次 prompt prefill 就能取回指定 token 位置、指定层的隐藏状态，用于 SafeGauge
式的表征分析与 probe 实验。普通请求的行为、性能与输出完全不变。

> 上游 SGLang 的原始 README 保留在 [`README.upstream.md`](README.upstream.md)。
> 本项目是 [sgl-project/sglang](https://github.com/sgl-project/sglang) 的 fork，
> 遵循 Apache-2.0。
>
> 快速上手：[`inscope/README.md`](inscope/README.md) ｜
> 交接入口：[`inscope/handoff.md`](inscope/handoff.md) ｜
> 详细设计：[`inscope/design/`](inscope/design/)

---

## 1. 实现了什么

在**既有生成接口**上增加一个一等字段 `feature_capture`。开启后，一次请求可以：

- **多层**：默认单请求最多 8 层，跨整个 residual stream；
- **多位置**：`last` / `explicit` / `suffix_last` / `all` 四种模式；
- **两种产出**：只做 prefill 只返回特征（`max_new_tokens=0`），或「文本 + 特征」同时返回；
- **零拷贝编码**：按模型 dtype 的原始字节 + base64 回传，避免转换失真与体积膨胀。

入口有三条，共用同一套校验与通路：

| 入口 | 说明 |
|---|---|
| `Engine.generate(..., feature_capture=...)` | 主要交付面 |
| 原生 HTTP `/generate` | 复用上面对话协议，`meta_info` 原样透出，无需额外代码 |
| `prefill_only` 模式 | `max_new_tokens=0`，跳过 decode / 采样 / logits |

**默认关闭。** `SGLANG_ENABLE_FEATURE_CAPTURE=0` 时不创建 feature plan、不分配
特征缓冲、不做任何 D2H 传输——关闭路径上不存在为采集准备的对象。

---

## 2. 设计

### 2.1 架构：三块拆分，模型侧只有一个函数名

```
feature_capture/  ← 通用运行时，模型无关（层号/位置/编码/传输全在这里）
      ↑
模型文件          ← 只声明 feature_capture_sites + 调一次 layer_capture(...)
      ↑
官方传输协议      ← DeviceAuxiliaryOutput → copy_to_host → consume（复用，不重写）
```

模型文件里**不认识**层号、token 位置、dtype、编码方式。加一个新模型 = 一行
`layer_capture(...)` 调用 + 一行类属性声明。这条约束是可验收的：以「模型中不出现
位置、编码、传输逻辑」为准，而不是以改动行数为准。

### 2.2 关键设计决策

完整决策表（含被否方案与理由）见
[`implementation-design.md` §0.2](inscope/design/implementation-design.md)。最影响
正确性的几条：

| 决策 | 选择 | 为什么不能选另一个 |
|---|---|---|
| **层号语义** | residual-stream block boundary，`L ∈ [0, N]`；`L < N` 与 HF `outputs.hidden_states[L]` 对齐 | 用 block 索引会让调用方手写 `+1`，且和 HF 语义错位 |
| **最后一层** | `L = N`（或 `-1`）走 `final_norm(capture=)`，表示最后 block 的 **pre-final-norm** 输出 | 当前 Transformers 的 Qwen3 `hidden_states[N]` **已经过 final norm**，直接对齐会算错；HF golden 必须挂最后 block hook |
| **采集点** | 挂基线已有的 `capture` 回调 | 手算 `hidden + residual` 会绕开融合 kernel 的 FP32 累加 |
| **结果回传** | 复用 `DeviceAuxiliaryOutput`，挂在 `LogitsProcessorOutput.feature_device_output` | 直接塞 `auxiliary_device_output` 会被 `sample()` 开头清成 `None` |
| **选行** | `torch.index_select` 到预分配输出 | 不需要 fused kernel：`capture` 给的 `value` 本就在显存，只产出小结果 |
| **缓存边界** | 精确到「最早未采集位置」+ 已采位置集合 | 固定 margin（如 512）会白算最多 512 token，且不处理重算导致的重复采集 |
| **结果编码** | 原始字节 + base64 | `.tolist()` 走 JSON 会让 4096 维 BF16 膨胀一个数量级并丢精度 |
| **进程开关** | 默认关闭 | 总是开启会让所有用户为用不到的功能买单 |

### 2.3 位置语义：`suffix_last` 是校验器，不是别名

`positions.mode` 支持四档：

- `last` —— 最后一个 prompt token
- `explicit` —— 显式位置列表，支持 Python 式负索引
- `suffix_last` —— 把 gauge 语句拼在输入末尾，锁定**它的最后一个 token**
- `all` —— 全部位置

`suffix_last` 若在分词后找不到 suffix，请求**报错**，不会静默退化成 `last`。
SafeGauge 类实验依赖「采的确实是 gauge 位」，静默退化会让整批 probe 数据错而不自知。

### 2.4 数值正确性

不靠「看起来对」：

- `L < N` 与 HF `outputs.hidden_states[L]` 对齐；`L = N` 用 block hook 作 HF 参考；
- 缓存命中重放有专门用例（重算不能重复采集）；
- FP8 与 BF16 的关系有专门用例，但**永不**断言两者逐位相等。

---

## 3. 兼容性

### 3.1 对上游

- 以 fork 分支为唯一真源，定期 `merge upstream main`，不靠 patch 文件流；
  发布补丁只作导出物。流程见 [`dev-workflow.md`](inscope/design/dev-workflow.md)。
- 不修改 `python/pyproject.toml` 的版本号或依赖，避免与上游持续冲突。
- 不碰基线的 `capture_aux_hidden_states` / `layers_to_capture`。用开关 + `finally`
  恢复的写法会在包装层与内层之间产生 tuple 解包不一致，并影响 EAGLE 采集。

### 3.2 对既有用户

| 场景 | 行为 |
|---|---|
| 不传 `feature_capture` | 与原生 SGLang 完全一致，无额外对象、无额外传输 |
| 传了 `feature_capture` | 该请求不走 CUDA graph；采集行以 eager 执行 |
| 未设 `SGLANG_ENABLE_FEATURE_CAPTURE` | 功能整体关闭，字段被拒绝而非静默忽略 |

### 3.3 v1 明确不支持（fail fast，不静默降级）

DP attention、CP、TBO、speculative decoding、disaggregation、multimodal、
split prefill、MoE 采集，以及 decode 阶段的采集。已验证配置为 CUDA / TP=1 / PP=1。

交付面只做 `Engine.generate()` 与原生 `/generate`；**不含** OpenAI 兼容层
（`/v1/chat/completions`、`/v1/completions`），原因见设计文档 §15。

### 3.4 已验证范围

- 模型：Qwen3-8B BF16、Qwen3-8B-FP8、Qwen3.5-9B（Dense text tower，验证解耦）
- Qwen3.5 的顶层 checkpoint 需用 `language_model_only=True` 加载；**v1 不接受视觉输入**
- 场景：多层 / 多位置 / 混合 batch / chunked prefill / radix cache 命中 / 并发 /
  `generate+features` 与普通生成一致性
- 平台：x86 + CUDA 已在前面 P0–P5 验证；**Jetson Thor（aarch64 / SM110）**
  的 DeepGEMM FP8 适配见下节

### 3.5 SM110（Jetson Thor）的 DeepGEMM 适配

DeepGEMM 原本在 SM110 上不可用，会被强制退回 Triton。本 fork 让它走通：

- **DeepGEMM 侧**：需要 `DeepGEMM-SM110` 构建出的 `_C.so`——把 host 侧
  `arch_major == 10` 的 dispatch 扩到 `10 or 11`，使 SM110 落到既有 SM100
  tcgen05 内核族。设备内核仍由 DeepJIT 运行时编译（`sm_110f`），无需改 `jit.hpp`。
- **本仓侧**：两处改动——SM110 加入 UE8M0 scale 判定（SM100 族 block-FP8 的 SF
  打包内核要求非负 2 的幂 scale，否则触发 CUDA 719）；以及不再对 SM110 无条件
  降级到 `triton`。

启用：`export SGLANG_ENABLE_JIT_DEEPGEMM=1`。

**Thor 上 `CUDA_HOME` 必须显式设置**（这一条和上游文档的说法不同）：上游称
`CUDA_HOME` 可选，会自动按 `$CUDA_HOME` → `PATH` 上的 `nvcc` → `/usr/local/cuda`
三级回退。但 Thor 的 CUDA 来自 pip `nvidia-*` 包，实测三级**全部落空**：

```text
1) CUDA_HOME/CUDA_PATH : 未设
2) PATH 上的 nvcc      : 无
3) /usr/local/cuda     : 不存在
```

所以 Thor 上要显式指向一个同时含 `bin/nvcc` 和 `include/curand_kernel.h` 的前缀，
否则构建期报 `Cannot locate a CUDA installation`，或运行期 JIT 找不到设备头文件。

**能力探测，不是版本硬编码**：启动时会问当前装的 DeepGEMM 支不支持 SM110——
优先用它的能力查询（`supported_arch_majors()` / `supports_arch()`，见
[需求文档 R4](inscope/suggestion_and_require.md)），老版本回退到版本号里的
`+sm110` 标记。**DeepGEMM 侧升级或改接口都不会让这里失效。**

`supported_arch_majors()` 是最可信的信号——它直接读已编译的 `_C.so`，因此能识别
「Python 侧是新的、`_C.so` 是旧的」这种错配；错配时该调用抛 `AttributeError`，
探测会吞掉它并回退到版本号，不会让 SGLang 在 import 期崩掉。

探测到不支持时（例如 `_C.so` 仍是原版），**回退 Triton 并打一条 warning**，
不会在第一个请求时崩掉，也不会静默算错。装对了才会启用 DeepGEMM。

实测（PrefillBench 真实数据，**仅 prefill、41 请求 / 1.44 M token、24/36 层**）：
5,331 tok/s vs Triton 3,639 tok/s，即 **1.47x**；短请求（~4k）**2.21x**。
截断层次意味着绝对值不代表全模型吞吐，但后端比值有效。

详细步骤见 [`inscope/README.md`](inscope/README.md)；
对 `DeepGEMM-SM110` 的改进诉求见
[`inscope/suggestion_and_require.md`](inscope/suggestion_and_require.md)。

---

## 4. 快速开始

```bash
git clone git@github.com:yeasen-z/sglang-inscope.git
cd sglang-inscope
source /path/to/env/bin/activate           # 任选 uv / venv / conda，脚本不建环境
bash inscope/scripts/setup_env.sh          # x86 与 aarch64 同一份脚本
```

```python
import os
os.environ["SGLANG_ENABLE_FEATURE_CAPTURE"] = "1"   # 必须在 import 前设置

import sglang as sgl
engine = sgl.Engine(model_path="/path/to/Qwen3-8B", tp_size=1, disable_cuda_graph=True)
result = engine.generate(
    prompt="Explain why the sky is blue.",
    sampling_params={"max_new_tokens": 0},          # 只 prefill，只返特征
    feature_capture={
        "layers": [0, 20, -1],
        "positions": {"mode": "last"},
        "encoding": "float_list",
    },
)
print(result["meta_info"]["features"])
engine.shutdown()
```

完整用法、请求契约字段表与 HTTP 示例见 [`inscope/README.md`](inscope/README.md)。

---

## 5. 代码落点

| 路径 | 内容 |
|---|---|
| `python/sglang/srt/feature_capture/` | **通用运行时**（模型无关）：`spec` 校验、`plan` 行映射、`bridge`、`recorder`、`encoding`、`capabilities` |
| `python/sglang/srt/models/*.py` | 薄适配层：只声明 `feature_capture_sites` + 调用 `layer_capture(...)` |
| `python/sglang/srt/layers/deep_gemm_wrapper/`、`quantization/fp8_utils.py` | SM110 / Thor 的 DeepGEMM 兼容 |
| `inscope/` | 本 fork 的设计、交接、环境脚本、测试契约 |
| `test/manual/feature_capture/` | 手动验收：HF 数值对照、P3–P5 场景 |

---

## 6. 两条不能妥协的约束

1. **模型适配与通用逻辑解耦**：`feature_capture/` 里不出现任何模型名；模型文件只
   声明 `feature_capture_sites`，并在已有 layer boundary 上调用
   `layer_capture(...)` / `final_capture(...)`；位置、编码和传输逻辑不得进入模型文件。
2. **数值与位置不能靠「看起来对」**：`L < N` 与 HF `outputs.hidden_states[L]` 对齐，
   `L = N` 是最后 block 的 pre-final-norm 输出，必须用 block hook 作 HF 参考；
   缓存命中重放、FP8 与 BF16 的关系都有专门用例；FP8 永不与 BF16 断言逐位相等。

---

## 7. 知识边界

探测 AUROC 高只说明表示里存在**可利用的信号**，不等于模型「理解」了该属性，
也不证明生成时真的使用了它。任何结论文档必须保留这条边界。
