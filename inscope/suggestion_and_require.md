# DeepGEMM-SM110 — 集成需求与建议

> 面向：负责 `yeasen-z/DeepGEMM-SM110` 的实现方。
> 来源：`sglang-inscope` 侧把 SM110（Jetson Thor）接入 DeepGEMM 时的实测结论。
> 约束：本文只提出需求，不改动 DeepGEMM-SM110 仓库本身。
> 实测环境：Jetson AGX Thor（aarch64, sm_110, 20 SM）、CUDA 13.0、torch 2.14.1+cu130、
> Python 3.12、无 sudo。

---

## 0. 现状摘要

DeepGEMM-SM110 的核心补丁是有效的，以下已在真机验证通过：

| 验证项 | 结果 |
|---|---|
| `build_sgl_deep_gemm.sh` 建 wheel | ✅ ~2 分钟，产出 `sgl_deep_gemm-0.2.3+sm110-py3-none-any.whl` |
| wheel 内 JIT 头文件完整性 | ✅ cutlass 690 / cute 112 / deep_jit 28 / deep_gemm 73 项 |
| 安装后版本号 | ✅ `0.2.3+sm110`（`dist-info` 与 `deep_gemm.__version__` 一致） |
| SM110 真实 dispatch | ✅ `bf16_gemm_nt` 在 (11,0) 上跑通 |
| JIT 编译架构 | ✅ `--gpu-architecture=sm_110f` |
| 内核家族 | ✅ 落到 `sm100_*` 即 SM100 tcgen05 族 |

**结论：补丁本身可用。本文全部内容都是围绕「让别人能自动装、装对、装错能看出来」的改进建议。**

---

## 1. 需求 P0 — 让自动安装成为可能

### R1. 需要可脚本化的一键构建入口

**现状**：✅ **已实现**（2026-10-22 复核）。`build_sgl_deep_gemm.sh` 现在自行检查
`CUDA_HOME` 是否是目录、`<CUDA_HOME>/bin/nvcc` 是否可执行、`nvcc --list-gpu-arch`
是否含 `compute_11x`、`PYTHON_EXE` 能否 import torch，且每条失败路径都给出可照做的
提示并以非零退出。**本节可以关闭。**

**原需求**（留档）：脚本自身应检查前置条件并以**可读的报错**退出，
而不是让 `nvcc` 报奇怪的错。至少覆盖：

- `CUDA_HOME` 是否设置，且 `<CUDA_HOME>/bin/nvcc` 可执行 ✅
- `nvcc` 是否能编译 `sm_110f`（`nvcc --list-gpu-arch` 含 `compute_110`）✅
- `git` 可用（子模块需要）✅（`git submodule update` 失败即报错）
- `torch` 可导入（已有）✅

### R2. 设备头文件的缺失应在构建期暴露

**现状**：✅ **已实现**。脚本探测 `<CUDA_HOME>/include/curand_kernel.h`，缺失时打印
警告并给出「拼一个自洽的 CUDA_HOME」或「加进 `CPATH`」两条出路，还会在
`headers :` 一行回显 present/MISSING。**本节可以关闭。**

**背景**：Thor 上 `nvcc` 与设备头文件常分属两处。若 `CUDA_HOME` 只指向有 `nvcc`
的那处，构建能过但运行期 JIT 会以另一种方式失败。

### R3. 构建产物应能直接用于 `pip install`

**现状**：✅ 已经满足——wheel 是自包含的（`include/` 全量打入）。

**需求**：在 README 显著位置写明**唯一推荐安装命令**，并强调 `--no-deps`：

```bash
pip install --no-deps --force-reinstall dist/sgl_deep_gemm-0.2.3+sm110-py3-none-any.whl
```

> 关于 `--force-reinstall`：`0.2.3+sm110` 与 SGLang 的 pin `sgl-deep-gemm==0.2.3`
> 是兼容的（PEP 440 的 `==` 忽略 local 段），所以正常 `pip install` 不会被降级。
> 但若目标环境里已存在**同版本**的 `0.2.3+sm110`，pip 会判定「已安装」而跳过；
> 重建后想覆盖就必须 `--force-reinstall`。README 应把这一点说清，避免"以为装了其实没装"。

---

## 2. 需求 P1 — 让"装错了"可被发现

### R4. 需要一个可被外部探测的能力标记（**最重要**）

**动机**：上层（SGLang）需要在**不实际调用内核**的前提下判断「当前装的 DeepGEMM 是否支持 SM110」。
目前唯一的信号是版本号里的 `+sm110` local 段，它有效但语义脆弱（见 R5）。

**需求**：在 `deep_gemm` 顶层暴露一个明确的、语义稳定的能力查询，例如：

```python
import deep_gemm
deep_gemm.supports_arch("sm110")   # -> True / False
# 或
deep_gemm.SUPPORTED_ARCH_MAJORS   # -> frozenset({9, 10, 11, ...})
```

**理由**：版本字符串是"人读"的，能力集合是"程序读"的。上层据此决定
是否启用 UE8M0 scale 路径、是否回退 Triton，比解析版本号可靠得多。

**验收**：未打补丁的 PyPI `sgl-deep-gemm==0.2.3` 上查询返回 `False`，
打了补丁的 wheel 上返回 `True`，且无需初始化 CUDA 上下文。

#### R4.1 上层已实现的识别契约（**已按你们的实现对齐**）

SGLang 侧的探测已落地（`deep_gemm_wrapper/configurer.py::_deep_gemm_declares_sm`），
按以下**优先级**识别，实现任意一种即可：

| 优先级 | 接口 | 语义 | 示例 |
|---|---|---|---|
| **1（你们已实现 ✅）** | `supported_arch_majors()` | 可调用，返回 arch major 序列 | `lambda: (9, 10, 11)` |
| 2 | `SUPPORTED_SM_VERSIONS` | 属性，SM 版本号集合，与 `get_device_sm()` 的 `major*10+minor` 一致 | `frozenset({90, 100, 110})` |
| 3 | `SUPPORTED_ARCH_MAJORS` | 属性，arch major 集合 | `frozenset({9, 10, 11})` |
| 4 | `supports_arch(x)` | 可调用，`x` 允许 `110` / `11` / `"sm110"` 任一形态 | `def supports_arch(x)` |
| 回退 | `__version__` 含 `"sm110"` | R5 版本号标记 | `"0.2.3+sm110"` |

**`supported_arch_majors()` 被放在首位，理由正是你们文档里写的那个场景**：
它读的是**已编译的 `_C.so`**，而不是 Python 侧，所以
「新 Python 侧 + 旧 `_C.so`」这种错配能被如实报告，而不会谎称支持。

对应地，上层对该调用同时捕获 `AttributeError`——旧 `_C.so` 没有
`supported_arch_majors` 符号时会抛这个异常；此时忽略它并继续往下列回退，
**不会让 SGLang 导入期崩溃**。

几点约定，请务必对齐：

- **必须能在纯 `import` 阶段回答**，不得初始化 CUDA 上下文（原因见文末时序说明）；
- `supports_arch` 若不是集合语义，请同时容忍 int 与 `"smNNN"` 字符串——
  上层会依次试 `110` / `11` / `"sm110"`，任一返回 `True` 即认定支持
  （因为 `False` 无法区分「不支持」与「参数类型不对」）；
- 集合/序列元素请用 `int`，上层会 `int()` 转换；非整数会被判为无效并回退。

**当前实测状态（2026-10-22）**：

| 载体 | `supports_arch` | `supported_arch_majors` | 上层走哪条 |
|---|---|---|---|
| 源码树 `sm110-merge` | ✅ | ✅ | 二进制探测（首选） |
| **已装 wheel `0.2.3+sm110`** | ❌ | ❌ | R5 版本号回退 |

也就是说：**你们源码已经实现，但当前环境里装的 wheel 还是旧的**。上层两种都能正确
处理——旧 wheel 走版本号回退、新 wheel 走二进制探测，不需要改上层代码。
重新构建并安装 wheel 后即自动切换到首选路径。

#### R4.2 探测失败时的上层行为（已实现）

若判定不支持 SM110，SGLang 会：

1. `ENABLE_JIT_DEEPGEMM=False` → `DEEPGEMM_SCALE_UE8M0=False`；
2. FP8 GEMM 后端自动落到 `triton`（`fp8_utils.initialize_fp8_gemm_config`）；
3. 打一条 warning 指向本文档。

**不会**静默产出错误结果，也**不会**在第一个请求时抛 `Unsupported architecture`。

### R5. 版本号的 local 段要保持 `+sm110` 语义

**现状**：✅ `sgl_deep_gemm/VERSION` = `0.2.3+sm110`。

**需求**：继续保持。这是目前 R4 未实现时**唯一**可用的探测信号，
上层会做 `"sm110" in deep_gemm.__version__`。若将来改名，请同步通知。

### R6. 提供一个轻量自检命令

**现状**：✅ **已实现并实测**。`python -m deep_gemm.doctor` 存在且随 wheel 一起安装，
输出如下（2026-10-22 真机，退出码 0）：

```
DeepGEMM doctor
============================================================
version               : 0.2.3+sm110
package               : .../site-packages/deep_gemm
_C.so                 : .../deep_gemm/_C.so (2,050,896 bytes, built 2026-10-22 14:22)  [ok]
CUDA_HOME             : <cuda-prefix>  [ok]
nvcc                  : .../bin/nvcc  [ok]
device headers        : .../include/curand_kernel.h  [ok]
torch                 : 2.14.1+cu130 (cuda 13.0)  [ok]
supported arch majors : sm9, sm10, sm11, sm12  [ok]
device                : NVIDIA Thor, sm_110 (compute capability 11.0), 20 SMs  [ok]
arch check            : sm11 is supported by this build  [ok]
JIT cache             : ~/.dj  (default; set DG_JIT_CACHE_DIR to move)
JIT cached kernels    : 977 entries in ~/.dj/cache

running a 128x128x128 bf16 GEMM through the dispatcher ...
dispatch              : bf16_gemm_nt ran, max|diff| vs fp32 ref = 0.0312  [ok]
============================================================

RESULT: OK
```

比原需求做得更好的两点：

1. **不打 `_C.so` 的 md5**——你们在 docstring 里说明了原因（二进制内嵌构建期源码路径，
   逐字节相同的构建 md5 也会不同，只会造成误报）。这个判断是对的。
2. **`arch check` 直接回答「sm11 是否被这个构建支持」**，正是 R4 要的语义。

**本节可以关闭。** 建议在 README 醒目位置提一句这个命令。

---

## 3. 需求 P2 — 降低使用门槛

### R7. 冷启动编译量应可调，且默认值要说明

**实测**：SGLang 侧会按 `_BUILTIN_M_LIST` 预热 M 形状。默认策略编译 **16384** 个 M
（`chunked_prefill_size=4096` 时），冷启动可达 **数分钟到 8 分钟**。
SGLang 已有 `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` 可降到 **2048**（8x 减少），
但**默认关闭且无文档**。

**需求（DeepGEMM-SM110 侧）**：README 里给出"首次启动慢"的预期管理，并说明：

- 这是 JIT 预热，**只发生一次**（cubin 会缓存）
- 缓存位置、大小、如何清理
- 如何在首次部署时用 `python -m sglang.compile_deep_gemm` 预热

> 注：该开关在 SGLang 侧，DeepGEMM 只需在文档里指路。

### R8. 缓存目录的优先级要写进 README

**实测结论**：`DG_JIT_CACHE_DIR` **是生效的**。把它指向一个全新目录后运行 FP8
GEMM，缓存确实写在该目录下（`<dir>/cache/<family>.<hash>/`），默认的 `~/.dj/cache`
不受影响。

机制（`include/deep_jit/utils/env.hpp`、`cache/disk.hpp`）：

- 变量名 = `<prefix>_JIT_CACHE_DIR`，其中 prefix 由上层 Python 传入（DeepGEMM 用 `DG`），
  找不到时回退到 `DJ_JIT_CACHE_DIR`
- 未设置任何变量时，默认 `$HOME/.dj`
- 值支持**冒号分隔的多路径**：第一个为可写根，其后为只读查找根

**需求**：README 写清这三点，尤其「多路径只读查找根」这个能力对
「镜像里预置缓存 + 运行时另写增量」很有用，但目前完全没文档。

### R9. `import deep_gemm` 的导入期要求要写清

**现状**：🟡 **一半完成**。报错已从裸 `AssertionError` 改成带完整修复指引的
`RuntimeError`（`cuda_helpers.py:31`），实测输出：

```
RuntimeError: Cannot locate a CUDA installation.

  CUDA_HOME / CUDA_PATH is not set, there is no `nvcc` on PATH, and
  ...
  两种修法（指向含 bin/nvcc 与 include/curand_kernel.h 的目录 / 加 CPATH）
```

**但仍发生在 `import` 期**（`deep_gemm/__init__.py:523` 调用 `find_cuda_home()`），
所以「只想 `import` 看版本、不编译任何内核」的场景依然必须先设 `CUDA_HOME`。

**剩余需求**：把该检查**延后到真正需要编译时**。这会改善两个真实场景：
`python -m deep_gemm.doctor` 想检查「CUDA_HOME 是否配错」却先在 import 期挂掉；
以及上层做能力探测时（SGLang 探测的是 Python 侧属性，本身不需要 CUDA_HOME，
但目前 import 这关就过不去）。

> 对 SGLang 的影响：实际部署里 `CUDA_HOME` 总是设好的，所以不阻塞；
> 但做能力探测的单测需要在有 `CUDA_HOME` 的环境里跑。

### R10. 避免 cwd 遮蔽（文档层面）

**现状**：🟡 **未文档化，但已有更好的替代**。`sgl_deep_gemm/README.md` 目前只有
`Introduction` / `Usage` 两节，没有提 cwd 遮蔽；不过 R6 的
`python -m deep_gemm.doctor` 已经绕开了这个歧义——它从已安装的包运行，
不受 cwd 影响。

**实测问题**：在仓库根目录（含 `deep_gemm/` 源码目录）执行
`python -c "import deep_gemm"` 会导入**源码树**而非已安装的 wheel，
报 `ImportError: DeepGEMM extension is missing`。这对刚装完 wheel 想验证的用户非常容易踩。

**需求**：README 的验证步骤要强调「**换到仓库外的目录**再验证」，
或指向 `python -m deep_gemm.doctor`（R6）。

> 注：SGLang 侧的 `setup_deepgemm.sh` 已在**自己的**验证环节避开这一点——
> 它把解释器检查放进临时中性目录执行（`cd "$NEUTRAL_DIR" && "$PY" -`），
> 因此即使用户恰好在 DeepGEMM 源码根下运行该脚本，也不会误导入源码树。
> 这只解决上层脚本自身的问题，**不影响** R10 对 README 的诉求。

---

## 4. 建议 P3 — 可选，长期收益

### R13. `CUDA_HOME` 在 Thor 上不是可选的（文档修正）

**实测问题**：README 的 "CUDA lookup" 一节称

> You do **not** always need to set `CUDA_HOME`. DeepGEMM looks for CUDA in
> this order: 1. `CUDA_HOME`/`CUDA_PATH`; 2. `nvcc` on `PATH`; 3. `/usr/local/cuda`.

在 Thor 上这条**不成立**，三级回退全部落空：

```text
1) CUDA_HOME / CUDA_PATH : 未设
2) PATH 上的 nvcc        : 无
3) /usr/local/cuda       : 不存在
```

原因是 Thor 的 CUDA 来自 pip `nvidia-*` 包（`nvidia-cuda-nvcc` 等），
既不在 `/usr/local/cuda`，也不在 `PATH` 上。

**需求**：README 的 Install 一节应把 `CUDA_HOME` 从"可选"改成
"**Thor 上必须设置**"，并说明要指向一个同时含 `bin/nvcc` 与
`include/curand_kernel.h` 的前缀。三级回退可以保留（对其他平台有用），
但要标注"在 Jetson Thor 上不适用"。

**验收**：新用户在 Thor 上照 README 走到 `bash build_sgl_deep_gemm.sh`
不会卡在 `Cannot locate a CUDA installation`。

> 上游侧已同步：`sglang-inscope` 的 README §3.5 与 `inscope/README.md`
> 都已把这句改成"必须显式设置"。

### R11. 预编译 wheel 发布（明确指出适用范围）

**现状**：✅ **已实现**。仓库发布了 GitHub Release `v0.2.3-sm110`，附件：

```
sgl_deep_gemm-0.2.3+sm110-py3-none-linux_aarch64.whl    4.6 MB
```

**这点很重要**：它把「必须先在本机编译才能用」变成「下载装即可」，对 Thor
用户是质的差别——省掉 clone 150 MB CUTLASS 子模块 + 约 1 分钟构建。

**仍需注意**（建议后续在 release notes 里点明）：`_C.so` 只对一个
(cpu arch, torch ABI, CUDA major) 三元组有效。这个 wheel 的 `linux_aarch64`
平台标签已经把它限定在 aarch64；但 **torch 版本和 CUDA major 没有编码进
wheel 标签**，所以 `torch 2.14.1 + cu130` 之外的组合仍需自行构建。
建议在 release notes 里写成：

> 适用：aarch64 + PyTorch 2.14.1 + CUDA 13.0。其他组合（不同 torch / CUDA
> major）请用 `build_sgl_deep_gemm.sh` 在本机构建——`_C.so` 的 ABI 与之绑定，
> 装错会在 `import` 期或首次内核调用时报错。

**不要**在仓库里提交 `.so` 二进制（release 附件可以，仓库内不行）。

### R12. 让 `setup.py` 与 `build_sgl_deep_gemm.sh` 的关系显式化

**现状**：仓库里有两条构建路径——

- 上游 `setup.py`（`name='deep_gemm'`，版本取自 `deep_gemm/__init__.py` = `2.8.0`）
- `sgl_deep_gemm/` + `build_sgl_deep_gemm.sh`（`name='sgl-deep-gemm'`，版本取自 `sgl_deep_gemm/VERSION`）

两条路径产出**不同 distribution 名、不同版本号**的包。新用户很容易 `pip install .`
走到上游 `setup.py`，得到一个 SGLang 用不上的 `deep_gemm 2.8.0`。

**建议**：README 开头用一段话说清「**SGLang 用户请用 `build_sgl_deep_gemm.sh`；
`setup.py` 是上游遗留路径，不要用**」。

---

## 5. 优先级建议

| 优先级 | 项 | 理由 |
|---|---|---|
| **P0** | R1, R3 | 自动安装的前提 |
| **P0** | R4 | 上层能力探测的唯一可靠信号 |
| **P1** | R6, R9, R10 | 直接决定用户能否自己排障 |
| **P1** | R5 | 当前唯一的探测信号，需保持稳定 |
| **P2** | R2, R7, R8 | 减少重复踩坑 |
| **P3** | R11, R12 | 长期可维护性 |

---

## 6. 上游（sglang-inscope）侧承诺

为配合上述改动，`sglang-inscope` 侧会做：

| # | 事项 | 状态 |
|---|---|---|
| 1 | **能力探测替代硬编码**：`configurer.py` 的 SM110 判定改为优先用 R4 接口、回退 R5 版本号 | ✅ **已实现并测试**（见 R4.1/R4.2） |
| 2 | **失败保持响亮**：探测到不支持则回退 Triton 并输出明确 warning | ✅ **已实现并测试** |
| 3 | **环境脚本自动检测**：`setup_deepgemm.sh` 完成 clone → 构建 → 安装 → 校验全流程 | ✅ **已实现并端到端验证** |
| 4 | **安装后校验**：装完立即检查能力查询与真实 dispatch，输出 `[ok]` / `[!!]` | ✅ **已实现并端到端验证** |

第 1、2 项已落地，意味着**DeepGEMM 侧可以独立演进**：无论你们实现 R4 的哪个接口、
或者暂时都不实现，上层都能正确判断并选择合适的后端。你们升级版本号、增删接口，
不会再让 SGLang 静默跑错或运行时崩溃。

#### 6.1 解耦的双向验证（2026-10-22 实测）

同一份 SGLang 代码在四代 DeepGEMM 上分别验证：

| DeepGEMM 形态 | 探测结果 | `ENABLE_JIT_DEEPGEMM` | 行为 |
|---|---|---|---|
| 新构建（有 `supported_arch_majors()`） | `True` | `True` | 走 DeepGEMM（首选路径） |
| 新 Python 侧 + 旧 `_C.so`（查询抛 `AttributeError`） | `None` | `True` | **不崩**，回退版本号 |
| 旧构建（只有 `+sm110` 版本号） | `None` | `True` | 回退版本号 |
| stock 上游（无标记、无接口） | `None` | `False` | 回退 Triton + warning |

第 2 行是关键：`supported_arch_majors()` 读的是已编译 `_C.so`，所以
「Python 侧新、二进制旧」的错配会抛 `AttributeError`，上层捕获后继续回退，
**不会让 SGLang 导入期崩溃**。

#### 6.2 `setup_deepgemm.sh` 端到端实测输出

```
== verify
deep_gemm version    : 0.2.3+sm110
supported arch majors: (9, 10, 11, 12)
SM110 dispatch       : [ok] reported by _C.so
device capability    : sm_110
dispatch smoke test  : [ok] bf16_gemm_nt ran
```

同时验证了三条失败路径均给出可照做的提示并以非零退出：
无 Python / Python 无法 import torch / `CUDA_HOME` 未设且 `nvcc` 不在 PATH。

**幂等性**：已装好 SM110 支持的 DeepGEMM 时直接退出（`FORCE=1` 可强制重建），
不会重复下载 150 MB 子模块。

> 注意时序：SGLang 侧的 `DEEPGEMM_SCALE_UE8M0` 在 `Scheduler.__init__` 期间确定，
> **早于** 模型加载。因此能力探测必须在模块导入期完成，不能在首次 GEMM 调用时才探测（R4 的"无需 CUDA 上下文"要求即源于此）。
