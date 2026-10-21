# 开发流程：fork 还是 patch

本文回答：**这个项目的代码怎么改、怎么提交、怎么交付、上游更新了怎么办。**
对照的是两份前作各自的工作流，以及它们为什么那样做。

---

## 1. 两份前作的工作流解剖

### 1.1 djs FeaturePrefill：patch 对上游 tag，附一个安装器

```text
patches/sglang-0.5.18-fyh-direct/
├── feature-prefill-v0.5.18.patch     89KB，17 文件，+887
├── feature-cache-tail-margin-1.patch  后续补丁，叠在上面
├── feature-no-kv-canary.patch         canary，未应用
├── apply.py + runtime/                更早的安装器：把 runtime 快照塞进本地 wheel
├── manifest.json                      upstream repo/tag/commit + patch sha256
└── README.md
```

`manifest.json` 钉死来源：

```json
{"upstream": {"repository": "https://github.com/sgl-project/sglang",
              "tag": "v0.5.18",
              "commit": "71de97b264b04dcd514cf904003028aefe9775c8"},
 "patch": {"filename": "feature-prefill-v0.5.18.patch", "sha256": "2c490cb7..."}}
```

应用方式（clean 的 tag 检出上）：

```bash
git apply --check /path/to/feature-prefill-v0.5.18.patch
git apply /path/to/feature-prefill-v0.5.18.patch
```

**它自己承认这套不好维护**，README 原话：

> `apply.py` and `runtime/` remain the installer for the project-local wheel.
> They are not the recommended source format for a fork.
> The patch is an exact deployed baseline, not a minimal upstream PR:
> it still includes legacy Qwen2 capture code and tokenizer-manager config
> changes outside FeaturePrefill.

三个问题，全部在这句话里：

1. **patch 是「部署快照」而不是「可评审的改动」**——里面夹着与 FeaturePrefill 无关的
   tokenizer-manager 配置改动和 legacy `fyh-capture:` 落盘代码；
2. **叠补丁没有依赖管理**——`feature-cache-tail-margin-1.patch` 必须按顺序叠在
   前面那个 89KB 补丁之上，前者一变，后者上下文全错；
3. **没有 git 历史**——为什么某一行是这样，只能靠 patch 文件里的 context 猜。

### 1.2 zyt CoolWatch：哈希钉死的 wheel + 影子源码 + 可导出补丁

```text
sglang-activation-runtime/
├── upstream.json              0.5.19 + wheel 文件名 + sha256
├── environment.json           python/torch/triton/... 版本清单
├── .source/                   从校验过的 wheel 抽出的源码（不进 git）
├── .venv/                     独立解释器，torch/CUDA 只读复用现有环境
├── src/sglang/srt/activation_capture/   新增模块，git 里的权威副本
├── patches/activation-capture.patch     对官方文件的可评审 diff
├── tools/bootstrap.py         校验 wheel hash → 抽源码 → 打补丁；不动已有 .source
├── tools/export_patch.py      从工作区反向导出补丁
├── tests/                     独立测试
└── validation.json            18 个原始文件 byte-identical、8 单测、HTTP 测试
```

它解决的是**一个特定约束**：

> 原 SGLang 安装和 `sglang_fe`、`sglang_det` 提取方案保留。
> 运行时加载独立的源码副本。

也就是说：**别人的环境不能动**。它靠 `./serve` 把 `PYTHONPATH` 指到 `.source` 实现共存，
并用 `validation.json` 证明「原安装里对应的 18 个文件一个字节都没变」。

好处是干净、可复现；代价是两套源码并存，改代码时要记得同步 `src/` → `.source/`：

> 新增模块以 `src/` 为准，调试前同步到 `.source/sglang/srt/activation_capture/`；
> 对官方文件的修改用 `tools/export_patch.py` 导出。

**这是一条真实的工作负担**，它自己在文档里要专门提醒。

---

## 2. 为什么两种都不直接适用

| | djs | zyt | 我们 |
|---|---|---|---|
| 有自己控制的 fork 吗 | 没有（改别人的部署环境） | 没有（保护既有安装） | **有**：`yeasen-z/sglang-inscope` |
| 仓库状态 | — | — | `main@92580581da`，**pristine 上游，无自定义 commit** |
| 运行时从哪来 | 打补丁的 wheel | 影子 `.source` | **fork 本身**，`pip install -e python` |
| 要保护的既有安装 | 无 | 有 | **无**（新环境 `/opt/miniconda3/envs/inscope`） |

他们用 patch/影子源码，是因为**没有仓库控制权**。
我们有一个干净的 fork 和一个新建的专用环境，这两个约束都不成立。

所以照搬任何一套，都是在为不存在的问题付成本：
- 照搬 djs → 得到一个不可评审的部署快照，且改一行要重生成 89KB 补丁；
- 照搬 zyt → 得到两套源码要同步，而 `pip install -e` 本来就是干这个的。

---

## 3. 推荐工作流：fork 优先，patch 只作为发布产物

### 3.1 唯一真源是 git 分支

```bash
cd /DATA/disk2/workspace/zmosg/sglang-inscope
git checkout -b feat/inscope            # 从 92580581da 切出
```

- **开发**：直接在 `python/sglang/srt/feature_capture/` 写代码 + 改 13 个文件；
- **运行**：`/opt/miniconda3/envs/inscope/bin/python`，包以 editable 装在这个工作区上，
  改完立即生效，不需要重新安装、不需要同步任何影子副本；
- **测试**：`test/registered/unit/feature_capture/`（CI 可发现）+
  `test/manual/feature_capture/`（本地 E2E）；
- **提交**：每个 P 阶段一个 commit，见 §4。

### 3.2 patch 什么时候生成、给谁

只有一个场景需要 patch：**别人要在 stock sglang 上复现，且不接我们的 fork**。

```bash
# 生成（只含 python/，不含 fork 里的测试与文档）
git diff main..feat/inscope -- python/ > inscope/patches/feature-capture-<ver>.patch

# 校验能干净应用
git apply --check --directory=. inscope/patches/feature-capture-<ver>.patch
```

配合一个 `upstream.json` 风格的钉死文件（借 zyt 的做法，成本极低、价值明确）：

```json
{
  "base": {"repository": "https://github.com/yeasen-z/sglang-inscope",
           "branch": "main", "commit": "92580581da6fc8542a997a00c1a8b71b03497680"},
  "patch": {"filename": "feature-capture-v1.patch", "sha256": "..."},
  "feature_capture_schema": "inscope.feature_capture.v1"
}
```

**关键纪律：patch 是从分支导出的产物，永远不手改 patch。**
djs 的问题正是反过来了——patch 是权威副本，代码只存在于 patch 里。

---

## 4. 提交纪律（为了 rebase，不是为了好看）

上游 sglang 的 `main` 每天在动。我们的 fork 现在钉在 `92580581da`（2026-10-16），
迟早要跟进。跟进成本完全由「改了多少既有文件」决定，所以：

### 4.1 分阶段提交，让每个 commit 可独立回退

| commit | 内容 | 依赖 |
|---|---|---|
| `feature_capture: add core contracts and recorder` | `feature_capture/` 全部通用模块与对应单测 | 无 |
| `feature_capture: wire request-scoped runtime path` | Engine/IPC/Req/ForwardBatch/ModelRunner/auxiliary output 接线与 request 测试 | 上一个 |
| `feature_capture: add Qwen3 model adapters` | Qwen3、Qwen3.5、language-model-only 配置与真实模型验收脚本 | 前两个 |
| `docs: record feature-capture validation and workflow` | InScope 设计、交接、环境脚本与手工测试文档 | 前三个 |

**模型适配单独成 commit**，因为它是最可能与上游冲突的部分；
单独成 commit 意味着如果某个模型文件被上游大改，可以只丢掉那个 commit 重写，
不影响通用运行时。

### 4.2 新增目录 vs 改既有文件

| 类型 | 文件数 | rebase 冲突风险 | 原因 |
|---|---:|---|---|
| **新增** `feature_capture/`（8 文件） | 8 | **零** | 上游没有这个目录 |
| 新增测试（`test/registered/unit/feature_capture/`） | 2 | **零** | 同上 |
| 改既有文件 | 13 | 中/低 | 见下表 |

冲突风险分级（这是架构选择的**真正**理由，不只是"干净"）：

| 文件 | 我们改什么 | 与上游冲突风险 | 缓解 |
|---|---|---|---|
| `models/qwen3.py` 等 3 个 | 换一个 `capture_output=` 实参 + 加 `capture=` + 类属性 | **中**：上游常重构层循环内部 | 只换实参，不碰循环体其他行；单独 commit |
| `managers/scheduler.py` | 3 处各 1–4 行，都在函数末尾 | 低 | 靠 `append_auxiliary_output` 单行插入 |
| `managers/schedule_batch.py` | `Req` 加字段 + 一行 `key_limit` 覆写 | 低 | 都在已知稳定的位置 |
| `managers/io_struct.py` | 加 4 个字段 | 低 | 纯追加 |
| `managers/tokenizer_manager.py` | 分词后校验 + 写 `meta_info` | 低 | 纯追加 |
| 其余 7 个 | 各 2–9 行追加 | 低 | — |

**对照 djs**：它改 `qwen2.py` +59/-2 和 `qwen3.py` +163/-12，
而且改动**在层循环的计算逻辑内部**（重排 `hidden_states + residual`、加 `break` 早停）。
上游只要重写层循环，这些 hunk 全部作废。
我们把改动压到「一个函数实参」，正是为了让这类重构不影响我们。

### 4.3 上游跟进的固定动作

```bash
git fetch origin main && git log --oneline HEAD..origin/main -- python/sglang/srt/{layers,managers,model_executor,models}
git rebase origin/main
# 1) 纯 CPU 契约测试：30 秒内给答案，不需要 GPU
python -m pytest test/registered/unit/feature_capture/ -q
# 2) 再跑 P2 的数值对照（GPU，慢）确认 capture 点没被重构挪走
```

第 1 步的价值：契约与行映射占了逻辑风险的一多半，而它们**没有 GPU 依赖**，
所以 rebase 后能在 30 秒内判断「这次跟进是不是安全的」。
这是把 `spec.py` / `plan.py` 设计成不 import torch 的额外收益。

---

## 5. 具体命令

### 5.1 建环境（P0）

```bash
bash inscope/scripts/setup_env.sh
# 环境变量可覆盖：ENV_PREFIX / PIP_INDEX / PY_VERSION / REINSTALL
```

已知的三处处理，写在脚本注释里：先装 torch 再 `--no-build-isolation`（否则下载两份 torch）；
不需要 cargo/nvcc（CUDA 版 pyproject 不声明 rust-extensions）；用镜像。

### 5.2 跑测试

```bash
PY=/opt/miniconda3/envs/inscope/bin/python
cd /DATA/disk2/workspace/zmosg/sglang-inscope

# 契约与行映射（CPU，秒级）
$PY -m pytest test/registered/unit/feature_capture/ -q

# 走 CI 注册路径（验证 register_cpu_ci 标注正确）
$PY test/run_suite.py --hw cpu --suite base-a-test-cpu

# 本地 E2E（P2 起，需 GPU 与模型）
export INSCOPE_QWEN3_FP8_MODEL=/DATA/disk1/models/Qwen3-8B-FP8
$PY -m pytest test/manual/feature_capture/ -q
```

### 5.3 导出并校验 patch

```bash
git diff main..feat/inscope -- python/ > /tmp/fc.patch
T=$(mktemp -d) && git worktree add --detach "$T" main
(cd "$T" && git apply --check /tmp/fc.patch) && echo "applies cleanly"
git worktree remove "$T"
```

---

## 6. 三条明确的纪律

1. **不 patch wheel、不维护 `.source` 影子副本。** 我们拥有 fork，
   也没有要保护的既有安装。引入影子副本只会带来「改了 A 忘了同步 B」的错误。
2. **不改 `python/pyproject.toml` 的版本号或依赖。** 依赖漂移会让
   `manifest`/`upstream.json` 失去意义，也让别人无法判断某个结果是哪套依赖跑出来的。
   需要额外的开发依赖时，装进 conda env，不写进 `pyproject.toml`。
3. **patch 是导出物，不是真源。** 任何对 patch 的手改都会丢失。

---

## 7. 与前作的流程对照（一页）

| 维度 | djs | zyt | 我们 |
|---|---|---|---|
| 真源 | patch 文件 | `src/` + patch | **git 分支** |
| 运行方式 | 打补丁的 wheel | `PYTHONPATH` → `.source` | `pip install -e` 到 fork |
| 既有安装 | 被改 | 被保护 | 无（新 env） |
| 上游钉死 | `manifest.json`（tag+commit） | `upstream.json`（wheel sha256） | 分支基点 commit + 未来加 `upstream.json` |
| 补丁叠加 | 手工按序叠，无依赖管理 | 单补丁 | 不需要（分支上就是全部） |
| 可评审性 | 差（夹带无关改动） | 好 | 好（commit 分阶段） |
| 模型改动位置 | 层循环计算内部 | 层循环内加 `tap()` | 换一个函数实参 |
| 上游重构的抗性 | 低 | 中 | **高** |
| 改动既有文件数 | 17 | 18 | **13**（且多为纯追加） |
