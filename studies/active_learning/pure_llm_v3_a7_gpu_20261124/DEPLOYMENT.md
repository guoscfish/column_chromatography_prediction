# A7 CUDA 部署

## 平台现状

北京时间 2026-11-24 的审查固定离线 V3 基线为 `b295939a2f6ec0bd90ed14e5c9a9b92aae0909d7`。数值运行的持久回执记录 `b6b03b9e` 和独立 CUDA 源码的 SHA。本次核对了 166 个共同 `src/qgeognn_al/` 与 `application/` 文件，均与离线基线逐字节相同。A7 包含独立的 CUDA 设备适配器，在新工作区复用这些已冻结的训练源码。

服务器使用 GPFS 下的 `/1/column_chromatography_prediction`。数值工作区是 `worktree`，解释器是 `environment/venv/bin/python`。数值队列、日志和 checkpoint 在 `formal_cuda/row4g_b32_20261124`。11 月 24 日的回执显示两条数值轨迹在 SSH 断开后继续推进。平台环境为 Ubuntu 24.04.1、Python 3.12.3、PyTorch 2.10.0+cu128、CUDA 12.8、cuDNN 91002、HTTPX 0.28.1、两张各 24 GiB 的 RTX 3090。

北京时间 11 月 25 日 09:39 恢复 SSH 连接后，只读探针显示 Random、Hybrid 与 CW-LCMD 各 5 个正式任务均已成功。两张 RTX 3090 当时没有计算进程，各有 24126 MiB 可用显存。GPFS 报告约 108 TiB 全局剩余，用户配额与实例租期未确认。五个 Random L333 锚点文件均存在。随后在独立目录安装 SDK，并核验五份锚点和私密 Key 池。以下命令可重新获取平台状态。

```bash
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
remote=/1/column_chromatography_prediction
ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes eit-hpc \
  "$remote/environment/venv/bin/python -B - probe" < "$study/deployment.py"
```

探针用只读元数据检查显卡、OpenAI SDK 版本、队列各方法状态、五份 Random anchor 文件是否存在、磁盘及 Key 池配置文件是否存在和权限。包版本来自当前解释器，GPU 显存来自 `nvidia-smi`。锚点“存在”需在预检中进一步核对合同和最初 checkpoint 的 SHA。探针不会打开 Key 文件或发请求。

## 已发布入口的依赖复核

| 已发布 V3 的实际路径 | 本轮 A7 的入口 |
| --- | --- |
| `trajectory.py:20,24` 的 `load_transition` 调用 `load_historical`。后者在 `frozen_state.py:46,100` 核对历史请求、目录和决策。 | 读取冻结 split、公共 4g 图和同 seed CUDA L333 连续工件或使用授权后的固定初始化初始训练。首轮状态没有历史 LLM 运行目录。 |
| `trajectory.py:153,185` 的 `_fit` 通过 `historical_round0_paths` 读取 LCMD 历史 context 和 scrubbed graphs；CUDA 分支第 203 行还依赖未发布数值脚本。 | `backend.py` 从公用图缓存准备同训练规则特征和 label-scrubbed graph；`cuda_runtime.py` 仅提供原训练和推理的设备适配。 |
| 旧 `runner select` 和 `trajectory advance` 分别绑定逐轮授权，旧正式配置为 Sol 6.1/xhigh。 | 新 scheduler 使用一次轨迹授权和逐轮合同，固定 Sol 6.2/ultra，只在接受的 freeze 后揭示。 |
| 旧 `transport_manager.py:303` 依赖 OpenAI 客户端默认 Key 选择，SDK 与配置严格相等。 | 正式配置锁 SDK 2.28.0，Key 在外部池中分配至具体物理请求。第三方能力在实际 round 0 前保持未核验。 |

[已发布 trajectory](../pure_llm_v3_20261124/pure_llm_v3/trajectory.py) · [历史快照桥接](../pure_llm_v3_20261124/pure_llm_v3/frozen_state.py) · [旧传输](../pure_llm_v3_20261124/pure_llm_v3/transport_manager.py)

## 构建与安装

在本地仓库根目录构建。包内逐文件 SHA 包含基线提交的模型代码、PureLLM-V3 runtime 和 Prompt、4g 原始与 canonical CSV、E0 metadata、原图缓存、五份冻结 Row split 和对应协议。正式包另含新研究入口、配置、授权模板及文档。原 V3 的传输模块使用经 SHA 固定的版本，在物理尝试开始前领取凭据租约。其余原 V3 和训练代码读取固定 Git blobs。平台探针回执、模型、历史 LLM 请求、已填授权和凭据由运行环境单独保管。安装器核对整体 SHA 及逐文件 SHA，并要求新的真实目录。

```bash
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
remote=/1/column_chromatography_prediction
bundle_dir=$(mktemp -d)
bundle="$bundle_dir/purellm-a7.tar.gz"
python3 "$study/deployment.py" build --output "$bundle"
sha=$(shasum -a 256 "$bundle" | awk '{print $1}')
release="$remote/purellm_a7/releases/$sha"
incoming="$remote/purellm_a7/incoming/$sha"

ssh -o BatchMode=yes eit-hpc \
  "umask 077; mkdir -p '$remote/purellm_a7/incoming' '$remote/purellm_a7/releases'; mkdir '$incoming'"
scp "$bundle" "eit-hpc:$incoming/purellm-a7.tar.gz"
ssh -o BatchMode=yes eit-hpc \
  "$remote/environment/venv/bin/python -B - install --archive '$incoming/purellm-a7.tar.gz' --destination '$release' --sha256 '$sha'" \
  < "$study/deployment.py"
```

CUDA 训练依赖现有数值实验环境。正式 SDK 固定 2.28.0。若平台解释器未安装该版本，使用本次独立的 provider 包目录和一组审核后的 Linux x86_64 Python 3.12 wheels。下列安装使用先前已准备的本地 wheel 目录，不修改共享环境。运行 `requirements_provider.txt` 明列 SDK 和平台既有冻结清单中缺少的三个直接依赖，HTTPX 等已合格的依赖来自原环境。

```bash
# 在远端 shell 运行。把 RELEASE_SHA 和 SDK_WHEELHOUSE 换为本次真实路径。
remote=/1/column_chromatography_prediction
release="$remote/purellm_a7/releases/RELEASE_SHA"
python="$remote/environment/venv/bin/python"
overlay="$remote/purellm_a7/provider/openai-2.28.0"
study=studies/active_learning/pure_llm_v3_a7_gpu_20261124
umask 077
mkdir -p "$overlay"
"$python" -m pip install --no-index --no-deps --find-links SDK_WHEELHOUSE \
  --target "$overlay" -r "$release/$study/requirements_provider.txt"
export PYTHONPATH="$overlay"
"$python" -B -c 'import openai, httpx; assert openai.__version__ == "2.28.0"; print(openai.__version__, httpx.__version__)'
```

如需离线收集供以上命令使用的 wheel，可在有批准的依赖镜像访问的机器执行 `python3 -m pip download --only-binary=:all: --no-deps --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --abi cp312 -r "$study/requirements_provider.txt" -d SDK_WHEELHOUSE`，再按组织允许的方式传输。通过版本导入只能证明 SDK 可用。Token4Research 的真实 ultra、served 版本和完整上下文兼容性待正式 round 0 验证。

## L333、作业隔离与资格核验

数值三方法都从有序 L333 使用固定初始化重新训练；合格的同 seed 公共 CUDA checkpoint 可以复用。默认锚点目录是 `$remote/formal_cuda/row4g_b32_20261124/trajectories`，命名 `random-<seed>`。A7 重新计算当前 U 的预测。它仅使用数值运行的 L333 模型文件及其验证身份的收据，其他方法的选择/标签/预测明细不进入 selector。若锚点缺失，只能在正式授权后按同协议重训 L333。

数值队列的 GPU 锁是其根目录中的 `physical-gpu-<gpu>.lock`，其 worker 将锁保留到退出并传给训练子进程。A7 的 GPU 阶段在同路径协调。不修改数值 `queue.json`，不暂停既有任务。API 阶段释放 GPU 资源，首轮资格验证通过后调度器才进入正常并发。仅探测环境不需要领取 GPU 锁。

Key 池路径、用户端静默录入和一次正式授权见 [正式运行说明](README.md)。部署后从 release 根目录运行 scheduler。作业的 `--run-dir` 应是独立的 GPFS 子目录。11 月 25 日的远端资格检查已完成五份同 seed 锚点的 CUDA 推理和 L333 请求构造，实际代理能力由获授权的首轮验证。

## 已执行的本地检查

- 使用 201 个已发布文件与 A7 backend、CUDA adapter 和正式配置建立独立目录。目录中无 NoCW 历史研究，也没有未发布的数值 CUDA 脚本。
- 在该目录实际准备五份公共图缓存并用合成模型预测构造正式 A7/M2 请求。五个 seed 的 L/U 都为 333/2997，初始历史为空。按 seed 157、887、2357、6101、12203 的请求 JSON 序列化字节数为 506191、506223、506323、506950、506611。这次未训练、未发 API，也未揭示新候选标签。
- 已对当时的部署包执行完整打包、校验安装和两套 CLI 导入。在干净安装目录通过 scheduler 五 seed 合成预检，得到上述同样的 5 个请求字节数。
- CPU-only 主机分别拒绝 CUDA 构建和单卡设备资格。部署 allowlist、正式资产覆盖、整体 SHA、fresh destination 与 tar 软链接拒绝的回归测试全部通过。

## 在同一基础镜像的新容器恢复

`environment/venv` 和 A7 的 SDK overlay 位于 GPFS。2026-11-25 的实机检查确认 venv 使用基础镜像的 `/usr/bin/python3.12`、Python 3.12.3、x86_64 和 glibc 2.39。CUDA 训练环境为 PyTorch 2.10.0+cu128；独立 overlay 提供 OpenAI 2.28.0，沿用共享环境中的 HTTPX 0.28.1。保留相同基础镜像、驱动、GPU 可见性和绝对挂载路径，可重新使用这两处目录。跨新容器的恢复仍需在目标容器实际验证。

1. 在同基础镜像的新容器挂载原 GPFS 到相同的 `/1/column_chromatography_prediction`。保留原 release、run、GPU lease 路径和合同。不要把正在运行的目录改作新的轨迹。
2. 用原 venv 解释器并显式设置原 SDK overlay。核对基础解释器、软件版本和 CUDA 可见性，重新运行 `contract` 并比较原批准 SHA。
3. 在新容器的私密 `/root/.config/purellm/` 重新配置 Key，或在任务环境注入平台 Secret。Key 不属于 GPFS、部署包或镜像。使用原正式授权，核对旧 worker 已退出。对状态未知的请求先作对账，再执行原命令的 `resume`。

```bash
remote=/1/column_chromatography_prediction
py="$remote/environment/venv/bin/python"
export PYTHONPATH="$remote/purellm_a7/provider/openai-2.28.0"
"$py" -B -c 'import sys, torch, openai; print(sys._base_executable, sys.version.split()[0]); print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), openai.__version__)'
```
