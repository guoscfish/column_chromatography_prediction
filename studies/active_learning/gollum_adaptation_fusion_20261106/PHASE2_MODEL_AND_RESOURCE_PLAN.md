# Phase 2 模型与资源审计

日期：2026-11-06。实际资源回执见 `phase2/*_offline_preflight.json`、`phase2/toy_gradient_flow.json`；模型服务端 metadata 见 `phase2/upstream/MODEL_METADATA.json`。没有新增模型下载、依赖安装、付费API或云计算任务。

## 候选模型与可比性

| 项目 | Model A：google-t5/t5-base | Model B：Qwen/Qwen3-Embedding-0.6B |
|---|---|---|
| 核实版本 | `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| 选择理由 | 与 GOLLuM 官方 t5-base encoder 最直接可比 | 专门的文本 embedding 模型，不是聊天榜单选型 |
| 参数/权重 | 全 encoder-decoder metadata：222,903,936 float32参数；实际仅加载encoder | 595,776,512 BF16参数，28层，hidden1024 |
| 单个 safetensors | 891,646,390 bytes（约0.830 GiB） | 1,191,586,416 bytes（约1.110 GiB） |
| 官方许可证 | Apache-2.0 | Apache-2.0 |
| 预训练目标 | 文本到文本框架，C4等去噪及多任务学习；非专用化学embedding | 多阶段文本相关性训练，改进InfoNCE/对比目标及model merging；非化学专用模型 |
| tokenizer | T5 SentencePiece/Unigram；有tokenizer.json快速实现，慢实现依赖sentencepiece | Qwen2Tokenizer/BPE；审计EOS处理，禁止套chat template |
| pooling | GOLLuM encoder最后层attention-masked mean，不做L2归一化 | 官方最后有效EOS token，随后L2归一化 |
| 输入长度 | 配置 n_positions=512，本研究冻结上限512 | config最大32768；本研究先限512。tokenizer_config的131072不能覆盖模型上限 |
| LoRA | 顶部3/12层所有Linear，r4/alpha16/rsLoRA；估计165,888 adapter参数，实际加载后核验 | 原则上顶部7/28层全部Linear；实际adapter参数与模块名需加载后统计 |

以上版本、参数数和精确权重字节来自公开 Hub metadata API；下载清单只应取一份 safetensors及必要配置/tokenizer，不能把 T5 仓库全部4.46 GB（含多框架重复权重）当成必要下载量。[T5 model card](https://huggingface.co/google-t5/t5-base)、[T5 config](https://huggingface.co/google-t5/t5-base/blob/main/config.json)、[T5 files](https://huggingface.co/google-t5/t5-base/tree/main)、[Qwen3 model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)、[Qwen3 config](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config.json)、[Qwen3 files](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/main)

Qwen3 pooling、EOS及训练目标有[官方技术报告](https://arxiv.org/html/2506.05176v3)和[Pooling配置](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/1_Pooling/config.json)支持。模型对通用文本、代码、数值字符串可编码，不构成SMILES化学保真、数值单调性或柱色谱表示有效的证据。T5 的未知token、SMILES round-trip、长SMILES超限和数值字段差分需要真 tokenizer 测试，当前未执行。对于Qwen3也必须实际确认EOS自动追加；本地代码缺失EOS即失败，不默默使用普通末token冒充官方EOS表示。

## 已检查的本地资源

硬件为 Apple M4 MacBook Air、24 GB统一内存，Metal硬件存在。当前受控 Python 运行时 CUDA不可用，MPS已编译但 `is_available=False`，因此本轮按CPU规划。默认系统python缺少研究库；复用现有 `chromatography` conda环境（Python3.11.14），不创建新环境。

已安装 Torch2.13.0、Transformers5.4.0、safetensors0.7.0、NumPy2.4.6、SciPy1.17.1、pytest9.1.1；**PEFT、GPyTorch、BoTorch、sentencepiece缺失**。Transformers满足Qwen3官方最低4.51要求。`tokenizer.json` 快速实现可能不需要sentencepiece，但需真权重/本地完整快tokenizer文件验证后才能确认。原生Torch LoRA/GP实现可用于工程预飞；正式官方库一致性对照仍需要经授权安装/锁定依赖。

检查默认HF缓存及配置的HF缓存路径，只发现Qwen1.5-0.5B-Chat，revision `4d14e384a4b037942bb3f3016665157c8bcb70ea`，单权重1,239,173,352 bytes。没有T5/Qwen3缓存。本结论覆盖已检查路径，不声称搜索了整台机器所有目录。预飞时可用磁盘约202 GiB（并行研究运行会变化）。24GB内存足以容纳静态权重不等于LoRA+全批MLL能安全训练L1005。

## 资源申请清单与阶段门槛

建议优先申请 **只下载T5-base一套指定revision文件（约0.9 GB）**，不同时下载Qwen3。若后续需要Qwen3，新增约1.2 GB。运行时float32权重内存下界分别约0.9GB（完整T5，encoder实际更少）和2.38GB（Qwen3）；还要另计激活、tokenizer、梯度、优化器及GP。rsLoRA不会为全部base参数维护Adam状态，但反向激活仍可成为主成本。当前本机CPU无额外付费成本；实际墙钟时长尚不可测，不能承诺。正式按官方PEFT/GPyTorch/BoTorch复现须另行申请并锁定兼容版本；sentencepiece仅在快tokenizer不足时申请。

恢复时先进行4条人工文本、3步的权重预飞，再测固定L333状态1步的时间/峰值RSS。预先设资源上限：CPU单线程、L333初测60秒/步和峰值RSS12GiB；超过即停止扩展并报告。L333通过后再计划157、887两个seed的L333/L813，四种T5变体、多个初始化、V1/V2分开拟合。只有Phase1训练门槛及合法内部选择协议冻结后才运行真实Pilot。扩大到五seed×四预算必须另有固定比较协议与Pilot继续证据；本轮不自动闭环AL。

工程预飞的euclidean距离/learned constant mean/ddof1，与Phase1的RMS距离/固定constant mean/ddof0并不相同，noise/jitter设置也不能默认一致。正式Pilot必须先统一这些数学设置，并明确是否复刻官方ScaleToBounds，再配对比较T5、Qwen、GNN及四种适配结构；当前预飞配置不能直接升级为公平性能实验。Phase1若仍出现noise下界、MLL改善但外样本NLPD恶化，只能继续实现检查，不能据此开始正式概率性能升级。

GP单个float64密集矩阵大小仅为8*N² bytes（L333约0.85MiB，L1005约7.71MiB），但求解、autograd及transformer缓存会持有多个张量。逐chunk forward后拼接保留全部图，**不等于**激活只占一个chunk。

如保留全图超过内存，可设计保持Exact MLL目标的两遍链式求导：先确定性提取全部z，令其为leaf求完整MLL及dL/dz；再按chunk重算编码器，用对应dL/dz做向量雅可比积，累计LoRA/Projection梯度，全部chunk结束后只step一次。必须保证同一参数、同一dropout mask/随机状态或固定dropout关闭，否则不是同一函数的梯度。该方案目前只设计，未实现/验证；它不是小批MSE或稀疏GP。若改稀疏GP，须另列数学目标和新对照，不能继续称Exact GP。

## 恢复命令（只读取已获授权准备好的本地文件）

```bash
export PYTHON_BIN=/Users/fish/miniforge3/envs/chromatography/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/phase2/preflight.py --model-id google-t5/t5-base --kind t5 --revision a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1 --steps 3 --output studies/active_learning/gollum_adaptation_fusion_20261106/phase2/t5_authorized_preflight.json
"$PYTHON_BIN" studies/active_learning/gollum_adaptation_fusion_20261106/phase2/preflight.py --model-id Qwen/Qwen3-Embedding-0.6B --kind qwen3_embedding --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 --steps 3 --output studies/active_learning/gollum_adaptation_fusion_20261106/phase2/qwen3_authorized_preflight.json
```

也可把 `--model-id` 换成用户指定的完整本地模型目录；加载设置始终 `local_files_only=True`、`trust_remote_code=False`。上述命令是人工标签工程预飞，不是色谱性能Pilot。真实Pilot runner与数据权限需承接总PROTOCOL及Phase1结果，当前未宣称已完成。
