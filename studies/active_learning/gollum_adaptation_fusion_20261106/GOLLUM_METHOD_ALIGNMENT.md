# GOLLuM 方法对齐审查

审查日期：2026-11-06。本次是源码核对与实现准备；没有复现论文的科学性能。

官方仓库通过 `git ls-remote` 核验，HEAD/main 为 **`c418d7ed3c17e5995f503f6df7c26e9f7c58d09f`**（短 SHA 页面：[c418d7e](https://github.com/schwallergroup/gollum/commit/c418d7e)）。公开源码/配置共 13 个文件、约 54 KB，已按该 SHA 保存到 `phase2/upstream/`；`SOURCE_MANIFEST.json` 记录每个 URL、字节数和 SHA256。未下载官方数据或任何模型权重。网页抓取会折叠空行，因此以下行号以保存的原始源码为准。

## 方法身份

论文 [Nature Machine Intelligence 原文](https://www.nature.com/articles/s42256-026-01283-z) 的目标是通过 GP marginal likelihood 学习语言表示，区分三种变体。Frozen LLM+GP 是参照；P-LLM 学习固定表示上的投影；LLM-phi 通过 LoRA 更新编码器；P-LLM-phi 同时更新 LoRA 和投影。它们不等同于让语言模型直接生成实验 ID。本项目过去的 Frozen Qwen1.5 hidden-state+Projection 仅对应 P-LLM 类型的探索，不能代替 T5 或 LoRA 的结论。

| 项目 | 核实的官方实现 | 本次实现与差异 |
|---|---|---|
| T5 骨干 | `T5EncoderModel`，读取 t5-base，禁用 base dropout；[text.py](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/src/gollum/featurization/text.py) L88–122 | `LocalTextEncoder` 保留该设置，只允许本地完整 safetensors；CPU float32 forward |
| Pooling | attention-masked token mean；padding 先置零再求和/除有效长度；[pooling.py](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/src/gollum/featurization/utils/pooling.py) L3–7 | 相同公式、保留 float32 pooling 再转 CPU float64；不使用 decoder、不生成文本 |
| 文本输入 | 最长512 tokens，官方代码允许截断；[text.py](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/src/gollum/featurization/text.py) L137–158 | 上限512，超限直接失败，避免静默丢失条件；这是有意的安全差异 |
| Projection | Linear+ELU，Xavier uniform，bias=0.01；定义的 Dropout 没有进入 forward；[deep.py](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/src/gollum/featurization/deep.py) L52–68 | 同公式和初始化；真正 T5 对照预设维度64，toy 测试仅8维 |
| LoRA | rank4、alpha16、dropout0.2、bias none、`use_rslora=True`；[deep.py](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/src/gollum/featurization/deep.py) L71–136 | 原生 Torch rsLoRA，缩放 alpha/sqrt(rank)，A 随机、B=0；未安装 PEFT；toy 为确定性梯度核验将 dropout 固定0 |
| LoRA 位置 | 按 block/layers 编号选顶部 round(25%) 层的全部 Linear；[layers.py](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/src/gollum/featurization/utils/layers.py) | 同样按层编号选全部 Linear；T5-base 12层对应9、10、11层，实际名字及参数数须加载后落盘 |
| GP | ScaleKernel(Matérn nu=2.5)，isotropic，ConstantMean，GaussianLikelihood；[pllm.yaml](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/configs/pllm.yaml) | 原生 Torch full-batch Exact MLL 数值预飞，float64 Cholesky；尚非经 Phase1 科学门槛验证的概率训练器 |
| 表示预处理 | 三份配置都是 `normalize_input: original`、`normalize_embeddings:false`、`standardize:true`、`normalize:false`、`scale_embeddings:true`；[llm_phi.yaml](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/configs/llm_phi.yaml) | 已实现训练标签 sample-std 缩放。预飞未复刻 `ScaleToBounds(-1,1)`；不能声称逐位复现官方训练 |
| 训练输入范围 | closure 每次用完整 train_x 构造 ExactMarginalLogLikelihood；[gp.py](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/src/gollum/surrogate_models/gp.py) L235–282 | 微分图保留全部行；transformer chunking 后拼回 NxD，GP 每步一个 NxN 协方差；没有小批量 MSE 替代 |
| 初始化维度 | `train.py` L138–161 根据实际模型重写 input_dim，T5=768；[train.py](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/train.py) | 从 config 读取 hidden_size，避免仅看 deep.py 默认1024/3584而误报维度 Bug |

## 训练轮数与不能照抄的实现细节

三份配置 `n_iters:50` 是 `train.py` L207–213 的 **BO采集轮次**，不是每次 GP/LoRA 拟合50 epochs。DeepGP 使用 AdamW（表示学习率0.002，GP配置0.2，默认两组 weight decay 都0.001）、StepLR 每步乘0.95；`train_mll_additionally:false`，不执行冻结表示后的额外 SciPy 拟合。[P-LLM-phi 配置](https://github.com/schwallergroup/gollum/blob/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f/configs/pllm_phi.yaml)

实际内层步数和停止交给 `fit_gpytorch_mll_torch`，官方没有显式 step_limit 或停止容差。当前 [BoTorch 源码](https://botorch.readthedocs.io/en/stable/_modules/botorch/optim/fit.html) 默认使用 ExpMAStoppingCriterion；官方 requirements/environment 未锁 BoTorch/GPyTorch 版本，因此不能据当前默认反推论文每次真实训练步数。缺少原运行日志时，这一项必须标记“无法确定”。本地预飞明确3步截断，不报告收敛。

`gp.py` L269–270 的梯度裁剪只出现在训练 closure 执行前，不能据此认为每次 backward 后都裁剪。本次把裁剪放在每次 backward 后、step 前。源码初始化字典使用 noise/lengthscale/outputscale 名称，却按 named_parameters 的名称过滤；GPyTorch 通常暴露 raw_* 名称，因此初始化可能被跳过，这是静态兼容性疑点，未安装 GPyTorch 做动态复核，不当作本项目旧训练器的原因。官方多处硬编码 CUDA，不能在当前 CPU 环境直接运行。普通 GP 和 DeepGP 的 predict 默认 observation_noise 分别 false/true，比较前必须显式统一。

## 本次证据边界

`lora_adaptation.ExactMLLHead` 目前采用euclidean距离、可学习常数mean、noise下界加exp参数化、独立固定jitter；`fit_exact_mll` 用训练标签sample std（ddof=1）。Phase1主实验采用RMS距离、固定constant mean及ddof=0等设置。因此当前代码是**训练管线工程预飞**，不能原样拿来与Phase1/旧Qwen做公平的T5性能比较。正式Pilot前必须冻结共同的距离定义、均值处理、noise/jitter、目标缩放、优化器以及是否采用官方ScaleToBounds；再把四种T5结构和Qwen/GNN强基线接入同一数学设置。

已通过 pooling、全批 Exact MLL、有限差分、重复输入 PSD/Cholesky、LoRA/Projection/GP 梯度、冻结参数、文本特征边界、身份哈希和离线标志测试。T5 与 Qwen3 真权重均不在已检查缓存，离线加载实际失败。原生 rsLoRA 工程测试通过不代表 PEFT 跨版本数值一致，也不代表 T5 的化学知识或概率质量已验证。Frozen T5、P-T5、T5-phi、P-T5-phi 的真实配对 Pilot、模板鲁棒性及下游 AULC 均未执行。
