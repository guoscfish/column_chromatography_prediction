# 冻结研究协议

2026-11-06，首次新增训练前冻结。旧研究只读，新结果均为开发性研究，不启动闭环AL。

## 数据与信息边界

只复用同一 corrected LCMD32 历史轨迹，首轮固定 seed157/887 × L333/L813。原缓存每状态是有序 L_t、SHA256选择的256个U_t、416行开发validation。逐项复核checkpoint、模型状态、L/U/validation有序身份、split、源CSV、特征缓存和Qwen模型/文本缓存SHA256；不读取U响应，不数值化test标签。来源中的历史 `fits.json.seed` 有覆盖问题，配对以目录/状态回执为准。

主要机制消融使用按 canonical SMILES 的无标签 SHA256 划分（salt `gollum_gp_split_v1:seed:smiles`），训练60%、校准20%、核验20%，全部取自当时已揭示 L_t，同一分子跨条件只能位于一个子集。三个子集不随方法/初始化变化；不依据响应或效果重采样。它改变了GP训练样本数，不能冒称全部L333/L813已用于GP拟合。Frozen Qwen未使用这些标签训练，内部核验可用于其GP的条件性外样本诊断。GNN及Fusion特征的历史checkpoint已经见过整个L_t：其内部核验仅是**冻结表示下的GP留出**，不是GNN端到端无泄漏核验。validation虽未拟合GNN权重，参与过checkpoint选型，始终标为开发证据。没有独立确认集；若以上数据不足以给出可信端到端结论，明确No-Go，不以内部拆分掩盖限制。

MLL仅接收fit标签，PCA、特征/目标标准化只在fit子集拟合。校准仅接收calibration标签，独立核验只评分，不向训练返回。Frozen baselines按同fit子集重拟合。另加seed157的L333/L813、两个表示、V1/init17的全L_t机制复跑，核对旧A40及预算200的行为；仅为数值机制证据。

## 预先限定的训练矩阵

主消融为 Qwen1024→64、旧式Early Fusion(PCA64×2→concat128)→64，均ELU/Matérn5/2/float64 exact MLL。每状态/端点/方法保留init17、29，不挑最好一次。

- A Original：L-BFGS-B预算40，其他设置与旧实现一致。
- B Converged-budget：仅将预算增加为200（名称表示尝试充分训练，不能以预算200宣称已收敛），ftol1e-9、gtol1e-5、maxls20、maxcor10。保留真实停止状态及有界投影梯度。
- C Regularized：B加固定Gaussian权重先验，负对数惩罚为 `0.5 * 0.01 * input_dimension * (||W||²+||b||²)`。
- D Noise-floor：B仅将标准化noise variance下界从exp(-9)提高至0.02。
- E Staged：B后冻结Projection，以相同MLL再优化核signal/noise/lengthscale至多100步，不重新标准化表示。

不同时搜索优化器/核/容量/惩罚强度。0.01和0.02是本次预先固定的一点敏感性消融，不是已选最优值；本轮不根据416开发validation调整。后续若选择强度，只能在新的合法训练内部验证，并重新冻结协议。训练保留全部失败和未收敛初始化。每个fit完成后落盘，单CPU线程；主运行墙钟上限2400秒，若到上限按固定遍历顺序截停并写原因，不能按效果删配置。

独立校准：固定使用calibration上Gaussian observation variance temperature MLE，`alpha=mean((y-mu)^2/v_obs)`，数值截断[1e-6,1e6]。报告alpha及核验NLPD、覆盖、宽度，不能单靠覆盖宣称改善。它是观测预测区间后处理，不把缩放后的观测方差冒充一致GP后验，也不据此重写IVR/IG。

Frozen Qwen/GNN/Structured/Raw/CW Matérn GP预算200作为同拆分基线；旧完整L_t结果单独报告。主要点预测指标为两个端点RMSE/历史L0标准差的平均；主要概率指标为原单位NLPD。90/95覆盖、宽度、标准化残差RMS、误差与方差相关、MLL、noise/signal、条件数、梯度、迭代与成本为共同次指标。不选择某个预算作为事后主指标。

## 采集与鲁棒性

原始未校准GP在完全相同256个合法候选上计算MaxVar、conditional IVR、greedy BatchIG，batch32。记录Top32、方差排序、批内绝对相关、分子/条件覆盖和耗时。模型只用模板A训练，模板B仅在核验和候选前向，比较均值、方差秩和三种选点重叠；B不得用于调参。SMILES-only沿用旧冻结实验及Phase2模板预检，主Phase1不把它新增为调参维度。

五seed不是20独立重复；先按初始化/端点/预算聚合seed，再报告配对seed差和最差seed。首轮只有2个开发seed，不能做最终性能宣称。

## 阶段门槛

首先要求梯度、核/变换一致性、身份和权限测试全部通过。扩展5seed×4budget只考虑在本轮**原始GP**中达到至少90%优化success且有界投影梯度<1e-3的候选，同时两个seed的核验90%覆盖均≥80%、标准化RMS≤1.5、NLPD不劣于同拆分Frozen Qwen，且无实质数值失败。此为预筛门槛，不是独立概率资格。若无候选通过，保留负结果并停止扩展；单靠temperature校准或GNN训练内划分不能解除门槛。

Phase2权重/依赖缺失时只源码对齐及toy工程测试，后者不产生T5/LoRA性能结论；新下载/安装/付费需要用户授权。Phase3先用现有GNN/Qwen训练加性核对照，lambda=0/1/训练lambda、容量控制，全部只用训练MLL。GNN/LLM单位对角核，共用signal，LLM不再另设amplitude，lambda避免与语言amplitude重复参数化。Residual需合法OOF/冻结外样本误差及互补证据，Factorized本轮仅设计。

正式AL本次禁止自动启动。即便未来概率检查通过，仍需另获授权进行同L333、batch32、同初始化/协议的QGeoGNN学习曲线，检验AULC、跨seed和最差轨迹。
