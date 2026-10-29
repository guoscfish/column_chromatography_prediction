# 配对 Free LLM32 CW 数值信息消融协议 v1

冻结日期：2026-10-29。状态：代码/离线协议冻结，尚未授权正式运行，尚未验证远端服务支持。`protocol.lock.json` 绑定 config、源码、测试、本文、SCHEMA_DIFF、CODE_AUDIT、初始数据文件和环境；`protocol.json` 为生命周期兼容的相同副本。任何变更需要新的显式协议版本，不能覆盖已冻结记录。运行前还必须由操作者完成 `preregister`，确认供应商支持模型/effort/输出限制并授权正式实验。本交付不创建该授权文件。

研究问题：Does numerical gradient-coverage information improve the decision quality and label efficiency of an LLM-driven active learning agent?

Arm A `llm32_cwaware_control`（LLM32-CWAware-Control）和 Arm B `llm32_nocw`（LLM32-NoCW）均自由选择32，pending0，完整候选池。唯一预注册的信息差异见 SCHEMA_DIFF：CW列和对应定义；packet hash 随之变化。Arm/variant 名称只在 host contract，避免方法名对 LLM 产生额外提示。两者共同从旧 prompt 中删除 hybrid 描述，并要求实测证据只能写入 typed observed-ID lists；这些共同规则不用于历史结果。

Seeds=157、6101，是已参与开发的两个 seed。相同 Row split、初始333观测、2,997候选和对应 QGeoGNN-V2 member0 checkpoint，从 L333 空记忆分叉。仅复用各 seed 的 L333模型、量化预测和初始观测，不继承历史 Astra/high 或 Sol/xhigh 的任何后续选点和记忆。后续模型/标签集合/候选池/反馈各自独立。每轮排序为 `sha256([version:seed:paired:round, opaque_id])`；不因 method、CW值或字段数改变，原 opaque IDs 两边一致，无新映射。

预算从333开始，每批32，至1005共21轮。525为第6批后的工程核验点；653为第10批、813为第15批、1005为第21批。除传输/协议错误、无法维持数据边界、人工明确取消外，不因早期效果差而停某 Arm；不以 L525 结果自动淘汰。禁止自动扩 seed 或超1005。未来有积极信号才考虑新的未参与开发 seed。

主要终点是 validation AULC333–1005：每个预算的 V1/V2 RMSE 分别除以冻结 L333 target scale，取平方平均后开方；沿标签预算做梯形积分再除672。主要配对差 `AULC(NoCW)-AULC(CWAware)`，正数表示 CW-aware 更好。报告每 seed 差、两 seed 均值及方向，不做两 seed 显著性结论。次要指标：validation NRMSE@525/653/813/1005、V1/V2 RMSE、MAE、R²、采样结构/条件/新颖性/相似度分布、各调用 token usage/耗时。货币价格未验证，不能把未知账单记作零。L525报告是工程诊断，不替换1005主终点。

训练条件从 `TRAINING_CONFIG` 逐项冻结：QGeoGNN-V2、Adam lr=.001、weight_decay=0、batch2048、maximum_epochs1000、patience100、原损失和结构、deterministic shuffle、validation combined normalized RMSE 选 best checkpoint。每次从相同 scratch 初始化重新拟合；初始化 seed=`outer_seed+2000003`，训练 RNG=`outer_seed+3000017`，CPU线程2。归一化、target scales和图预处理仅复用 L333 context。两 Arm 共享初始状态，后续由于不同训练集而产生的模型差异是主动学习轨迹的处理效果。

传输完全来自同一 config，拟定 Responses endpoint=`https://token4research.cn/v1`，model=`gpt-6.1-sol`，reasoning=`xhigh`，tools=[]，store=false，SDK max_retries=0，timeout600秒，max_output_tokens32768（包含供应商计入该限制的全部输出），计划正文上限30,000字符。不传 temperature/top_p，使用同一供应商默认值，不能声称采样完全确定。每批至多3个答案（初次+2次同类格式纠错），每答案最多3次物理传输尝试；2秒、4秒 backoff。总次数跨重启累计；请求 started 而无完成记录时停止、人工核对账单，不自动重发。每次完整记录实际 messages/config/哈希、答案、失败和usage，不传工具或本地路径。首个 response.model 在两 Arm 间共同锁定；不接受模型变更或工具输出。

每批32个唯一 U_t ID、预测量预测、请求/目录/协议/checkpoint hash，全部在 label reveal 前冻结。禁止未知、重复、已观测、validation/test、另一 split ID；禁止在 hypotheses 的实测证据列表引用候选。所有自由文本的 row-ID 引用均拒绝，要求用 typed observed-ID lists，降低“候选被当作测量证据”的歧义。一般性、无ID的化学陈述无法自动验证全部语义，应保留审阅记录。禁止补点、替换、LCMD fallback、放宽输出数量或临时增加纠错次数。NoCW 输出或记忆出现禁用信号时 fail-closed；不静默改写叙事。纠错请求也经过序列化扫描，不能把含 CW 的无效答案送回 NoCW。

validation 仅供训练 checkpoint selection、完整配对 milestone 后的报告；不得写入 LLM包、反馈或科学记忆。全部新标签只经 RestrictedLabelStore 在冻结后显式授权的 advance 访问。报告无需 reveal test。模型更新反馈沿用原接口：历史真实测量、采集前误差和新轮候选的当前 checkpoint 预测；没有向任何一边额外引入 validation reward 或 oracle-error 字段。

dry-run 使用真实 L333格式和完整池，mock JSON完成解析/验证/冻结/重放；第二轮仅使用明确标记的合成反馈（-123/-122哨兵），并沿用初始预测模拟输入结构，不代表真实更新模型。dry-run 目录独立，冻结标记永久禁止用于 advance；不会创建正式 runtime、读取新真值或拟合模型。

请求/拟合成本（不计可复用的 L333）如下：

| 终点 | 每 seed 每 Arm 主采集/新拟合 | 每 seed 两 Arm | 两 seed 两 Arm | 最多答案数 | 最多物理请求 |
|---|---:|---:|---:|---:|---:|
| L525 | 6 / 6 | 12 / 12 | 24 / 24 | 72 | 216 |
| L1005 | 21 / 21 | 42 / 42 | 84 / 84 | 252 | 756 |

最多数是全部成功走完轨迹的有限纠错/传输上界；实际错误可以提前 fail-closed。达到 L1005 相比 L525 额外60个主采集和60次拟合（全四条轨迹）。后续 stage 包含推理/梯度抽取计算，需单独显式 flag；不等于一次新拟合。

技术判断：足以测试“提供预计算 CW 数值信息”的增益，不能归因为纯化学知识或完全无数字的LLM。输入长度差异、随机采样、模型服务漂移、两开发 seed、保留信息与 CW 的相关性，以及轨迹分叉都要写入最终分析。低复杂度改进：交错两 Arm 的调用顺序（seed157先A、seed6101先B，下一轮反转），尽量同一服务版本和时间窗口；保留预注册的成对分析。若未来增加同状态决策诊断，应另行预注册，不混入本轮标签效率主结果。确认阶段使用未参与开发 seed。
