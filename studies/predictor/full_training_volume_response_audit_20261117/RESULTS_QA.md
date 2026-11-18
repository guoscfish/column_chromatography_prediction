# 充分训练体积响应审计：独立结果QA

**结论：通过，保留已披露的标签边界事故例外。** 仅使用本轮已有合法派生差分和无truth原始预测，未读绝对target、旧正式predictions.gz或调用label接口；0新训练、0推理、0API。不能把本轮全过程描述为从未看到test数值。初期文本预览的3ID在所有seed隔离，本QA逐ID核对无例外。

正式评分覆盖82056个模型×pair×端点评分事件，同pair比较覆盖33296个formal×AL×pair×endpoint×method事件。每个比较单元的Fully trained、AL L525/L1005/L1485使用完全相同ID与同一合法delta_true；四方法没有通过缺失pair改变分母。三个正式模型的原训练身份仍3330行，本次预测train为3327/3328/3328、validation各416；全部source-test和3个quarantine在评分中为0。

## 独立数值重算

不导入原common.metrics、summary或评分函数，直接从delta_true/delta_pred重算绝对真实/预测响应、差分MAE、幅度误差、方向/非平坦正确/非平坦反向/平坦率以及幅度比和零响应基线误差改善。真实有意义变化严格>0.1/0.5/1mL，预测平坦严格<0.1mL；主MAE使用全部pair。正式per-model、三个模型等权均值及sample std、matched逐formal×AL及全三formal非test子集汇总全部一致，容差2e−10。模型平均的幅度比与幅度均值之比均独立检查，未把两口径混用。

正式raw q50预测重算pair差分最大误差1.42e-14mL；matched raw q50最大误差1.42e-14mL。AL原实现对内存float32预测相减，重读CSV后转回float32可精确复现；若全用float64相减最大偏差为8.17871e-06mL，属于舍入边界。正式模型按CSV float64相减。已有合法delta_true在正式表、matched表及交叉模型相同pair间一致。pair身份、只变一因素、PE/EA按EA比例上升与V按µL上升方向均从metadata独立核对。

整体Row NRMSE修复已核：从每端点RMSE/冻结尺度独立重算`sqrt(mean(normalized_RMSE²))`，不是算术均值；6个seed×role汇总均一致。完整416行validation的21项历史aggregate对照均通过1e−6。这里没有重新读取绝对标签，因此不把该部分称为独立逐行重算Row误差。

## 曲线pool与支持域

15个预固定case按metadata与合法共同pair的hash排名完整重建，每AL seed取3个，ID/base/系列/全域局部体积min–max均一致；900曲线行严格覆盖三个正式模型及对应三预算AL的两个端点、五个水平。局部支持域内648行，局部外推252行，标记逐点正确；30–150µL均在全源X域范围内，但这不表示每个分子系列都支持该区间。

曲线和原始合法预测在相同真实输入水平的120个重合标量独立交叉检查，最大差1.13359e-05mL，低于5e−4浮点图重建容差。曲线表不含真实误差列，真实overlay只来自预固定observed IDs且无test/quarantine。支持域min–max含全源无标签X，区间内新水平是插值敏感性，不等于每一级都有真实测量；外推点不能用于真实恢复结论。

pool的metadata排序不依赖预测或真实值；冻结plan早于pool，源码在曲线预测前写pool。此QA可验证排序与文件证据，不能把本地时间戳当作外部签名的预注册证明。若按全30–150µL span选恢复/平坦/seed差异示例，必须说明可能包含局部外推，只能作模型灵敏度描述。

## 复现与来源

运行`qa_results.py`即可从冻结派生表重建本MD/JSON，不进行模型加载或forward。所有实际读取的派生表、预测、manifest及相关源码SHA在RESULTS_QA.json中，结束前再次验证稳定。局部pair并非独立科研重复，不给伪独立显著性结果；三个正式模型与五个AL seed的交叉比较不称配对随机种子实验。
