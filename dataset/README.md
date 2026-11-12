# 基础输入与数据权限

本目录保留原始受跟踪CSV及化合物目录，原数据字节未因Framework V2裁剪而改变。对应历史实验的完整来源位于固定 [archive快照](../configs/archive_manifest.json)，当前最小checkpoint/scaler/cache/partition资源见 [保护登记](../configs/protected_artifacts.json)。

原始数据覆盖4g/8g/25g/40g及C18/NH2/CN/DCM等柱/洗脱体系；文件本身存在不等于它仍是未观察的确认性测试数据。训练、验证选模、测试真值和结果读取历史以 [数据消费登记](../configs/data_consumption_register.json) 为准；本轮没有新读标签，不能据此重置既有消费状态。

- 当前模型特征/单位/编码见 [输入合同](../docs/model/INPUT_SCHEMA.md)。
- `src/qgeognn_al/predictor` 接收显式提供的数据与图/预处理资源，不会自动从历史实验目录寻找真值。
- 合成CLI不加载本目录。真实数据操作应先准备独立授权只读snapshot及SHA，候选特征不得包含U/test真实标签。
- `application/` 原发布脚本中的额外Excel、`dataset_save`或外部资源路径不构成现有完整生产runner的资格证明。

Git archive只恢复已跟踪字节。ignored/private checkpoint、原始运行工件、未提交请求或外部文件需要独立的受控备份；当前树和archive都不能替代这些备份。
