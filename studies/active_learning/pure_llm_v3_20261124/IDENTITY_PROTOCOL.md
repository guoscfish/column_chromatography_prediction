# 样本身份及请求绑定

在同一 seed 的固定 L0∪U0 上对底层样本 ID 做字典序排序，分配 0001–3330。该排序不读取采集价值。候选使用 C-，观测使用 O-；角色迁移时数字部分不变。独立分子编号使用固定池 canonical SMILES 排序，M- 全程稳定。长 ID 和短 ID 使用独立编码开关。已测角色、当前池、历史同底层实验、分子和试剂分别确定，过去选点不可再次当作候选。

真实请求标识是 seed、round、checkpoint SHA、split SHA、资料来源 SHA、packet digest、角色映射 digest、Prompt 版本与内容 SHA、schema 和实际发送 payload、配置及逻辑请求序号的绑定。传输落盘对每次物理尝试附加 attempt_id，并记录 provider response ID 及回复摘要。逻辑纠错重新绑定请求和范围，最终 freeze 记录接受的响应身份、选择文件哈希、记忆文件哈希和底层样本 ID。恢复时重建并核对原请求，已完成结果从回执读取。历史哈希复制兼容模式仅用于旧 schema 对照。

不同 seed 的短编号可以相同。编号只在当前请求和冻结映射中有意义。未知 ID 原样诊断到可识别的安全坐标，宿主不根据编辑距离寻找替代点。自然语言中的科学身份绑定有剩余人工复核工作，见 [科学证据协议](SCIENTIFIC_EVIDENCE_PROTOCOL.md)。
