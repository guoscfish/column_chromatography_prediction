# E2 Random active-learning chain smoke

状态：**通过**。本目录只验证E2工程闭环，不用于判断Random或主动策略的科学优劣。

- 固定D28 `e2_4g_row_seed_42` partition；L0=375，其中validation=57，test=417。
- Random查询2轮，每轮25条；K=2成员每轮从seeded random source-free初始化重新训练。
- 不加载训练过完整4g标签的checkpoint，避免L0/U0/test标签泄漏；这是一条E2专用source-free协议，不修改Gate 0冻结的4g→8g `last2+head`迁移合同。
- 输入scaler只用固定L0-train拟合一次，随后跨轮次与策略冻结；validation、U0和test不参与拟合。
- test身份每轮固定，checkpoint与test prediction均发生变化，round state可按split/config hash恢复。
- `queried_slice_diagnostics.csv`含预注册的selected总数、tail、compound、批内表示距离、uncertainty与reveal后真实误差。

限制：当前4g canonical数据在历史reader阶段已经应用60/120 mL阈值，因此本smoke的tail计数结构性为0；tail acquisition机制只能在后续保留完整574行的8g E4中正式解释。
