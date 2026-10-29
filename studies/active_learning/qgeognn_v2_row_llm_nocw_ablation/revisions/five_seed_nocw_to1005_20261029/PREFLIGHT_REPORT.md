# Five-seed execution preflight

2026-10-29：130 tests passed，29.56秒；新五seed调度/密钥隔离/锁测试与原121项兼容性测试全部通过。未执行任何测试模型训练或真实API请求。

五seed各2997候选、333初始观测；完整实际NoCW编码与对应历史Free删除CW列后的编码逐值精确一致，候选排序和精度均相同。mock完成32选点、freeze和审计重放，所有NoCW序列化扫描通过。`dry_run_audit.json`记录各seed checkpoint hash和结果；dry-run真实调用/拟合/新标签均0。

五份本地key非空且互不相同，只记录文件路径，不显示密钥值。每seed独立长锁，不同seed可并行；共享模型版本标识仅短暂原子锁。实际worker单元测试只调用llm32_nocw，CW-aware调用/拟合为0。首次正式完整请求将验证服务是否接受模型、effort及上下文，不额外发付费探针，不回退或截断。

原两seed协议仍独立validate通过，hash为1de45e5277b4e869c399178acc0218a4b2e0b832206c843ee90e56ad8f7fdeae。未修改旧src、旧冻结数据或任务前已有工作。

历史参考核对：`reports/free_llm_five_seeds_to1485_20261024/figures/current_summary.json`确认五seed均1485；157/6101保留Astra/high早期轮次，887/2357/12203全程gpt-6.1-sol/xhigh。新的五NoCW全部sol/xhigh。该历史模型差异必须随结果报告，不伪称完全配对。
