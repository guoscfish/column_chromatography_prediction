# 无付费 Preflight 报告

日期：2026-10-29。结论：离线工程预检 PASS；正式服务支持/价格仍待操作者确认及单独授权。未发送真实LLM请求，未训练QGeoGNN，未揭示新实验标签，未访问validation/test真值。不存在正式runtime、formal_authorization或served-model lock。

最终测试结果：**121 passed**，23.29秒。含新NoCW测试及旧 Scientist V2、feedback adapter、row protocol、full-pool、dialog bridge、LCMD、short sequential、LLM screen兼容性测试。118条依赖弃用/环境警告（含 requests 的字符检测依赖警告）；没有测试失败。没有运行包含真实/合成QGeoGNN训练的其他测试套件；训练接口一致性采用拦截fit函数的参数断言，而非启动模型拟合。

| 必需测试 | 结果与证据 |
|---|---|
| 1 输入差异 | 两seed真实冻结L333全量输入，各2997候选/333观测；ID、排列、SMILES、条件、描述符、预测和记忆精确相等。只删除2997个CW值和定义，packet hash对应变化；逐值差异见 `dry_run/seed_*/input_diff.json` |
| 2 NoCW泄漏 | raw catalog、encoded table、prompt、初始及第二轮嵌套记忆、实际request与纠错回合均扫描。测试CW别名/中文/梯度距离/coverage、未知numeric字段、候选真值、外来记忆注入均拒绝 |
| 3 选点合法性 | mock正常32选点通过；少选、重复、观测、未知/其他split、未测证据、自由文本ID证据、额外字段、CW叙事等均fail-closed，无补足/替换；纠错最多3答案；失败重启后总传输不增加 |
| 4 实验隔离 | 每seed两Arm共用同一checkpoint hash；mock选不同32，形成不同L365/U365；原lifecycle.state在合成receipt下重建各自状态并验证边界；拒绝跨Arm反馈。真实标签store在dry-run中被替换为一旦reveal就失败的哨兵，测试仍通过 |
| 5 历史兼容 | 没有修改任何已有源码/测试/runner或历史冻结结果；旧Free32、LCMD、CW16+LLM16相关测试通过；仅保留任务开始已有两个live-status修改及既有未跟踪诊断目录。见compatibility_audit.json |
| 6 完整dry-run | 每seed每Arm两轮：真实L333全量输入→mock JSON→验证→batch freeze→重放；下一轮使用合成测量-123/-122，绝无新真实标签。正式推进不能接受dry-run冻结；没有正式runtime |

额外审查发现并处理：method salt改变候选排列；DIRECT_PROMPT保留CW说明；encode_catalog字段并集会继续传CW；记忆中的reason/hypotheses会二次传播；旧transport在重启后重给重试额度；旧主report基于test而非validation。新实现均在隔离模块中处理，旧实现不改。

初始checkpoint：

| Seed | 安全复用路径（仓库相对） | SHA256 |
|---|---|---|
|157|`studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_157/lcmd/round_00/model/best.pt`|`87e92ccdabc1dd64ecf89a3061624450ce125906c4cbb988d7263b190991b1c0`|
|6101|`studies/active_learning/qgeognn_v2_row_sequential_b32/runtime/seed_6101/lcmd/round_00/model/best.pt`|`bcfffa247c3d24c8c74993ddb66cb59741ae14b1e67e8ed9543a8d1af7c6c494`|

原始catalog来自旧Free的round0，但只复用与这些checkpoint匹配的候选预测/初始观测，没有加载其LLM答案或后续反馈。L365及更高预算的历史模型不能用于本实验主轨迹。

实际执行的安全命令：

```bash
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python -m pytest -q \
  tests/active_learning_v2/test_nocw_ablation.py \
  tests/active_learning_v2/test_scientist_v2.py \
  tests/active_learning_v2/test_feedback_ablation_adapter.py \
  tests/active_learning_v2/test_row_protocol.py \
  tests/active_learning_v2/test_full_pool_selector.py \
  tests/active_learning_v2/test_dialog_bridge.py \
  tests/active_learning_v2/test_lcmd.py \
  tests/active_learning_v2/test_short_sequential_b32.py \
  tests/active_learning_v2/test_llm_screen.py --disable-warnings
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_nocw.py prepare
KMP_DUPLICATE_LIB_OK=TRUE /Users/fish/miniforge3/envs/fish/bin/python scripts/studies/run_qgeognn_v2_row_llm_nocw.py dry-run
```

没有执行 `preregister/select/advance`。模型/endpoint/effort/输出预算是拟定并冻结的配置，不能声称已被远端接受；正式运行前依README确认支持并单独授权。provider模型别名不能保证内部权重永远相同，served model string锁定只能检测可见版本变化。temperature/top_p未传，随机性和长度/注意力效应仍存在；两个开发seed只提供开发证据。自然语言语义/外部先验无法完全用regex证明无泄漏，但新NoCW轨迹没有被系统提供任何预计算CW信息。

实际L333序列化messages的content长度（含prompt）：seed157 aware=897,258、NoCW=873,522字符；seed6101 aware=902,638、NoCW=878,833字符。这不是token数。正式运行前还需确认服务上下文容量能够容纳完整输入、后续科学记忆及输出预算；不得以shortlist或截断来绕过容量不足。删CW自然改变输入长度，不能把注意力/长度效应完全从数值信息效应中分离。

技术评审：该设计足以作为信息接口消融，值得在确认服务支持后投入两seed成对实验成本。主要结论应来自新Arm的配对差，历史五seed仅作参考。建议按协议交错调用两个Arm，完整跑到1005，之后再用未参与开发seed确认；不需先增加更多变体。若预算有限，应减少确认范围而不是按L525胜负提前淘汰。
