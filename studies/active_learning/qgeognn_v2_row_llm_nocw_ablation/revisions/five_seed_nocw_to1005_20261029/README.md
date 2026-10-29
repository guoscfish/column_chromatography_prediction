# Five parallel NoCW workers

本次仅运行NoCW，seeds=157/887/2357/6101/12203，每seed独立key，统一gpt-6.1-sol/xhigh。终点1005，共105次主采集、105次新拟合；不新跑CW-aware。历史五seed Free已到1485，作为参考结果保留。

运行入口：`scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py`。所有命令在仓库根目录使用fish环境：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
PY=/Users/fish/miniforge3/envs/fish/bin/python
SCRIPT=scripts/studies/run_qgeognn_v2_row_llm_nocw_five.py
$PY "$SCRIPT" status
$PY "$SCRIPT" validate
# 首次启动；本目录formal_authorization.json绑定用户本次明确授权。
$PY "$SCRIPT" launch --allow-paid-llm --allow-training --allow-label-reveal
# 仅在检查失败原因后，显式恢复单seed；绝不重置失败请求/纠错预算。
$PY "$SCRIPT" worker --seed 157 --allow-paid-llm --allow-training --allow-label-reveal
# 完整五seed到相应节点后，生成独立validation报告；不读test真值。
$PY "$SCRIPT" report --budget 1005
```

`launch.json`保存各seed进程、目标及计数；`progress/seed_*.json`和日志给出状态，`selections/`和`runtime/`保存冻结请求、批次、反馈、模型。默认入口仅status，重复launch拒绝，不无限重启。若进程失败，先读结构化attempt记录并核对不确定账单，不能删除started标记后盲目重发。

历史Free的157/6101含Astra/high前缀，887/2357/12203全程gpt-6.1-sol/xhigh。不得把五seed汇总直接称为全配置一致的新配对因果消融。可分别描述三个同模型seed与两个混合前缀seed；这些都不是未参与开发的新确认seed。
