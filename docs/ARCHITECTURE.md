# 架构入口

当前候选架构见 [NEW_ARCHITECTURE.md](repository/phase2/NEW_ARCHITECTURE.md)，实测与局限见 [TEST_REPORT.md](repository/phase2/TEST_REPORT.md)。稳定源码仅维护 `predictor/`、`al/`、`infra/` 三个子包与一套配置驱动回合入口。

第一期原型的完整架构记录保留在不可变提交 `4ee480378b0293f1985e3ee4a09df2c78f77901e:docs/ARCHITECTURE.md`，可用 `git show` 查看。该记录中的并行旧目录与待迁移状态不再代表本分支当前布局。
