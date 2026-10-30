# SFT 测评协议

## 整任务自主测评

使用 `robotwin_hil/overlay/scripts/policy_eval_record.py`（兼容运行目录中为 `scripts/policy_eval_record.py`）或任务对应的同等记录入口。先核验服务进程实际加载的 checkpoint、`discrete_state_input`、`norm_stats`、提示词、相机顺序、动作维度与输出的绝对关节语义。`handover_to_tray_v2_promptfix` 历史基线采用完整 `H=10` chunk、`frequency=30` 配置、同一批冻结 seed、无专家预筛、无人工介入；该字段不能单独证明物理循环恰为 30 Hz。不同任务的 step limit 与仿真配置必须单独记录。

一个 seed 对应一条正式记录。先单 seed smoke，确认可启动、动作合法、终局检查可读，但 smoke 不进入分母。正式运行时每条保存 seed、成功标记、步数、终局 `check_success()`、任务几何指标和异常。成功率为 `成功的有效正式 rollout / 原定正式 seed 数`；初始化/服务/仿真错误单列为 infra error，不当作策略失败，也不能悄悄剔除后仍用原分母。验证唯一 seed、无缺失，再汇总。

对比两个 SFT checkpoint 应使用同任务、同几何/仿真补丁、同 seed、同提示词、同状态开关及动作执行长度；逐 seed 配对报告两边成功/失败变化。若协议不同，应分表展示，不能把发布方专家筛选场景的成功率和本项目未筛场景直接作差。

## 与 HG-DAgger 末段测评的区别

整任务自主测评从初始化开始让策略控制；[末段测评](../hg_dagger/HG_DAGGER_EVALUATION.md) 则先由专家完成前五阶段，只考察合格的第六阶段。两者分母、任务难度及因果问题都不同。SFT 测评不使用 `i/r/x` 人工接管、不启用增强采样；若改用 `eval_one_stage_policy.py`，必须明确标注为**条件末段自主成功率**。
