# SFT 测评协议

## 新实验的主协议：官方控制逻辑，同轮保存 D 轨迹

新 SFT 正式测评**统一使用最小记录版官方入口** `scripts/eval_policy_xpolicylab_official_trace.py`，一次运行同时得到成功率与后续冻结 D 评价所需轨迹，不再为 D 重测。它保留固定版官方场景/指令/策略控制逻辑，仅新增被动记录；应称为“官方控制逻辑加被动记录”，不能称为完全未修改的官方脚本。历史 `policy_eval_record.py` 与旧综合扩展 `eval_policy_xpolicylab.py` 均不是新入口；后者测评逻辑已移除，同名文件只保留旧 HIL/DAgger 的导入兼容函数。运行前冻结 RoboTwin 提交、任务/指令文件、`demo_clean` 配置、策略服务加载的 checkpoint、训练配置与 `norm_stats`；`--ckpt_setting` 只是结果目录标识，不能证明服务实际加载的权重。

在 `prepare_runtime.py` 生成的隔离运行副本根目录、策略服务已启动后，`handover_block` 的单进程 **50 集接口示例**是：

```bash
python scripts/eval_policy_xpolicylab_official_trace.py \
  --task_name handover_block \
  --env_cfg_type aloha_agilex \
  --policy_name Pi_05_RobotTwin \
  --host 127.0.0.1 --port 18493 --protocol ws \
  --root_dir "$PWD" --device_id 0 \
  --task_config demo_clean \
  --seed 0 --test_num 50 --eval_batch false \
  --expert_check true --instruction_type seen \
  --frequency 50 \
  --trace_dir /absolute/path/to/new-trace-directory
```

示例中的端口、GPU 和 `frequency` 必须与实际服务/实验配置核对，不能把 `--frequency 50` 当作仿真物理频率或动作 horizon 的证明；H 由服务返回的 chunk 决定。官方脚本把 `--seed 0` 换算成从 scene seed 100000 起搜索，并用脚本专家检查筛选场景，直到有 50 个正式 episode；**不是固定取 100000–100049，也不是固定 31000–31049**。`seen`、专家预筛、任务配置、动作语义、策略服务及版本必须与待比较实验一致。这里的“官方入口对齐”指所运行版本的脚本与参数，不自动等于任何公开榜单的完整评测协议。

原版脚本没有逐查询 RGB/state/action 归档；此外，部分动作已经执行后若 rollout 抛异常，原版计数流程可能把该局当失败。最小记录版会保留官方计数，同时把该局单列为 `infra_error`、`success=null`，禁止混入成功/失败 D 分析。验收时必须检查 manifest、rollout 异常、有效场景和服务状态；出现基础设施异常时不能将该率当作干净的策略成功率。

记录结构、成功/失败/异常字段及验收门见[最小记录版说明](OFFICIAL_INTERFACE_TRACE.md)。它不改变固定版官方 `build_instruction`、scene seed 搜索、专家预筛、策略调用或成功检查；单进程模拟对照检查开关记录前后的 seed、指令、策略调用、动作及终局一致，但不是跨不同 RoboTwin 提交的逐随机数等价证明。只保存旧 `_result.txt`/视频的历史评测无法补算 D，也不能沿用旧终局标签。

## 历史整任务自主测评（非上述官方主协议）

旧实验曾使用 `robotwin_hil/overlay/scripts/policy_eval_record.py`（兼容运行目录中为 `scripts/policy_eval_record.py`）或任务对应的同等记录入口；它们只用于追溯原有结果，不是新 SFT 的默认测评入口。先核验服务进程实际加载的 checkpoint、`discrete_state_input`、`norm_stats`、提示词、相机顺序、动作维度与输出的绝对关节语义。`handover_to_tray_v2_promptfix` 历史基线采用完整 `H=10` chunk、`frequency=30` 配置、同一批冻结 seed、无专家预筛、无人工介入；该字段不能单独证明物理循环恰为 30 Hz。不同任务的 step limit 与仿真配置必须单独记录。

一个 seed 对应一条正式记录。先单 seed smoke，确认可启动、动作合法、终局检查可读，但 smoke 不进入分母。正式运行时每条保存 seed、成功标记、步数、终局 `check_success()`、任务几何指标和异常。成功率为 `成功的有效正式 rollout / 原定正式 seed 数`；初始化/服务/仿真错误单列为 infra error，不当作策略失败，也不能悄悄剔除后仍用原分母。验证唯一 seed、无缺失，再汇总。

对比两个 SFT checkpoint 应使用同任务、同几何/仿真补丁、同 seed、同提示词、同状态开关及动作执行长度；逐 seed 配对报告两边成功/失败变化。若协议不同，应分表展示，不能把发布方专家筛选场景的成功率和本项目未筛场景直接作差。

## 与 HG-DAgger 末段测评的区别

整任务自主测评从初始化开始让策略控制；[末段测评](../hg_dagger/HG_DAGGER_EVALUATION.md) 则先由专家完成前五阶段，只考察合格的第六阶段。两者分母、任务难度及因果问题都不同。SFT 测评不使用 `i/r/x` 人工接管、不启用增强采样；若改用 `eval_one_stage_policy.py`，必须明确标注为**条件末段自主成功率**。
