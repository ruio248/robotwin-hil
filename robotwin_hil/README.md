# HIL 扩展层

`robotwin_hil/` 不修改官方 `RoboTwin/` submodule。现有入口仍依赖 RoboTwin 内部导入路径，因此先用 `python3 -m robotwin_hil.prepare_runtime` 在 Git 忽略的 `outputs/robotwin_hil_runtime/RoboTwin/` 生成兼容运行目录。

| 功能 | 源码位置 | 生成后入口 |
| --- | --- | --- |
| 自定义 `handover_to_tray` 任务、恢复专家、阶段/安全门 | `overlay/envs/handover_to_tray.py`、`overlay/env_cfg/`、`overlay/description/` | `$ROBOTWIN_ROOT/envs/handover_to_tray.py` |
| 人工接管与 HG-DAgger | `overlay/scripts/hg_dagger_handover.py`、`overlay/scripts/one_stage_dagger.py` 等 | `robotwin_hil/local_serving/run_one_stage_dagger.sh` 或 `$ROBOTWIN_ROOT/scripts/hg_dagger_handover.py` |
| 候选采样、覆盖率、Gaussian / DART 消融 | `overlay/scripts/coverage_sampling/`、`overlay/scripts/coverage_benchmark/` 及相关入口 | `$ROBOTWIN_ROOT/scripts/` |
| 专家支撑函数 D 与尾部价值实验 | `overlay/scripts/coverage_benchmark/`、`overlay/scripts/tail_value/`、`overlay/scripts/train_d_*` | `$ROBOTWIN_ROOT/scripts/` |
| Pi0.5/OpenPI、XPolicyLab 适配 | `compat/XPolicyLab/policy/Pi_05_RobotTwin/`、`local_serving/` | `$ROBOTWIN_ROOT/XPolicyLab/` |
| 策略/任务测评 | `overlay/scripts/eval_*.py`、`eval/ubuntu_dump_eval/` | `$ROBOTWIN_ROOT/scripts/` 或按实验配置指定 `ROBOTWIN_EVAL_ROOT` |

`overlay/` 中与官方同名的文件（如 `envs/_base_task.py`、`scripts/collect_data.py`、`envs/dump_bin_bigbin.py`）是**本项目修改版**，仅在生成的 HIL 目录中生效。直接运行官方 `RoboTwin/` 不会启用这些改动。`compat/XPolicyLab/` 是历史适配源码快照，保留原许可证；不能把它和官方 submodule 内的 `XPolicyLab` 误认成同一版本。

具体运行协议与实验结果在仓库根目录 `docs/`，单次实验脚本在 `experiments/`。旧文档或机器脚本中的绝对路径应按当前 `$ROBOTWIN_ROOT`、checkpoint 与数据盘位置核验后使用。
