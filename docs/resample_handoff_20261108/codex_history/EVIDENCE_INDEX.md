# RESample 证据索引

| 结论 | 等级 | 位置 |
| --- | --- | --- |
| 会话主名是「实现 RESample 类采样覆盖算法」 | `VERIFIED` | `threads.name` = `01a0ba85-24c0-7e00-aa7a-0f7ef7eb4355`；续段 `01a192bd-e098-7193-ae26-bf39c364208e` |
| 低分权重大，vanilla 均匀，enhanced 用 beta | `VERIFIED` | `robotwin_hil/overlay/scripts/coverage_sampling/core.py` `coverage_weights`、`DecisionSampler.select` |
| full-chunk 判别器要求 H=10 | `VERIFIED` | 同文件 `SamplingConfig.__post_init__` |
| Gaussian 模式不加夹爪噪声 | `VERIFIED` | `_isotropic_gaussian_noise` |
| TD chunk loss 的 y 与保守项 | `VERIFIED` | `robotwin_hil/overlay/scripts/tail_value/chunk.py` `chunk_critic_loss` |
| 四任务拒绝非 off 的 es-mode | `VERIFIED` | `dagger/runner.py` 约 459 行；`dagger/adapters/live_object.py`；`stack_blocks_three.py` |
| H50 sigmoid D 脚本存在 | `VERIFIED` | `experiments/handover_block_own5k_20261031/run_single_sigmoid_d_h50.py` 等 |
| HEAD `11ddf23`，工作树脏 | `VERIFIED` | 2026-11-08 本地 `git status` / `git log` |
| 转向 RoboTwin、短 HIL、先退右臂 | `REPORTED` | `_extract/resample/USER_TIMELINE.md` `U0090` `U0105` `U0159` `U0244` `U0251` |
| 接管率当时没有明显变化；coverage 挤在常数附近 | `REPORTED` | `U0563`（2026-10-03，`seg_0930_1003`） |
| 选择判别器并改为 chunk 加权 | `REPORTED` 决策；加权平均实现为 `VERIFIED` | `U0807`；`core.py` |
| 增强采样组“确实有提高” | `REPORTED` 且随后被「重新再来」打断 | `seg_1013_1018` 末条；下一段首条 |
| handover 24/50、stack 21/50 | `REPORTED` | `U1256` `U1287`，`seg_1031_1108` |
| 11-07 工程交接笔记 | `REPORTED` 二次摘要，本文件不把它当原始实验 | `~/Desktop/deepseek_task/Resample_Agent.md` |

命令级 stdout 不在本目录。按 thread id 查 `thread_items.item_type='commandExecution'`。
