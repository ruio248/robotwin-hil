# RESample 当前状态

核对时间：2026-11-08 晚，只读。没有重跑测试，没有 SSH。

## 代码在哪

主工作树：`/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling`

- remote：`ssh://git@ssh.github.com:443/ruio248/robotwin-hil.git`
- 分支 `main`，HEAD `11ddf23`（2026-11-05，`Align DAgger HIL segment thresholds and verify HDF5 exports`）
- `git status`：相对 HEAD 约 73 项本地改动（含 `dagger/runner.py`、各 adapter、若干 docs，以及删除旧 eval 脚本）。未提交。本轮没有 fetch，远端是否更新是 `UNKNOWN`。

在线入口是 `dagger.runner`。采样库在 `robotwin_hil/overlay/scripts/coverage_sampling/`。H10 TD chunk critic 在 `robotwin_hil/overlay/scripts/tail_value/chunk.py`。H50 sigmoid D 的训练脚本在 `experiments/handover_block_own5k_20261031/` 与 `experiments/*_official_d_20261102/` 一类目录，不在四任务 runner 的允许采样路径上。

## 现在实际会跑的算法

四任务适配器 `handover_block`、`place_empty_cup`、`rotate_qrcode` 继承 `LiveObjectAdapter.supports_sampling = False`。`stack_blocks_three` 显式 `supports_sampling = False`，注释写明不要继承 handover 的历史 H10 critic。`dagger/runner.py` 在 `--es-mode` 不是 `off` 时直接报错。

因此当前四任务的有效采集算法是 **HG-DAgger / 人工阶段选择的短 HIL**，不是 RESample。`VERIFIED`。

`handover_to_tray` 直接继承 `TaskAdapter`，默认 `supports_sampling = True`。它是历史方法平台，不是这四个科研任务。

采样器若被旧任务打开，行为是：同一观测上采 K 条绝对关节 chunk，用 coverage 权重做加权平均；Gaussian 模式是均匀均值加臂关节噪声。权重公式见 `RESEARCH_HISTORY.md`。H=10 的 full-chunk 判别与 H=50 的实验 D 不要混用。

## 实验完成情况

| 事项 | 等级 |
| --- | --- |
| 接管交互、交还、短 HIL、阶段退让，经过多轮用户现场测试并反复改代码 | `REPORTED` 过程；当前源码仍保留这套入口 `VERIFIED` |
| 2026-10-03 前后 enhanced 相对 off 的接管率没有明显拉开 | `REPORTED`（用户待办 `U0563`） |
| 判别器在当时指标表上被选为继续改代码的版本；TD 的成功/失败走势被用户认为更合理 | `REPORTED`。本轮未重算那张表 |
| handover_block 官方接口 5k：24/50；stack 5k：21/50 | `REPORTED`（`U1256`，`U1287`）。未打开结果文件 |
| 四任务 H50 在线 RESample 相对 DAgger 基线的再训练提升 | 没有。当前入口拒绝该模式 |
| 11-08 心跳里的四任务训练与 50 局测评是否已完成 | `UNKNOWN`（未登录 A800） |

## 阻塞

1. 工作树脏，且实验目录大量未跟踪。交接不能假设 `git clone` 等于当前行为。
2. 在线 RESample 与当前四任务 H50 策略、官方评测协议没有接上。H10 采样器不能直接套到 H50 checkpoint。
3. 专家可恢复性仍是未验证的科学前提。近距离失败已经出现过。
4. 远端训练进程状态未知。11-08 会话心跳还在盯 A800，不要凭此假定 GPU 空闲，也不要假定结果已齐。

## 接手后的三件事

1. 在本地把 `git status` 和 `experiments/` 里最新官方评测 README 对上，确认 48%/42% 是否仍有文件，并标明 seed 与预筛拒绝数。不要重训。
2. 只读确认四任务 DAgger 数据是否已按当前 runner 导出（HIL 段最短长度以 `11ddf23` 及脏 diff 里的实际代码为准）。
3. 若要恢复在线 RESample，先写清用哪一个 D（H10 TD、H10 判别器，还是 H50 sigmoid）、K、窗口和“低分才采样”的约束，并在一个已对齐的任务上离线证明它能分开同观测的专家动作与策略动作。不要直接打开四任务 `--es-mode enhanced`。

## 不要做

不要为了“把 RESample 接回去”去改 `supports_sampling`，除非用户重新授权。本交接不授权训练、清数据或停远端进程。
