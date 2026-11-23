# SFT 阶段评测结果

本目录汇总这轮 DAgger 对比所使用的 SFT 基线。五个干净基线各使用固定 50 个 seed 和指令，与同任务 LoRA 配对；评测均无基础设施错误或跳过场景。`click_alarmclock` 只有历史诊断 SFT 结果，单列如下。

| 任务 | H | 协议 | SFT 成功 | 状态 |
|---|---:|---|---:|---|
| `place_empty_cup` | 10 | `h10_paired_20261122` | 36/50 (72%) | 干净配对基线 |
| `handover_block` | 10 | `h10_paired_20261122` | 8/50 (16%) | 干净配对基线 |
| `place_dual_shoes` | 10 | `h10_paired_20261122` | 24/50 (48%) | 干净配对基线 |
| `beat_block_hammer` | 50 | `h50_paired_20261123` | 21/50 (42%) | 干净配对基线 |
| `pick_dual_bottles` | 10 | `h10_pick_sftnorm_20261123` | 16/50 (32%) | 干净配对基线 |
| `click_alarmclock` | 50 | `h50_retest_20261122` | 34/50 (68%) | 历史诊断；筛选阶段 1 个基础设施错误 |

逐 seed 标签见 [seed_results.csv](seed_results.csv)。每行保留协议、任务、H、checkpoint、seed、指令、结果和 rollout 步数。

## 解释边界

- SFT 基线率来自自主 rollout；LoRA 的提升只与同任务、同协议、同 seed 的干净基线比较。
- `click_alarmclock` 历史 SFT 出现 seed 100006 的专家筛选错误；不能把 34/50 与固定 seed 的 LoRA step 65 36/50 当作严格配对提升。
- `click_clock` 是独立任务，当前没有可发布的对应 SFT 结果。
- 原始 manifest、查询 NPZ 和视频保留在本地 `data/evaluations/` 证据归档；此目录仅发布紧凑结果。
