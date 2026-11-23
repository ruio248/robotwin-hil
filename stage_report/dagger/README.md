# DAgger 采集、训练与评测结果（2026-11-23）

生成时间：2026-11-23T15:26:20+08:00

## HIL 采集与训练选择

| 任务 | 已保存 episode | 采集 HIL 片段 | 训练选用片段 | seed 记录 |
|---|---:|---:|---:|---|
| `place_empty_cup` | 14 | 15 | 15 | [采集 seed 索引](collection_seed_index.csv) |
| `pick_dual_bottles` | 15 | 15 | 15 | [采集 seed 索引](collection_seed_index.csv) |
| `place_dual_shoes` | 15 | 17 | 15 | [采集 seed 索引](collection_seed_index.csv) |
| `handover_block` | 9 | 16 | 15 | [采集 seed 索引](collection_seed_index.csv) |
| `beat_block_hammer` | 15 | 15 | 15 | [采集 seed 索引](collection_seed_index.csv) |
| `click_alarmclock` | 15 | 15 | 15 | [采集 seed 索引](collection_seed_index.csv) |

采集 episode 的 `episode.json` 保存 seed、指令、控制片段和帧；本目录发布采集 seed 索引。完整原件、轨迹和视频保留在本地证据归档中，未纳入 Git。
`place_empty_cup` 的原始 HIL 采集来自 `/hdd/h10_hil_recollect_20261114/place_empty_cup/`；15 个片段及对应 LeRobot 数据已纳入归档。

## 统一 seed 正式评测

每个任务在同一协议下使用固定 50 个 seed 和指令；原始 `manifest.json` 为正式成功标签；本目录的 `seed_results.csv` 发布逐 seed 成功标签。轨迹与视频保留在本地证据归档中。

| 协议 | 任务 | checkpoint | 成功数 | 状态 |
|---|---|---|---:|---|
| h10_paired_20261122 | `place_empty_cup` | `sft` | 36/50 (72%) | complete |
| h10_paired_20261122 | `place_empty_cup` | `lora_step_69` | 41/50 (82%) | complete |
| h10_paired_20261122 | `place_empty_cup` | `lora_step_139` | 39/50 (78%) | complete |
| h10_paired_20261122 | `place_empty_cup` | `lora_step_209` | 39/50 (78%) | complete |
| h10_paired_20261122 | `handover_block` | `sft` | 8/50 (16%) | complete |
| h10_paired_20261122 | `handover_block` | `lora_step_49` | 6/50 (12%) | complete |
| h10_paired_20261122 | `handover_block` | `lora_step_99` | 12/50 (24%) | complete |
| h10_paired_20261122 | `handover_block` | `lora_step_149` | 13/50 (26%) | complete |
| h10_paired_20261122 | `place_dual_shoes` | `sft` | 24/50 (48%) | complete |
| h10_paired_20261122 | `place_dual_shoes` | `lora_step_48` | 25/50 (50%) | complete |
| h10_paired_20261122 | `place_dual_shoes` | `lora_step_97` | 25/50 (50%) | complete |
| h10_paired_20261122 | `place_dual_shoes` | `lora_step_146` | 31/50 (62%) | complete |
| h50_paired_20261123 | `click_alarmclock` | `lora_step_65` | 36/50 (72%) | complete |
| h50_paired_20261123 | `beat_block_hammer` | `sft` | 21/50 (42%) | complete |
| h50_paired_20261123 | `beat_block_hammer` | `lora_step_24` | 28/50 (56%) | complete |
| h50_paired_20261123 | `beat_block_hammer` | `lora_step_49` | 27/50 (54%) | complete |
| h50_paired_20261123 | `beat_block_hammer` | `lora_step_74` | 29/50 (58%) | complete |
| h10_pick_sftnorm_20261123 | `pick_dual_bottles` | `sft` | 16/50 (32%) | complete |
| h10_pick_sftnorm_20261123 | `pick_dual_bottles` | `lora_step_49` | 21/50 (42%) | complete |
| h10_pick_sftnorm_20261123 | `pick_dual_bottles` | `lora_step_99` | 20/50 (40%) | complete |
| h10_pick_sftnorm_20261123 | `pick_dual_bottles` | `lora_step_149` | 19/50 (38%) | complete |

### `pick_dual_bottles` 原始 SFT 统计量重训

训练选用 15 个 HIL 片段、797 帧；batch size 32（SFT 16 + HIL 16），每 50 step 扫过一次 HIL 帧，共 3 epoch / 150 step。LoRA only、EMA 关闭、seed 42；step 49、99、149 各保存一个 adapter。三个 adapter 的 `norm_stats.json` 与原始 SFT 文件按 SHA256 完全一致（`8a8bfc2d2c99a184c22bdaeadab070d6a187abba219fc924ddf8a72fdc0d4f2f`）。
训练 checkpoint 校验：完成。旧混合统计量 adapter 清理：完成。
三个 LoRA 中最佳：`lora_step_49`，21/50；它与上表 SFT 使用同一批 seed 和指令。

## 需要保留的限制

- `click_alarmclock` 历史 H=50：SFT 34/50，step 32 为 36/50，step 65 为 39/50，step 98 为 36/50。历史四组都跳过了筛选阶段出现异常的 seed 100006；step 32 在 seed 100008 的指令与 SFT 不同。新 step 65 使用冻结清单，结果为 36/50。历史记录见 `data/evaluations/h50_retest_20261122/click_alarmclock/`。
- `beat_block_hammer` 历史 H=50：SFT 24/50，step 24 为 24/50，step 49 为 30/50，step 74 为 31/50；各组独立筛选 seed，正式比较使用上表的新配对结果。历史记录见 `data/evaluations/h50_retest_20261122/beat_block_hammer/`。
- 独立任务 `click_clock` 未找到对应 adapter、采集或评测数据；现有记录属于 `click_alarmclock`。
- `pick_dual_bottles` 旧版 LoRA step 49、99、149 使用重新混合的统计量（SHA256 前缀 `6b173ffff5e7`），原始 SFT 为 `8a8bfc2d2c99`；旧版 0/50 不作为可信的统一 seed 结论。新原始统计量训练版及其配对复测列于上表。
- 查询轨迹保存三路图像、机器人状态和 H 步动作；未保存每个物理步的物体位姿、接触和完整仿真状态。

交互查看：[评测浏览器](index.html)。逐 seed 数据：[seed_results.csv](seed_results.csv)；来源 manifest 校验和：[provenance.csv](provenance.csv)。
