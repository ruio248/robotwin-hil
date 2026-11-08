# 进度共识（与用户 2026-11-08 对齐）

本页记录已口头确认的阶段叙事，以及交接快照里的精确状态。  
未在本轮重新 SSH 验收 A800 测评终值时，标 `UNKNOWN` / `REPORTED`。

## 研究阶段（已确认）

1. **自定义任务 `handover_to_tray`（口头 hand_over_tray）**  
   作为方法平台：HG-DAgger 短纠偏 + 增强采样试验。中间曾观察到接管率升高、以及“增强组有提高”的阶段性判断；归因不干净，且随后因官方协议对齐要求推倒重来，**不能写成已定论的 RESample 增益**。

2. **要做多任务验证**  
   自定义 tray 协议难与官方榜单对齐；改为用官方 π0/π0.5 任务做 DAgger 泛化验证。

3. **筛任务（目标约 ≤10）**  
   主要看官方单任务成功率是否落在可做 DAgger 的区间（约 40%–80%），以及是否适合写接管/恢复脚本；过程中做过约十任务测评与 D 相关分析。

4. **四个任务落地 HG-DAgger**  
   `handover_block`、`rotate_qrcode`、`place_empty_cup`、`stack_blocks_three`。  
   当前采集入口 **`--es-mode off`**（在线 RESample 未接回这四条路径）。

## 当前工程进度（已确认方向，细节见快照）

| 事项 | 共识说法 | 快照细节 |
| --- | --- | --- |
| HIL 采集 | 四任务各约 15 条短纠偏段已用于本轮训练 | `REPORTED`/`VERIFIED` 见 `Resample_Agent.md` §B |
| A800 训练 | 已训一波 DAgger（H50、`state=True`、SFT50+HIL15、300 step → `299`） | 实验名 `dagger_four_h50_5050_300_20261107` |
| 正式测评 | **训练已完成；测评批次未全部收官** | 不是“完全没开始测” |

约 2026-11-07 13:07 CST 的 A800 `retry1` 快照（`REPORTED`，见 `Resample_Agent.md` §C）：

| 任务 | DAgger299 |
| --- | --- |
| `handover_block` | complete：**0/50** |
| `rotate_qrcode` | 当时 running（快照非终值） |
| `place_empty_cup` | 当时尚无 retry1 结果 |
| `stack_blocks_three` | 当时尚无 retry1 结果 |

2026-11-08 会话心跳仍在盯 A800；**此后 QR/cup/stack 是否跑完 = `UNKNOWN`（交接整理时未再验收）**。

## 口语 vs 精确说法

- 口语：“A800 训了一波 DAgger，现在等测评”  
- 精确：**训练完了；A800 正式测评进行中/未收齐（handover 已出 0/50，另三任务待确认）**；另有 Ubuntu 传 checkpoint / 对照测评线，完成度未验收。

## 下一步（PROPOSED，未授权自动执行）

1. 只读核对 A800 `evaluations/*_retry1` 是否已齐，做 NPZ/summary 终验。  
2. 同机同协议对照旧 SFT4999 vs DAgger299；跨机器/`_base_task`/prompt 未对齐时不作因果结论。  
3. 在未冻结资格与评测协议前，不要打开四任务 `--es-mode enhanced`。

## 不要做

- 不要把 tray 历史结果与四任务 H50 混表。  
- 不要把接管率上升当成最终成功指标。  
- 不要重复启动已有 `retry1` runner。  
- 不要擅自改 `supports_sampling`。
