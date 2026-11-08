# 进度共识（与用户 2026-11-08 对齐）

本页记录已口头确认的阶段叙事，以及交接后的正式测评结论。  
Ubuntu `ubuntu1` 终值已本地打开 summary，标 `VERIFIED`；A800 `handover_block` `retry1` 仍按交接快照标 `REPORTED`。

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

## 当前工程进度（2026-11-08 晚更新）

| 事项 | 共识说法 | 细节 |
| --- | --- | --- |
| HIL 采集 | 四任务各约 15 条短纠偏段已用于本轮训练 | 见 `Resample_Agent.md` §B |
| A800 训练 | 已训一波 DAgger（H50、`state=True`、SFT50+HIL15、300→299） | `dagger_four_h50_5050_300_20261107` |
| 正式测评 | **已测任务相对 SFT 全面掉点；多任务 DAgger 科研闭环未走通** | 见 `02_FOUR_TASK_DAGGER_EVAL_NEGATIVE_20261108.md` |

### 正式成功率

| 任务 | DAgger299 | 旧 SFT4999 参照 | 状态 |
| --- | --- | --- | --- |
| `handover_block` | **0/50** | ~24/50 | A800 `retry1`，`REPORTED` |
| `rotate_qrcode` | **16/50** | ~39/50 | Ubuntu `ubuntu1`，`VERIFIED` |
| `place_empty_cup` | **4/50** | ~32/50 | Ubuntu `ubuntu1`，`VERIFIED` |
| `stack_blocks_three` | 未测 | ~21/50 | 本机缺 299 checkpoint |

## 口语 vs 精确说法

- 口语：“多任务 DAgger 没走通”  
- 精确：**采集/训练能跑完，但已测三任务 DAgger299 自主成功率显著低于交接中的 SFT 参照；不能宣称流程科研成功，也不能在未对齐协议前把锅只扣成算法天生副作用。**

## 下一步（PROPOSED，未授权自动执行）

1. 拉取并补测 `stack_blocks_three` 299。  
2. 同机同协议对照 SFT4999 vs DAgger299。  
3. 用严格 accepted HIL 重导出/重训，隔离 `diagnostic_all` 例外。  
4. 在未冻结资格与评测协议前，不要打开四任务 `--es-mode enhanced`。

## 不要做

- 不要把 tray 历史结果与四任务 H50 混表。  
- 不要把接管率上升当成最终成功指标。  
- 不要把 A800 与 Ubuntu 局数合并。  
- 不要擅自改 `supports_sampling`。
