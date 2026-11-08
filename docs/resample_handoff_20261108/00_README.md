# RESample / 四任务 HG-DAgger 交接包（Ubuntu）

更新时间：2026-11-08 晚（Asia/Shanghai）  
本包只做交接与只读说明，**不授权**继续旧会话的训练、清理磁盘、停止远端进程或覆盖实验目录。  
**2026-11-08 晚补充**：已测 DAgger299 相对 SFT 全面掉点，见 `02_FOUR_TASK_DAGGER_EVAL_NEGATIVE_20261108.md`——多任务 DAgger **科研闭环未走通**。

## 目标机路径约定

Ubuntu 项目根（交接文档写入处）：

```text
/hdd/robotwin-hil-github/docs/resample_handoff_20261108/
```

同机实验输出对照（若已同步）：

```text
/hdd/robotwin-hil-github/outputs/dagger_four_h50_5050_300_20261107/
/media/ruio/hdd/robotwin-hil-github/outputs/dagger_four_h50_5050_300_20261107/
```

A800 训练根（node3 / `new_power_3`）：

```text
/data-training/robotwin-hil-lyt/outputs/dagger_four_h50_5050_300_20261107/
```

本机源码工作树（Mac，脏工作树 ≠ HEAD）：

```text
/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling
```

## 先读顺序

1. `00_README.md`（本页）
2. `01_PROGRESS_CONFIRMED.md`（当前进度共识）
3. `02_FOUR_TASK_DAGGER_EVAL_NEGATIVE_20261108.md`（正式测评负结果）
4. `docs/Resample_Agent.md`（2026-11-07 工程审计主交接）
5. `codex_history/CURRENT_STATE.md`
6. `codex_history/RESEARCH_HISTORY.md`
7. 需要回查原文时再看 `codex_history/EVIDENCE_INDEX.md` 与 `TRANSCRIPT_INDEX.md`

更完整的四项目总入口在 Mac 上的  
`~/Desktop/deepseek_task/codex_history_handoff/00_MASTER_INDEX.md`；本包只携带 RESample 相关定稿。

## 包内文件

| 路径 | 作用 |
| --- | --- |
| `00_README.md` | 本入口 |
| `01_PROGRESS_CONFIRMED.md` | 与用户确认过的阶段叙事与当前卡点 |
| `02_FOUR_TASK_DAGGER_EVAL_NEGATIVE_20261108.md` | 四任务 DAgger 正式测评负结果（流程未走通） |
| `evidence_eval_20261108/` | Ubuntu `ubuntu1` formal summary JSON 副本 |
| `docs/Resample_Agent.md` | 工程细节、机器、命令、P0/P1 |
| `codex_history/CURRENT_STATE.md` | 2026-11-08 只读状态 |
| `codex_history/RESEARCH_HISTORY.md` | 方案为何改、负结果 |
| `codex_history/EVIDENCE_INDEX.md` | 结论 → 证据 |
| `codex_history/TRANSCRIPT_INDEX.md` | Codex 会话分段 |
| `codex_history/00_MASTER_INDEX.md` | 四项目总览（上下文） |

原始长摘录（`USER_TIMELINE` / `FINALS` 等）仍在 Mac：

```text
~/Desktop/deepseek_task/codex_history_handoff/_extract/resample/
```

未整包复制到 Ubuntu，避免体积与凭据风险；需要时按 `TRANSCRIPT_INDEX.md` 回查本机 SQLite。

## 证据等级

- `VERIFIED`：当前源码 / 本地结果文件支持
- `REPORTED`：会话或交接笔记陈述，本轮未重跑/未 SSH 复核
- `PROPOSED`：讨论过，无闭环实现或正式结果
- `UNKNOWN`：缺失、冲突，或快照后状态未再查

## 硬约束

1. 四任务 CLI 当前必须 `--es-mode off`；不要为了“接回 RESample”改 `supports_sampling`，除非用户重新授权。
2. 不要重跑已有 A800 `retry1` 正式测评批次；不要把 A800 与 Ubuntu 局数合并。
3. `handover_to_tray` 是历史方法平台；科研四任务是  
   `handover_block` / `rotate_qrcode` / `place_empty_cup` / `stack_blocks_three`。
4. H10 TD、H10 判别器、H50 sigmoid D、四任务 HG-DAgger 不是同一分数，禁止混比。
