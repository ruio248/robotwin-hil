# Codex 四项目历史交接总入口

更新时间：2026-11-08（Asia/Shanghai）。本轮只做历史恢复与只读核对，没有改项目源码、没有提交 Git、没有启动训练或连接机器人。

证据等级：

- `VERIFIED`：本机当前源码、Git 状态或本地结果文件直接支持。
- `REPORTED`：Codex 会话或既有交接笔记中的陈述，本轮没有重跑、也没有登录服务器复核。
- `PROPOSED`：讨论过或写过计划，没有对应实现或正式结果。
- `UNKNOWN`：记录缺失、互相冲突，或本轮无法访问。

会话时间戳在提取索引里是 **UTC**。北京时间 = UTC+8。

## 怎么读

1. 先读本页，确认项目边界和最大风险。
2. 进入对应项目的 `CURRENT_STATE.md`，看现在代码到底在哪、什么还不能声称。
3. 再读 `RESEARCH_HISTORY.md`，按时间看方案为什么改，尤其是否定结果。
4. 需要原文时，用 `EVIDENCE_INDEX.md` 回到会话编号，再到 `_extract/<project>/`。

可检索的原始摘录（不是压缩后的唯一副本）在：

`~/Desktop/deepseek_task/codex_history_handoff/_extract/`

各项目目录下的 `RESEARCH_HISTORY.md` 是本轮定稿。`_extract/*/DRAFT_RESEARCH.md` 只是检索过程中的辅助稿，不作为交接结论；若与定稿冲突，以定稿和 `EVIDENCE_INDEX.md` 为准。

每个项目有 `USER_TIMELINE.md`（非心跳用户消息，单条最多 4000 字）、`USER_ONELINERS.md`、`FINALS.md`、`FILE_CHANGES.tsv`、`DECISION_CANDIDATES.md`。这些文件来自 `~/.codex/thread_history_1.sqlite` 的 `userMessage` / `final_answer` / `fileChange`，不是 `history.jsonl`。`history.jsonl` 在本机不存在。

不要把 `_extract` 里的用户提示当成已经执行成功的实验。助手的 `final_answer` 仍是 `REPORTED`，除非本轮打开了对应代码或结果文件。

## 四个项目

| 目录 | 会话标题 | 当前阶段 | 先读 |
| --- | --- | --- | --- |
| [resample/](resample/CURRENT_STATE.md) | `实现 RESample 类采样覆盖算法` | 四任务 HG-DAgger 基线与官方评测对齐；在线 RESample 被当前任务适配器拒绝 | 当前四任务 `es-mode` 必须是 `off` |
| [context_qgf/](context_qgf/CURRENT_STATE.md) | `context-q — Gaussian ablation analysis`，前身 `context-q` | 协议已拆成 from-scratch、冻结 Actor、续训；Gaussian / 结构消融代码在本地，服务器完成度未复查 | 不要把冻结 Actor 消融叫成完整模型 |
| [am_qam/](am_qam/CURRENT_STATE.md) | `AM`、`ubuntu-rg-AM` | PAGR 仍是探索；toy 上多种稳健化没有稳定超过 lean AM；TRQAM 方向收缩只有诊断和 10 步试跑 | 不要把新算法写成已有性能提升 |
| [qam_rlinf/](qam_rlinf/CURRENT_STATE.md) | `QAM 融入RLinf` | 干净 PR 工作树在 `feat/qam-rlinf` `396d2c7`；入口仍是 `train_qam.py`，共用离线入口尚未接上 | 独立 QAM 仓库数字 ≠ RLinf 集成数字 |

## 当前最大风险

1. **协议混用。** Context-Q 的 from-scratch、冻结 Actor、Native500 续训到 530k/700k 是三种不同比较。RESample 的 H10 TD coverage、H10 判别器和 H50 单 D 也不是同一个分数。
2. **负结果容易被后来的新名字盖住。** PAGR / Directional Robust / norm-only 在已有本地报告里没有稳定增益。RESample 早期接管率没有明显提高，后来转到判别器和官方评测对齐。
3. **工作树是脏的。** RESample 主仓库相对 `11ddf23` 有约 73 项本地改动。Context-Q 的 gaussian 与 temporal 工作树也有未提交修改。只 clone HEAD 恢复不了当前行为。
4. **服务器状态是 11 月 7 日笔记里的快照。** 本轮没有 SSH。Scene seed、结构消融、DAgger 监控心跳在 11 月 8 日会话里仍出现，是否还在跑是 `UNKNOWN`。
5. **PR #1811 的 GitHub 实时状态未复查。** 本地提交仍是 `396d2c7`，`train_offline_rl.py` 仍只接受 IQL。

## 推荐阅读顺序

1. 本页。
2. `qam_rlinf/CURRENT_STATE.md`：边界最清楚，适合先建立“已实现 / 未按评审改完”的读法。
3. `am_qam/RESEARCH_HISTORY.md`：负结果最集中，避免把 PAGR 当成已成功算法。
4. `context_qgf/RESEARCH_HISTORY.md`：先看协议三分，再看 Gaussian。
5. `resample/RESEARCH_HISTORY.md`：从真机成本转到 RoboTwin，再到官方接口对齐；在线增强采样尚未接到当前四任务。
6. 各项目 `EVIDENCE_INDEX.md`，然后按需打开 `_extract`。

## 本轮没有恢复完整的东西

- `~/.codex/history.jsonl` 不存在。不能把别的索引误当成完整提示词历史。
- 大量 rollout JSONL 已不在 `sessions/` 里，但 SQLite `thread_items` 仍有用户消息和 `final_answer`。命令的完整 stdout 没有抄进交接包（避免凭据和体积）；需要时按 `EVIDENCE_INDEX.md` 的 thread id 回查 SQLite。
- `contextCompaction` 记录几乎只有 id，没有可用的压缩摘要。
- 同一逻辑对话被 Codex resume 拆成互不相交的 thread id。只读 `threads` 表里的主 id 会丢掉后半段。分段对照见各 `TRANSCRIPT_INDEX.md`。
- `archived_sessions/` 只有两个 `.jsonl.zst`（2026-06-27、2026-08-21），本轮没有解压。
- 远端 checkpoint、正在运行的 PID、PR 评论的最新回复，都没有在本轮重新拉取。
