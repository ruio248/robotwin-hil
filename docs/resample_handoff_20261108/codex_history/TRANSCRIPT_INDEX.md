# RESample 会话索引

逻辑对话是多段 resume 拼起来的。主线程 id 与续段 id 的用户消息 **没有交集**，必须合并。时间是 UTC。

原始库：`~/.codex/thread_history_1.sqlite`，表 `thread_items` / `thread_turns`。
展示名来自 `~/.codex/state_5.sqlite` 的 `threads.name`。`session_index.jsonl` 只有名字，不是正文。

摘录：`../_extract/resample/USER_TIMELINE.md`（编号 `U####`）、`FINALS.md`（`F####`）、`FILE_CHANGES.tsv`。

## 主链（增强采样 → 官方评测对齐）

| 标签 | thread id | 用户消息时间 (UTC) | 约用户条数 | rollout 文件 | 主题 |
| --- | --- | --- | --- | --- | --- |
| `seg_0911_0914` | `01a09011-7b9c-7573-b511-22e23700c572` | 2026-09-11 → 09-14 | 105 | 不在 `threads` 表；JSONL `UNKNOWN` | 真机成本过高，转向 RoboTwin 长程任务 |
| `seg_0914_1026_robotwin_plan` | `01a0a11b-fc71-7412-bcfa-b5498d5a61ee` | 09-14 → 10-26 | 65 | 同上 | handover-to-tray 基线计划；第二轮 DAgger 是否混入 policy rollout |
| `robotwin_hil_parent` | `01a0af9c-f358-7822-9d8f-25d2b0a56b8e` | 见 SQLite | 约 90 turns | JSONL **缺失** | 名称 `robotwin_hil`。接管、交还、只存 HIL 短段 |
| `robotwin_hil_seg` | `01a0b52e-c1ef-7920-90d0-97619743a626` | 09-18 → 09-19 | 25 | 文件名曾挂在 parent rollout 后缀 | 专家脚本近距离救不回；先退右臂 |
| `named_parent` | `01a0ba85-24c0-7e00-aa7a-0f7ef7eb4355` | 与下段并列，消息不重复 | 223 | **存在** `sessions/2026/11/01/rollout-...01a0ba85..._01a192bd....jsonl`（759 MB） | 展示名就是 `实现 RESample 类采样覆盖算法` |
| `seg_0924_0930` | `01a0d4e0-a176-7462-b693-83ac4952c800` | 09-24 → 09-30 | 85 | 无 threads 行 | `codex/coverage-enhanced-sampling`；比较 enhanced / vanilla / off 的接管请求 |
| `seg_0930_1003` | `01a0f135-a557-7c20-8b72-daca5d6a39fa` | 09-30 → 10-03 | 46 | 无 threads 行 | 删掉旧增强采样数据重来；coverage 不应逐条回放 |
| `seg_1003` | `01a10106-224a-76b1-b729-c577869ca25d` | 10-03 | 16 | 无 threads 行 | 四候选差异随步数 |
| `seg_1003_1004` | `01a1017b-78e0-7851-be0d-4891cf587dde` | 10-03 → 10-04 | 30 | 无 threads 行 | 终点 reward 不足；B 曲线太平 |
| `seg_1004_1013` | `01a1055c-2beb-7191-b207-729c1ea88e94` | 10-04 → 10-13 | 248 | 无 threads 行 | TD / 判别器 / R3M / DINOv2 指标比较 |
| `seg_1013_1018` | `01a13481-a04c-7630-8e76-1d21460c0338` | 10-13 → 10-18 | 143 | 无 threads 行 | 用户说增强采样组有提高，要求论文解释；随后要求按官方设置重来 |
| `seg_1018_1031` | `01a14f85-cd67-79a2-bc69-5fd5b3e0314b` | 10-18 → 10-31 | 371 | 无 threads 行 | 任务筛选、Handover 与官方不对齐、仓库结构 |
| `seg_1031_1108` | `01a192bd-e098-7193-ae26-bf39c364208e` | 10-31 → 11-08 | 470 | 挂在 named_parent 的 rollout 后缀 | 官方接口重训与四任务 DAgger 监控 |

续段之间有首尾复读（例如 10-13「重新再来过吧」同时是上一段末尾和下一段开头），合并时间线时不要当成两次独立决策。

## 子会话（spawn）

父线程 `01a0ba85-...` 在 `thread_spawn_edges` 中有三个 open 子线程，rollout 仍在：

| thread id | 名称 | 文件 |
| --- | --- | --- |
| `01a1a06d-f785-73d1-b5ad-a67c6fcdadc5` | Review dagger migration architecture | 约 1.1 MB |
| `01a1a071-7dfa-7522-8774-0eade30be66e` | Split Dagger task adapters | 约 0.5 MB |
| `01a1a076-eaa0-70d2-a051-ab7cc361d03e` | Review top-level Dagger refactor | 约 1.3 MB |

它们是代码评审，不是另一套算法实验。

## 相关但不要并进主结果

- `019f6e82-...` 展示名涉及失败发现与增强采样测试分层。本索引未把它的全文并入主链。
- `01a081b1-...`「调研Online主动探索长尾状态」、`01a085ca-...`「解释探索采样与恢复轨迹」、`01a09f2f-...` Tail-Likelihood、`01a0a4e9-...` coverage principle：偏阅读与概念，不是 RoboTwin 主实验链。
- PiperX 真机 RESample B pilot 的笔记在 `~/Desktop/ruihao/RESAMPLE_B_PILOT_EXPERIMENT_REPORT_20260911.md`，与后来的 RoboTwin D 不是同一数据。

## 检索方法

```sql
-- 例：某段全部用户消息
SELECT created_at_ms, rollout_ordinal, item_json
FROM thread_items
WHERE thread_id = '01a192bd-e098-7193-ae26-bf39c364208e'
  AND item_type = 'userMessage'
ORDER BY rollout_ordinal;
```

命令输出在同一表的 `item_type='commandExecution'`，本交接包没有整段复制。
