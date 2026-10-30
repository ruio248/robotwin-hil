# HG-DAgger：执行与数据采集

## 启动前

先确认生成运行目录已更新、图形桌面可见、任务和服务端加载同一策略 checkpoint/配置/`norm_stats`，独立策略服务端口身份明确。`robotwin_hil/local_serving/start_local_policy_server.sh` 是服务入口；采集器只连接既有服务，不应为重开采集而停止其他端口。`HIL_DRY_RUN=1` 可检查 `robotwin_hil/local_serving/run_one_stage_dagger.sh` 组装的命令；无 viewer 的 headless 冒烟不能验证真实按键。

末段对照的仓库根目录启动示意（输出目录与报告文件须替换为本次新路径）：

```bash
ONE_STAGE_ARM=baseline MANUAL_TARGET_SAVED=30 \
  MANUAL_OUTPUT_DIR=/data/one_stage_baseline \
  bash robotwin_hil/local_serving/run_one_stage_dagger.sh

ONE_STAGE_ARM=enhanced MANUAL_TARGET_SAVED=30 \
  HIL_ONE_STAGE_PAIRED_REPORT=/data/one_stage_baseline/session_actual.json \
  MANUAL_OUTPUT_DIR=/data/one_stage_enhanced \
  bash robotwin_hil/local_serving/run_one_stage_dagger.sh
```

`ONE_STAGE_ARM=baseline` 是 K=1、增强采样关闭；`enhanced` 为 K=32、D 对完整 H10 chunk 评分和加权，从阶段 6 首次决策自动开启。增强组读取**完成的** baseline session，并跑其合格末段 seed 清单；即使先收满目标有效 HIL，也不得为了整齐数字中途截断配对清单。Gaussian 是另一个独立 arm，需显式设定 σ，不应混入上述两组。

## 现场操作与记录

操作者在 GUI 中按 `i` 请求接管、`x` 放弃当前 episode、`q` 结束；单阶段不按 `r`。`hg_dagger_handover.py` 在每条记录保存 `stage6_handoff`、`manual_takeover_requests`、`expert_result`、`hil_frames`、`joints_legal`、`valid_hil`、`autonomous_success`、`save_decision`，会话报告保存总量与 `aborted`。有效 HIL 需发生介入、有 HIL 帧、专家恢复成功、最终任务成功且规划关节合法；X 与接管后失败不能算有效数据。

原始帧在 `raw/episode_*/frames/*.pkl`，元数据在 `episode.json`，总表在 `episodes.jsonl` / `session_*.json`。`robotwin_hil/overlay/scripts/export_hg_dagger_dataset.py` 可按 `full` 或 `hil` 导出；带 `diagnostic_only` 或 `accepted_for_training=false` 的记录会跳过。继续中断会话只能用相同配置 `--resume` 并检查已有报告，不能在旧目录上直接重启。采集结束先核查 `aborted=false`、计数与原始数据完整，再启动离线整理/训练。
