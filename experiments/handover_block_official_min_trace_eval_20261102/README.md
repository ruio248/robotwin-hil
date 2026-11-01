# handover_block：官方源码副本 + 最小记录版测评

目标：用官方 PR #510 指令修复后的 `handover_block`、自训官方数据 5k checkpoint/4999，按 `docs/sft/SFT_EVALUATION.md` 的正式流程一次生成成功率和 D 所需的同局查询轨迹、真实成功/失败标记。本实验不训练 D，轨迹也不回流 D 训练。

Ubuntu 源副本：`/media/ruio/hdd/handover_block_official_min_trace_eval_20261102/official_runtime`（必须挂载 `/dev/sda1`）。它从本仓库固定的 RoboTwin `4c5c753` 复制，**仅**增加 `eval_policy_xpolicylab_official_trace.py` 与 `official_policy_trace.py`；`envs/_base_task.py`、任务与指令均逐文件与官方源比较。通用 `prepare_runtime.py` 会叠加 HIL `_base_task.py`，不能用于此处声称官方原样的 SFT 测评。未使用的 `runtime_robotwin` 是在发现这个差异前生成的 HIL overlay 副本，不参加正式运行。

首次运行在上述源副本根下做 smoke，尚未进入仿真便因 GitHub checkout 缺少 `assets/objects/objaverse/list.json` 而退出；`control/state.json` 与 `smoke/eval.log` 保留。之后将先前官方运行副本的 `assets/` 缺失文件复制进来，`diff -qr` 核对资产树逐文件一致。代码文件仍与固定官方源码相同。**唯一后续结果根**为 `/media/ruio/hdd/handover_block_official_min_trace_eval_20261102_retry1`；不覆盖第一次失败目录。

服务沿用旧的同一 5k checkpoint 配置及 `server_entry.py`、`server.yaml`；启动前验证其 Pi_05 服务源码与新官方副本的 SHA 一致。GPU0、端口 18496，服务 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55`，仿真 `XLA_PYTHON_CLIENT_PREALLOCATE=false`。`run.py` 先做 1 局技术 smoke，随后单进程正式 50 局：`demo_clean`、`seed=0` 从 100000 起、`expert_check=true`、`seen`、`frequency` 字段 50、完整 H50 绝对关节动作。smoke 不进入正式率。

正式归档 `formal/trace/manifest.json` 和每局 `episode.json` 保存 `success=true/false/null`、`status=success/policy_failure/infra_error`、`rollout_failed`、异常、专家拒绝 seed；每次查询另存 RGB/state/动作 chunk。只有 `status=complete`、`d_analysis_ready=true`、50 唯一计数 seed、零技术异常且官方结果计数一致才能报告干净成功率、用冻结 D 分析。不接受部分进度或旧无轨迹 24/50 的标签。

注意与此前 Ubuntu 未修改官方脚本 `ea8b211` 的 24/50 测评不同：本次固定 RoboTwin `4c5c753`，除指令 RNG 不同外，官方 `_base_task.py` 也有差异，因此两轮不是严格同环境配对，不应把率差解释成 checkpoint 效果。

运行方式（新结果根只启动一次）：`/usr/bin/python3 run.py`。状态、PID、服务日志、正式日志依次看 `control/state.json`、`control/runner.pid`、`control/server.pid`、`control/server.log`、`formal/eval.log`。发生技术错误时脚本保留归档、标记失败，不自动重启；不覆盖旧结果。最终结果待验收后填写。
