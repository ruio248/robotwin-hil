# SFT 官方控制逻辑的最小逐查询记录版

新 SFT 正式测评统一运行 `robotwin_hil/overlay/scripts/eval_policy_xpolicylab_official_trace.py`。它以本仓库固定的 `RoboTwin/scripts/eval_policy_xpolicylab.py`（提交 `4c5c753d66c54c3f6ce855ddf426bcaf3e327250`）为基础，仅增加被动记录；官方 `RoboTwin/` 源码不改。隔离运行副本由 `python robotwin_hil/prepare_runtime.py --output /absolute/path/to/new-runtime` 生成，脚本及 `official_policy_trace.py` 位于副本的 `scripts/`。旧综合扩展 `eval_policy_xpolicylab.py` 的评测实现已经退役；同名文件只保留供旧 HIL/DAgger 脚本导入的工具函数，直接运行会报错。

在策略 websocket 服务启动且核实 checkpoint、H 和 `norm_stats` 后，在隔离副本根目录运行：

```bash
python scripts/eval_policy_xpolicylab_official_trace.py \
  --task_name handover_block --env_cfg_type aloha_agilex \
  --policy_name Pi_05_RobotTwin --host 127.0.0.1 --port 18493 --protocol ws \
  --root_dir "$PWD" --device_id 0 --task_config demo_clean \
  --seed 0 --test_num 50 --eval_batch false --expert_check true \
  --instruction_type seen --frequency 50 \
  --trace_dir /absolute/path/to/new-trace-directory
```

`--trace_dir` 必须指向尚不存在的目录；当前只支持单进程和绝对关节动作。`--frequency` 传给策略作为观测字段，不是动作 horizon 或物理仿真频率。专家预筛拒绝的场景记作 `expert_rejected`，不进正式分母。每条有效 episode 保存原样指令、真实终局、每次真实 `get_action` 的三路 RGB、state、完整动作 chunk 和实际执行动作数；不额外调用策略，也不用于 D 训练。`query_XXXX.npz` 是查询输入与返回值，不是连续视频。每次查询写压缩数组会增加测评时长；可先做 1 局技术 smoke，正式必须使用另一个新目录且 smoke 不计入 50 局。

成功/失败标记复用 `official_policy_trace.py`：每局 `episodes/episode_XXXX_seed_YYYY/episode.json` 及总 `manifest.json` 的对应 episode 都含 `success`（成功 `true`、正常策略失败 `false`、技术异常 `null`）、`status`（`success`、`policy_failure` 或 `infra_error`）、`rollout_failed`、`exception`、`counted_in_official_result`、`d_eligible`、`scene_seed` 和指令。成功来自官方循环在执行动作后读取的 `task_env.eval_success`，由任务 `check_success()` 驱动。正常失败是直到原任务终止仍未成功，不包括专家预筛拒绝或基础设施错误。总 manifest 汇总 `successful_episodes`、`policy_failure_episodes`、`query_covered_official_episodes`，并与官方计数校验。

如果场景初始化、服务、仿真、rollout 或记录写入异常，归档为 `infra_error` 并保留原因；即便上游计数把部分动作后的异常计为失败，该局的 `success` 也必须为 `null`，整个运行不能验收为干净的 D 样本。只有 `manifest.json` 为 `status=complete`、`d_analysis_ready=true`、50 个唯一被计场景、每局至少一次真实策略查询、无基础设施异常，才能使用同一轮终局标签和查询输入给冻结 D 做整轨评分。中断的 48/50 之类部分归档不能作为正式 50 局结果。

本入口保持固定 `4c5c753` 版官方脚本的场景/指令 RNG；此前 Ubuntu `ea8b211` 运行副本不是同一脚本版本，不能把两轮视为逐随机数配对。若更新 RoboTwin 提交，须重新核对源脚本 SHA 和控制逻辑差异。模拟对照测试见 `robotwin_hil/overlay/scripts/tests/test_official_trace_entry.py` 与 `test_official_policy_trace.py`；它们验证开关记录前后的控制行为和标签/异常门控，不代替实际 GPU 仿真的 50 局验收。
