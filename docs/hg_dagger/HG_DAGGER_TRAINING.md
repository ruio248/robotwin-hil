# HG-DAgger：数据整理与策略训练

## 先明确监督来源

保存的 HIL episode 包含策略控制片段与专家接管片段，不能把整条轨迹都标成专家动作。`robotwin_hil/overlay/scripts/export_hg_dagger_dataset.py --mode hil` 只导出 HIL 控制帧；动作标签按当前帧到下一保存帧的绝对关节目标构造，连续片段不能跨 `control_mask` 的策略/HIL 边界。诊断、X 放弃、接管失败或不合法轨迹均不进入训练。纯 HIL-only、SFT+HIL 和 C 组复合数据是**不同消融**，必须各自冻结 episode manifest。

单阶段 C 组的专家前缀没有保存在接管 raw 轨迹中；raw 中的策略前缀也不能冒充脚本专家。`robotwin_hil/overlay/scripts/prepare_one_stage_iwr_data.py` 对选定有效 HIL 先执行 `manifest`，再 `export-hil`，随后按同 seed 用原生脚本重跑阶段 1–5 的 `collect-expert`，最后 `validate`；两路数据分别保存来源与校验和。再经 LeRobot 转换与训练 loader 合成“脚本专家前缀 + HIL 后缀”复合源，与原 SFT 数据按外层批次 50:50 混合。复合源并不等于“30 条纯 HIL”。

## 继续训练的控制变量

训练入口 `robotwin_hil/overlay/scripts/run_hg_iwr_balanced_5050.sh` 会准备数据、计算混合源 `norm_stats` 并先做 loader smoke；其默认 `RUN_TRAIN=0`，只有显式 `RUN_TRAIN=1` 才正式训练。它还有 LoRA、EMA、batch、步数等可覆盖开关，**默认值不是所有历史实验的配置**。`train_iwr_balanced_5050.py` 需要基线 SFT 与 HIL/composite 的 repo ID、mix manifest、assets 和初始 SFT 参数。复现原 C 组时应单独核对是否使用原始 SFT `norm_stats`；不能把“重算混合 stats”的 launcher 默认行为误写成原 C 组。

与原 SFT 基线做可比续训时，明确 `discrete_state_input=False`、H10、原 SFT checkpoint 与其 `norm_stats`、全参数/LoRA、EMA、相同训练步数、batch 与 seed。曾有 `state=True` 的独立三任务 SFT，它们不能直接作为这轮 `state=False` 续训的初始化。检查训练日志和末步参数/资产完整；loss、有效 HIL 数和最终测评成功率分别报告，不互相替代。
