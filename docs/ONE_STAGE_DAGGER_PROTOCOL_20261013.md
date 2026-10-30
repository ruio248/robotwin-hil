# One-stage HG-DAgger：末段配对采集协议

目的：只比较脚本专家完成前五阶段后、`guided_tray_placement`（第六阶段）的普通采样与判别器增强采样。它是**条件于专家前缀的末段实验**，不是整任务自主成功率或整任务接管率。

## 固定流程

1. 使用同一任务配置、`v2_promptfix_9999` checkpoint、30 Hz 控制配置及修正后的提示词。脚本专家按原 `play_once` 的动作顺序完成阶段 1–5，包括第 5 阶段的左臂松开与后撤；此时不保存策略/HIL 训练帧。
2. 仅当专家规划成功、右臂持杆、左臂已松开、物体未放妥且自动状态判定为第 6 阶段，才启动策略并开始录制。前缀失败单独记为 `prefix_failed`，不进入末段分母。
3. 普通组从末段首个策略决策开始单次采样；增强组从同一时点自动启用 K=32、完整 10×14 chunk 的判别器评分及加权动作。两个采集组不需手按 `e`，也不让操作者选择恢复阶段。
4. 操作者按 `i` 请求介入；策略当前 chunk 作废，由脚本按真实物理状态自动选择恢复分支并运行至结束。此协议不允许按 `r` 交还策略。`x` 放弃当前轨迹，`q` 退出会话。
5. “有效 HIL”要求发生介入、录有 HIL 帧、脚本恢复成功、任务最终成功且规划关节合法。失败的恢复和未介入失败都保留统计记录，但默认不保存到原始训练数据目录。自主成功单独计数。

## 配对方式与边界

先运行 baseline，直到 30 条有效 HIL 或达到最大 rollout 数；之后增强组读取 baseline 的 `session_*.json`，只重跑其中**完成了末段测试**的 seed，按原顺序跑完整个清单，即使增强组提前得到 30 条也不提前停止。baseline 的前缀失败与人为放弃仍计入其总工作量，不伪装成成对末段结果。

相同 seed 的脚本专家前缀在 SAPIEN 中不是逐数值可重放：冒烟测试观察到最多约 0.01 rad 的末段关节差。因此这里是“同 seed、同脚本、交接状态接近”的配对，而非位级一致的仿真快照。增强组自动检查左右关节最大差各 ≤0.02 rad、红杆位置各分量 ≤0.002 m、红杆四元数分量 ≤0.01、托盘位置/姿态各分量 ≤1e-6；超限记 `paired_prefix_excluded`。每条报告保存实际分量差。该容差不适用于声称精确同状态的因果分支实验。

主要比较 `valid_hil_events / eligible_stage6_rollouts`、达到 30 条所需尝试次数及墙钟时间，并同时报告自主成功、接管后恢复失败、前缀失败及配对排除数。高介入率本身不等于策略更好。普通单采样与 K32 增强采样是**完整流程比较**；单独归因判别器权重需额外 K32 均匀权重对照。

## Ubuntu 图形桌面启动

先核实独立策略服务 `127.0.0.1:18304` 运行且对应 `v2_promptfix_9999`。脚本只连接服务，不启动/停止服务，也不触碰 `18300`。以下命令须在 Ubuntu 远程桌面内的终端执行；SSH headless 冒烟不验证按键或 viewer。

```bash
cd /hdd/robotwin-hil-enhanced-sampling
ONE_STAGE_ARM=baseline \
MANUAL_OUTPUT_DIR=/hdd/robotwin-hil-enhanced-sampling/outputs/one_stage_baseline_20261013 \
bash local_serving/run_one_stage_dagger.sh
```

baseline 正常结束后，取其实际 `session_*.json` 路径，再运行：

```bash
cd /hdd/robotwin-hil-enhanced-sampling
ONE_STAGE_ARM=enhanced \
HIL_ONE_STAGE_PAIRED_REPORT=/hdd/robotwin-hil-enhanced-sampling/outputs/one_stage_baseline_20261013/session_实际时间.json \
MANUAL_OUTPUT_DIR=/hdd/robotwin-hil-enhanced-sampling/outputs/one_stage_enhanced_20261013 \
bash local_serving/run_one_stage_dagger.sh
```

默认从新 seed `100000` 开始、baseline 最多 200 次、目标 30 条有效 HIL。输出路径需是新的或通过 `HIL_RESUME=1` 恢复同配置已完成的会话；不要用旧全程 HIL 的输出目录。`HIL_DRY_RUN=1` 仅打印命令。不得把带 `--auto-intervene-step` 的冒烟数据并入正式采集。

## 已完成的冒烟

- Ubuntu headless、seed 100000：普通组和增强组均从专家前缀进入阶段 6，执行一个策略动作后以测试钩子触发介入，脚本自动恢复成功。
- 增强组首决策日志：`active=true`、`num_candidates=32`、`policy_inference_requests=1`、评分基于完整 chunk。
- 普通组启用录制与 `valid_hil` 自动保存：报告为 1 个合格末段场景、1 条有效 HIL、1 个保存 raw episode、65 帧。
- 从普通组会话报告读取 seed 的增强组配对测试通过；该次左右臂关节最大差为 0.0050/0.0074 rad，杆位置分量最大差 0.00071 m，均在预设容差内。
- 仅为受控软件冒烟，不是自然按键接管率或 30 条正式结果。图形窗口与按键仍需在实际远程桌面首次启动时核验。
