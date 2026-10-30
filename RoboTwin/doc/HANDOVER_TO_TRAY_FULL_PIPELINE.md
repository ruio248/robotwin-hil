# `handover_to_tray_v2_promptfix`：任务、数据、测评与专家支撑函数

> 记录截至 2026-10-30。本页描述本项目自建的 **`handover_to_tray`**，不是 RoboTwin 官方的 `handover_block`，也不是后来测评中用 `handover_block` 代替 `handover_to_tray` 的公开权重实验。下文把已经运行过的结果与建议/诊断严格分开。文内链接相对本文件，代码路径相对仓库根目录。

## 1. 从 `handover_block` 到 `handover_to_tray`

原型是 [`handover_block.py`](../envs/handover_block.py)：左臂抓红色长块、抬起、移至双臂交接位姿，右臂抓取，左臂松开并后撤，右臂放到蓝色目标上。它固定从左侧起步，末端成功判定主要检查目标附近的位置和右夹爪是否张开。

本项目另建 [`handover_to_tray.py`](../envs/handover_to_tray.py)，保留上述双臂交接动作骨架，但不是简单改名：

| 项目 | 本项目的改动 |
| --- | --- |
| 物体和目标 | 红色竖杆 `handover_bar` 与蓝色托盘底板；托盘两侧加静态挡边，使末段姿态误差可见。 |
| 场景分布 | 可选左/右起始臂和 `v1`/`v2_interior` 几何。正式 `v2_promptfix` 配置固定 `source_side: left`、`geometry_profile: v2_interior`；后者把起点和托盘限制在较容易规划的内部区域。 |
| 专家阶段 | 拆为 ①左手抓杆 ②抬升 ③移到交接位 ④右手抓杆 ⑤左手松开并后撤 ⑥右手对齐放入托盘、左手回原点。阶段快照保存关节、杆/托盘位姿和规划状态。 |
| 物理成功门 | 杆与托盘功能点的 `|x|,|y|<0.040 m`、`|z|<0.025 m`、水平朝向误差 `≤15°`，并要求**左右夹爪均张开**；不是“只要碰到蓝色区域”。 |
| 接管恢复 | `infer_recovery_state()` 从当前抓持、夹爪接触、物体位置和托盘误差判断分支；`recover_from_current_state()` 从当前物理状态重规划最短可行专家后缀，不重置场景，也不盲目重播完整 `play_once()`。掉出可恢复工作区或规划/抓持核验失败时拒绝保存。 |

任务配置见 [`handover_to_tray_v2_promptfix.yml`](../env_cfg/task_config/handover_to_tray_v2_promptfix.yml)：干净背景、三路 RGB（头部、左右腕部）、双臂 14 维关节状态/动作，原生 `save_freq: 15` 是**保存步距**，不能当成策略物理执行的 15 Hz。固定英文提示词为：

```text
Pass the red bar from the left arm to the right arm and place it in the blue tray.
```

提示词修复脚本 [`derive_handover_to_tray_v2_promptfix.py`](../scripts/derive_handover_to_tray_v2_promptfix.py) 只把错误重复的旧句子改成上句，并核对/硬链接数据 payload；**它不会改变已录制场景的几何**。`v2_interior` 是环境配置/采集时的选择，不能把“改 prompt 元数据”误写成“把 v1 物理场景变成 v2”。

## 2. 数据如何收集与保存

### 2.1 原始脚本专家示范与 SFT 基线

原始脚本采集入口 [`run_handover_to_tray_collection.sh`](../scripts/run_handover_to_tray_collection.sh) 先检查专家 smoke、冻结与训练不重叠的 dev/test seeds，再调用 RoboTwin 原生 `collect_data.py` 获取 **500 条成功的专家规划轨迹**，逐条验收 HDF5 的图像、关节和任务成功。该入口的 v1 转换脚本只将**前 450 条**转为 LeRobot v2.1，保留剩余 50 条作原生验证；不能把这 500 条等同于后来所有 `v2_promptfix` 文件的逐字节来源。

后续 `v2_promptfix` SFT/判别器所用的 LeRobot 数据集标识是 `ruio248/robotwin_handover_to_tray_v2_promptfix`。已核验的判别器缓存共 **450 条 episode**；其中 405 条训练、45 条内部验证（再拆 15 条温度校准、30 条诊断）。训练/诊断分割和每个原始文件的 SHA 记录在 [`fit/split.json`](../../outputs/coverage_benchmark_v1_20261009/fit/split.json)。数据集与场景代码必须同时核验：仅凭数据集名字不能证明其每个 episode 都是在当前 `v2_interior` 代码版本下生成。

SFT 使用 Pi0.5、`H=10`、三相机、14 维 joint、`discrete_state_input=False`；训练变换对 12 个机械臂关节采用 delta 表示，输出变换还原为仿真执行的**绝对关节目标**。基线推理检查点是 `v2_promptfix_9999`，评测/采集要同时锁定相应 `norm_stats`、提示词和任务配置，不能只换模型权重。配置模板在 [`port_openpi_config.py`](../../local_serving/port_openpi_config.py)，部署配置在 [`pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml`](../XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml)。

### 2.2 HG-DAgger 人工接管数据

全程采集入口是 [`hg_dagger_handover.py`](../scripts/hg_dagger_handover.py)，详细按键/导出说明见 [`HIL_DAgger_README.md`](../../HIL_DAgger_README.md)。策略 rollout 时监督者按 `i` 请求接管：尚未执行的策略 chunk 余量作废；脚本专家按当前物理状态选择阶段与恢复分支。全程模式允许按 `r` 交还策略；`x` 丢弃当前 episode，`q` 退出会话。每帧记录 `policy`/`hil` 控制来源和阶段；**策略动作不会伪装成专家标签**。采集先保存原始帧 `raw/episode_*/frames/*.pkl`、`episode.json`、`episodes.jsonl`、`session_*.json`，之后才离线导出 HDF5/视频/LeRobot。

用于比较普通采样与增强采样的**单阶段实验**另见 [`ONE_STAGE_DAGGER_PROTOCOL_20261013.md`](../../docs/ONE_STAGE_DAGGER_PROTOCOL_20261013.md) 和 [`run_one_stage_dagger.sh`](../../local_serving/run_one_stage_dagger.sh)：专家先执行阶段 1–5，只有右臂持杆、左臂松开且阶段 6 可接续时才交给策略；这段专家前缀不计入策略/HIL 训练帧。普通组从阶段 6 单次采样；增强组从同一阶段使用 K=32 完整 chunk 的低支撑度重加权。按 `i` 后本协议由专家恢复到终局，不再 `r` 交还。先收普通组，再从其正式 `session_*.json` 读取合格 seed 跑增强组；同 seed 的专家前缀只是近似配对，还要检查关节、杆和托盘交接快照容差。

单阶段的一条**有效 HIL**须同时满足：确有接管、存在 HIL 帧、脚本专家恢复成功、最终 `check_success()` 成功、规划关节合法，且未按 `x` 放弃。计数应分别报告进入阶段 6 的有效 rollout、接管请求、专家恢复失败、自主成功和实际保存的有效 HIL；前五阶段失败不进入阶段 6 分母。`30 条有效 HIL` 是保存目标，不等于 30 次 rollout，也不是接管率。诊断 seed/自动按键 smoke 不进入正式训练集。

## 3. 策略如何测评

**整任务自主测评**关闭人工接管和增强采样，用 `v2_promptfix_9999`、修正提示词、`handover_to_tray_v2_promptfix`、绝对 joint 完整 H10、频率配置字段 30、固定 `seed 31000–31099`，逐 seed 运行并读取任务原生 `check_success()`；不能用专家预筛剔除失败场景。历史记录为 **31/100**。命令与逐 seed 输出格式见 [`HIL_DAgger_README.md` 的 policy_eval_record 部分](../../HIL_DAgger_README.md#64-sft-策略评测记录policyevalrecordpy)。`--frequency 30` 是传给策略的字段，不应单独解释成实测 30 Hz 物理控制环。

**末段条件测评**先由脚本完成阶段 1–5，再在阶段 6 关闭人工接管，比较普通与 K32 增强采样。独立 `seed 100000–100019` 的无人干预复核结果见 [`RESULTS_ZH.md`](../../outputs/one_stage_policy_eval_dagger_seeds_20261014/RESULTS_ZH.md)：整任务从头运行 2/20；合格末段普通 7/20；增强组与普通组共同合格的 16 个 seed 上分别为 4/16 与 5/16。4 个增强交接因快照不符被排除，不能记作策略失败。两组使用独立策略采样噪声，这不是精确同状态的因果对照；该小样本**未证明增强组提升自主成功率**。

人工在场的历史末段采集统计是普通组 20 个合格交接中 14 次请求 `i`，增强组 18 个中 17 次请求 `i`。这是人的**接管请求频率**，不是无人干预失败率、有效 HIL 保存率或策略训练收益；两批数据的分母也不同。若评估“收满 30 条有效 HIL 的效率”，应再列出各组实际尝试数、X 取消、接管后失败、保存成功数和耗时，不能只报 `i` 的比例。

## 4. 专家支撑函数 D 怎么训练、怎么用于采样

这里的“专家支撑”是**同观测的动作来源判别**，不是成功概率或专家从任意坏状态救回的概率。对脚本专家访问过的同一观测 `o`（三相机 RGB + 14 维 state），正例是随后完整的专家绝对关节 chunk `A_E∈R^{10×14}`；从冻结 SFT 策略在**同一观测**独立采样 `K=4` 条完整策略 chunk `A_{π,k}` 作负例。候选没有先在仿真里各自执行。数据构造和 episode 划分见 [`coverage_benchmark/data.py`](../scripts/coverage_benchmark/data.py)、[`fit.py`](../scripts/coverage_benchmark/fit.py)。

部署的主模型 `disc_s42.pt` 使用冻结的 **ImageNet ResNet18** 对三路图像编码（每路 512 维，共 1536 维），拼接标准化的 14 维 state 和展平的 10×14 动作；[`ChunkCoverageCritic`](../scripts/tail_value/chunk.py) 为三层宽 256 的 Linear–LayerNorm–SiLU MLP，输出原始 logit `B(o,A)`，**高分表示更像训练集中的专家动作**。state/动作归一化只从训练侧估计，标准差下限 0.05。R3M、DINOv2 是后续冻结编码器消融，不能把它们的权重与主模型混用。

训练用 405 条 SFT episode 的 **11,461 个有效锚点**；15 条/430 锚点做校准，30 条/844 锚点只做内部诊断。每个锚点有 1 个专家正例、4 个策略负候选。seed 42，AdamW `lr=1e-4, weight_decay=1e-4`，batch 256，梯度裁剪 1，10,000 步；平衡来源 BCE 为

```text
L = 0.5 * mean(softplus(-B(o,A_E)))
  + 0.5 * mean_k(softplus(B(o,A_π,k)))
```

因此四个负候选整体只占一半损失权重。部署检查点与 SHA 见 [`fit/models.json`](../../outputs/coverage_benchmark_v1_20261009/fit/models.json)。校准集固定 `σ_cal=1.5994237859`、`β=100`（预设上限）：K4 目标 `ESS/K=0.5`，实得 **0.88484**，不能写成“已达到 0.5”。增强采样按 [`HIL_CHUNK_DISCRIMINATOR_20261012.md`](../../docs/HIL_CHUNK_DISCRIMINATOR_20261012.md) 对 K32 条完整 `[10,14]` chunk 打分：

```text
w_k = softmax_k[-β (B_k - min_j B_j) / σ_cal]
A_exec[h,j] = Σ_k w_k A_k[h,j]
```

低专家支撑分数被赋予更高权重，是**刻意增加困难动作采样**，不是“选最像专家的安全动作”。同一候选权重作用于其全部 10 步和 14 维；执行的是加权完整 chunk，而不是挑一个候选或只平均第一个动作。此处没有硬 KL 约束。在线默认微批大小及服务端是否共享视觉/KV 前缀应以实际运行版本和会话日志为准，不由 K=32 推断 32 次独立图像编码或一次无分批推理。

## 5. D 的好坏如何衡量：已有结果与不能推出的结论

| 检验问题 | 统计口径 | 已有 handover 结果 |
| --- | --- | --- |
| 能否识别专家动作来源？ | 同观测 `B_E>B_{π,k}` 的比例；先在 episode 内平均，再对 episode 等权，平局记半分。 | 主 `disc_s42` 在 30 条内部诊断、844 个观测上为 **0.7106**，episode bootstrap 95% CI **[0.6881, 0.7310]**；随机线 0.5。见 [`expert_rank/metrics.json`](../../outputs/coverage_benchmark_v1_20261009/expert_rank/metrics.json)。 |
| 候选分数能否真正改变采样？ | 同观测分差、全局校准尺度、ESS/K、加权与均匀动作 RMS、候选动作多样性一起报。 | 主模型 K4 校准 `ESS/K=0.88484`，未达到 0.5；只有分差或低 ESS 不能证明动作更有用。见 [`fit/calibration.json`](../../outputs/coverage_benchmark_v1_20261009/fit/calibration.json) 和 [`COVERAGE_CHUNK_OFFLINE_RESULTS`](../../docs/COVERAGE_CHUNK_OFFLINE_RESULTS_20261009.md)。 |
| 能否区分成功与失败轨迹？ | 每条自主轨迹在保存观测上平均 K4 原始分数，再对 31 成功/69 失败的 episode 算 AUC（成功为正类）。 | 历史 100 条 rollout 上，ImageNet 主模型全轨 AUC **0.7424**，R3M 消融 **0.7405**；但仅观测对照分别 **0.9874/0.9729**，轨迹长度/阶段可能解释大量分离。见 [`编码器大指标报告`](../../docs/COVERAGE_ENCODER_DISCRIMINATOR_BIG_METRICS_20261010.md)。 |
| 能否提前预测接管或失败？ | 固定前缀/同阶段比较，并用有真实接管的 episode 与无接管对照估计 AUC。 | R3M 回顾性全轨 AUC **0.7405**，但前 10/15 个保存观测的成功/失败 AUC **0.439/0.488**；接管前下降与自主轨迹伪窗口相比的事件 AUC **0.481**（95% CI **[0.385, 0.578]**）。没有证明可用的在线预警。见 [`hil_event_success_failure_validation/metrics.json`](../../outputs/hil_event_success_failure_validation_20261026/metrics.json)。 |

上述 ImageNet 主模型与 R3M 消融的输入编码器不同，不能跨行把绝对分数、校准尺度或 AUC 当成同一检查点的连续实验。2026-10-28 对原 ImageNet 结构另一次从头重训，在同 100 条历史轨迹上得全轨 AUC **0.766**、所有轨迹共同前 16 个观测 AUC **0.608**；它是另一个检查点的结果，不替换线上 `disc_s42.pt`，详见 [`success_failure_retrain.json`](../../outputs/handover_to_tray_disc_retrain_20261028/success_failure_retrain.json)。

**最终验收顺序**应是：先确认同观测配对和 episode 级留出排序；再看 K 候选的分数/动作分辨率与校准后实际权重；然后在独立策略 rollout 上报告整轨和固定前缀/同阶段的成功失败 AUC、仅观测对照；最后用真实接管请求与同阶段无请求负例检验提前量/误报，并用合格阶段的有效 HIL 保存数及独立无接管成功率验证闭环。全轨 AUC 有分离、HIL 前评分下降、或接管请求变多，任一项都**不单独**证明 D 能救回失败或提升训练后成功率。

## 6. 复核入口

- 任务与六阶段/成功门：[`envs/handover_to_tray.py`](../envs/handover_to_tray.py)、[`env_cfg/task_config/handover_to_tray_v2_promptfix.yml`](../env_cfg/task_config/handover_to_tray_v2_promptfix.yml)。
- 原始专家数据验收：[`validate_handover_to_tray_dataset.py`](../scripts/validate_handover_to_tray_dataset.py)；prompt-only 派生：[`derive_handover_to_tray_v2_promptfix.py`](../scripts/derive_handover_to_tray_v2_promptfix.py)。
- 交互采集与单阶段规则：[`hg_dagger_handover.py`](../scripts/hg_dagger_handover.py)、[`one_stage_dagger.py`](../scripts/one_stage_dagger.py)、[`run_one_stage_dagger.sh`](../../local_serving/run_one_stage_dagger.sh)。
- 策略逐 seed 测评：[`policy_eval_record.py`](../scripts/policy_eval_record.py)、[`eval_one_stage_policy.py`](../scripts/eval_one_stage_policy.py)。
- D 训练/评分：[`coverage_benchmark/fit.py`](../scripts/coverage_benchmark/fit.py)、[`expert_rank/metrics.json`](../../outputs/coverage_benchmark_v1_20261009/expert_rank/metrics.json)、[`coverage_encoder_discriminator_bigmetrics.py`](../scripts/coverage_encoder_discriminator_bigmetrics.py)、[`validate_hil_and_outcomes_20261026.py`](../scripts/coverage_benchmark/validate_hil_and_outcomes_20261026.py)。

本页不把旧文档中的“建议训练 2,500–5,000 步”、软件冒烟或远端排队状态写成已完成的正式实验结果；任何复现实验都应另存运行命令、模型/norm_stats SHA、seed 清单、逐 episode 记录和 `complete` 状态。
