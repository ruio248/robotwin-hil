# 完整 action chunk Coverage 的冻结视觉编码器消融（2026-10-09）

## 结论先行

把原 ImageNet ResNet18 换成更强或更贴近机器人操作的冻结视觉模型，**没有稳定改善主判别器的专家/候选排序**。TD-B 用 DINOv2-S/14 在 seed 42 有改善，但 seed 43、44 的配对区间触及零；R3M TD-B 的训练种子波动很大。因此现阶段不能断言“视觉预训练不足是主要瓶颈”，更不能把离线分数差异解释为执行成功率提升。三种编码器在同一批已保存 HIL 轨迹上，都显示首次接管前候选均分下降；这只是回顾性诊断，受阶段和时间混杂。

## 协议与不可混淆的条件

- 从 `codex/coverage-encoder-ablation` 分支开展；原工作区的已有改动未清理或覆盖。本轮没有 commit/push，也没有启动新的仿真分支执行。
- 源是原有的 450 条 SFT episode 的完整 chunk 候选缓存：405 train、45 val；有效转移分别为 11,461 train、1,274 val。按既有 seed=42 episode 级切分，再把 45 val 固定分为 15 calibration（430 anchor）和 30 diagnostic（844 anchor）。这些是**内部分析**，不是全新独立 heldout。
- 三种编码器读相同三视角原始 RGB 和相同 `0,10,20,...` 宏观测。原 ImageNet ResNet18 直接复用已审计视觉缓存并重新做视角内 L2；R3M/DINOv2 从原视频按相同帧号解码。解码首帧与原帧 RGB 一致，旧/新 ResNet 特征的余弦检查通过；少量浮点差异来自运行环境。状态、专家 `[10,14]` 绝对关节目标、K=4 策略完整 chunk 候选及 episode split 原样保留。并未重新采策略动作。
- ImageNet ResNet18 权重 SHA-256 以 `f37072fd` 开头。R3M ResNet18 从[作者发布的模型](https://huggingface.co/surajnair/r3m-18)取得，权重 SHA-256 为 `58d7bfcda80d81a4c54582e0dd3b0a73aaa2cafff111325eff729419ea989c1e`。DINOv2-S/14 从[官方模型](https://huggingface.co/facebook/dinov2-small)固定 revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`，权重 SHA-256 为 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`。
- 输入预处理统一沿用原 Coverage 的 224 长边双线性缩放、居中零填充和 ImageNet 均值/标准差。每个视角特征单独 L2 归一化，再合并成 1536 维；DINOv2 的每视角 384 维右侧补零到 512 维。这保持评分头维度和三个视角的模长一致，但 DINOv2 与 ResNet 的架构和预训练语料同时改变，不能只把差异归因于“预训练质量”。R3M 也按该统一输入预处理评估，而非声称完全遵循其原生裁切流程。
- 所有编码器都**重新训练**相同 256 宽、三层评分 MLP。TD-B：α=1、γ=.99、H=10、20,000 steps；专家/策略来源判别器：10,000 steps、balanced BCE。两种目标各跑 seed=42/43/44。若直接拿旧检查点对比，会混入本轮视角归一化变化，因此旧分数不作同条件基线。
- 主要指标是逐观测专家 chunk 的分数超过四条同观测策略候选的比例，平局计半分；先对每 episode 平均，再对 30 个 episode 平均。区间为 episode 级 10,000 次 bootstrap。0.5 是随机排序基准，不是成功率。温度在 15 条校准 episode 上固定，β 上限 100；不能把不同方法都称为达到了相同 ESS/K。
- HIL 只使用既有 50 条已保存轨迹的 3,988 个全轨迹观测，其中首次人工接管前 1,233 个；使用原已保存 K=4 完整 chunk 候选，未调用策略服务。首次接管索引与原始 `control_mask` 的第一帧逐条核对一致。评分按首次接管帧对齐，以每条轨迹最初 10 个策略帧的均分为零偏移，再除以各模型在 SFT 校准集的**全局**评分标准差。晚期窗口 `-10..-1` 与早期窗口 `-20..-11` 比较，单位是校准标准差，不是原始 B 分。

## 结果

30 条内部诊断 episode 的专家相对候选排序，三个数字分别对应 seed 42／43／44：

| 冻结编码器 | TD-B 20k | 来源判别器 10k |
|---|---:|---:|
| ImageNet ResNet18 | .536 / .544 / .548 | .786 / .782 / .781 |
| R3M ResNet18 | .521 / .635 / .585 | .784 / .788 / .788 |
| DINOv2-S/14 | .583 / .572 / .565 | .794 / .782 / .781 |

同 seed 42、同 30 条 episode 的配对差（相对重新训练的 ImageNet 基线）：TD-B 的 DINOv2 为 `+0.047`，95% episode bootstrap 区间 `[+0.022, +0.071]`；R3M 为 `-0.015`，区间 `[-0.040, +0.011]`。判别器的 DINOv2 为 `+0.008`，区间 `[-0.003, +0.020]`；R3M 为 `-0.002`，区间 `[-0.013, +0.009]`。TD-B 的 DINOv2 在 seed 43/44 分别为 `+0.027`、`+0.017`，两次区间均包含零；R3M 则从负到正且幅度很大。不要把单一种子的区间当成覆盖训练随机性的总体置信区间。

seed 42 的诊断候选权重中位 ESS/K：TD-B 为 ImageNet `.985`、R3M `.926`、DINOv2 `.969`，说明 TD 加权仍接近均匀；判别器为 `.499`、`.458`、`.499`。在同一诊断集上，加权平均相对均匀平均的完整 chunk 机械臂关节 RMS 位移，TD-B 分别约 `.108`、`.203`、`.157` mrad；判别器约 `1.017`、`1.111`、`.974` mrad。位移是离线控制输出差异，不代表真实执行效果。

50 条 HIL 轨迹的晚期减早期候选**均分**（全局 SFT 标准差单位）都为负：TD-B 分别为 ImageNet `-2.76`、R3M `-3.56`、DINOv2 `-2.72`；判别器分别为 `-1.41`、`-1.39`、`-1.71`。每个比较都有 50 条轨迹共同参与。曲线显示的是这些被保存且发生过人工接管的轨迹上存在接管前下降；没有未接管的配对对照和在线分支复现，因此不能以斜率或绝对幅度宣布哪个编码器更适合控制。

## 文件和下一步

- 本地机器可读汇总：`outputs/coverage_encoder_ablation_20261009/summary.json`；每个 `eval_<encoder>_s<seed>/metrics.json` 有分 episode 分数和 bootstrap；HIL 各 `hil_score_<encoder>/metrics.json` 有逐对齐帧均值/四分位和窗口差。`report_manifest.json` 的 `complete=true` 表示**本次离线编码器消融**的文件齐全，不代表原两层 Coverage 计划的在线执行比较完成。
- 图：`outputs/coverage_encoder_ablation_20261009/expert_rank_encoder.png/.pdf` 与 `hil_takeover_encoder.png/.pdf`。训练检查点的 seed 42 推理文件已复制到本机 `outputs/coverage_encoder_ablation_20261009/checkpoints/`；seed 43/44 仍留在训练服务器。训练服务器原缓存、全部训练日志与模型保留于 `/data-training/robotwin-hil-lyt/outputs/coverage_encoder_ablation_20261009/`；Ubuntu 的 HIL 重编码缓存和评分保留于 `/hdd/robotwin-hil-enhanced-sampling/outputs/coverage_encoder_ablation_20261009/`。原 HIL 18300 服务未改动。
- 原两层方案的同状态执行比较仍有快照回放复现门槛未过；本实验没有绕开或放宽该门槛。若要判断视觉编码器是否提升**执行效果**，必须先解决那个门槛，或另外设计相同 seed 的安全在线 A/B 评估。本轮结果更支持先研究评分目标、温度与候选动作效应，而不是仅凭单一预训练编码器的离线峰值选型。
