# 专家支撑函数 D：训练方法

## 模型与目标

主要实现的图像编码器冻结，三路图像特征与标准化的 14 维 state、展平的 `10×14` 动作 chunk 拼接，送入 `ChunkCoverageCritic` MLP 输出标量 logit `B(o,a)`；高分代表更像专家来源。ImageNet ResNet18 与 R3M 是不同编码器实验，权重和特征缓存不可互换。模型入口见 `robotwin_hil/overlay/scripts/tail_value/chunk.py`，多任务训练入口见 `robotwin_hil/overlay/scripts/coverage_benchmark/train_screened_task_discriminators.py`。

同观测上的平衡二分类目标为

\[
L=\tfrac12\,\mathbb E[\operatorname{softplus}(-B(o,A_E))]
 +\tfrac12\,\mathbb E_k[\operatorname{softplus}(B(o,A_{\pi,k}))].
\]

它不会把失败策略轨迹当训练负例，也不把 D logit 当校准后的任务成功概率。训练侧拟合特征/动作标准化；源数据、checkpoint、特征编码器及 split 必须写入 manifest。

## 两种已实现的拆分

- 多任务筛查入口 `train_screened_task_discriminators.py`：每任务 30 条成功专家演示，五折 episode 级轮换，每折 18 train / 6 calibration / 6 test；默认 10k 步、batch 256、seed 42，记录每折 checkpoint、校准、OOF 与阶段指标。
- 对齐自训策略的 `train_d_support_sft50_test20_20261030.py`：源 train50 做五折 30 fit / 10 calibration / 10 OOF test；最终模型再用全部源 50 条训练，以 train50 的 OOF 分数校准，独立 test20 仅评一次。该入口显式要求原生 15 Hz、H10 和来自同一训练源的配对数据。

不同输入编码器、采样频率、源数据或策略 checkpoint 必须另起实验 ID。温度校准在独立 calibration / OOF 分数上选取目标 ESS/K（代码中的目标 0.5、β 上限 100），这是**采样集中度目标**，不是成功率阈值。线上 K=32 评分时还应记录实际 ESS/K，不能拿 K=4 离线值直接替代。

训练完成检查各折/最终 checkpoint、标准化参数、校准文件、模型与数据 SHA 一致性；之后按 [评价指标](D_EVALUATION_METRICS.md) 与 [独立测评流程](D_EVALUATION.md) 判定是否值得用于采样。
