# 专家支撑函数 D：训练数据

## 配对样本的定义

D 学的是 `D(o, a)`：在**同一真实观测** `o`（三路 RGB、14 维状态、任务指令及可核验阶段）下，对一条专家未来动作 chunk `A_E` 与同一策略给出的 `K` 条候选 chunk `A_π,k` 评分。源标签是“专家/策略动作”，**不是**“成功/失败轨迹”，也不是人工 HIL 事件。策略候选只用于离线成对比较，生成配对样本时不执行这些负例来改变观测。

跨任务筛查入口为 `robotwin_hil/overlay/scripts/coverage_benchmark/collect_matched_pairs.py`，后续可用 `augment_matched_policy_candidates.py` 补 K=4，用 `annotate_pair_stages.py` 添加可验证阶段，并用 `encode_matched_pairs_r3m.py` 产生冻结编码。每个 episode 的 `pairs.npz` 至少包含 `state[n,14]`、`expert_chunks[n,10,14]`、三路 RGB、采样时间索引；`policy_candidates_k4.npz` 是 `[n,4,10,14]`；R3M 特征为 `[n,1536]`。阶段若无法由专家子计划调用点核验，必须标为 unknown，不从结果倒推阶段。

对自训策略的数据对齐尤其重要：专家演示必须是**该策略训练所用源数据**；在相同观测下用该策略的 checkpoint 重新采样候选。`train_d_support_sft50_test20_20261030.py` 对应的是原生 15 Hz、H10 的 train50 与另取 test20 独立专家演示；它要求两集合 seed 不重叠、每条专家成功、`source_frequency_hz=15`。不应拿别人的 checkpoint、不同 demo 来源或重采样到另一频率的动作 chunk 来声称同分布支持。

## 数据质量与防泄漏

- 冻结 `scene_seed`、episode ID、专家数据 manifest、策略 checkpoint/`norm_stats`、图像编码器权重 SHA、动作单位和时间对齐。图像、状态、专家 chunk 与策略 chunk 必须对应同一 `sampled_state_index` / `sim_step`。
- 以 episode 为单位拆训练、校准、OOF 测试及独立 test；同一个 episode 的相邻锚点不能跨集合。统计标准化与温度校准只能在训练/校准侧拟合。
- 验证 finite 数值、三相机 dtype、chunk 形状、候选独立性、专家成功与文件校验和。失败专家轨迹、`diagnostic_only=true` 或 `accepted_for_training=false` 的诊断记录不得混入训练。
- D 的数据只覆盖专家访问状态；能否识别策略即将失败或真人接管请求，需要另外的独立策略 rollout/HIL 标注验证。
