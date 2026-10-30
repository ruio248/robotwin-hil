# 文档导航

文档按主题归档，**流程说明**与**历史实验记录**分开。流程说明只定义数据、训练与测评口径；带日期的报告保留当时的结果与限制，不能跨场景筛选、任务版本、频率或仿真补丁直接比较。源码使用方式见仓库 [README](../README.md)。

## SFT：专家演示到基础策略

按 [数据收集](sft/SFT_DATA_COLLECTION.md) → [训练](sft/SFT_TRAINING.md) → [测评](sft/SFT_EVALUATION.md) 阅读。

## 专家支撑函数 D

流程：[训练数据与同观测配对](expert_support/D_TRAINING_DATA.md) → [训练与校准](expert_support/D_TRAINING.md) → [评价指标](expert_support/D_EVALUATION_METRICS.md) → [独立测评流程](expert_support/D_EVALUATION.md)。

历史报告：[多任务专家支撑函数训练与测评](expert_support/EXPERT_SUPPORT_FUNCTION_REPORT_20261028_ZH.md)；其 [机器证据快照](expert_support/evidence/expert_support_function_20261028.json) 与正文分开保存。

## HG-DAgger：接管、采集与再训练

流程：[专家接管逻辑](hg_dagger/HG_DAGGER_TAKEOVER_LOGIC.md) → [执行与采集](hg_dagger/HG_DAGGER_EXECUTION.md) → [数据整理与训练](hg_dagger/HG_DAGGER_TRAINING.md) → [测评](hg_dagger/HG_DAGGER_EVALUATION.md)。

详细协议与历史诊断：[HIL/DAgger 操作手册](hg_dagger/HIL_DAgger_README.md)、[交接实现](hg_dagger/HG_DAGGER_HANDOVER.md)、[单阶段末段配对协议](hg_dagger/ONE_STAGE_DAGGER_PROTOCOL_20261013.md)、[跨任务接管恢复设计](hg_dagger/ROBOTWIN_DAGGER_RECOVERY_DESIGN_20261026.md)、[三任务失败重放](hg_dagger/THREE_TASK_FAILURE_RECOVERY_REPLAY_20261025.md)。

## 采样方法与消融设计

- [增强采样总览](sampling/ENHANCED_SAMPLING.md)、[完整 chunk 判别器接入](sampling/HIL_CHUNK_DISCRIMINATOR_20261012.md)、[共享前缀候选生成](sampling/PI05_SHARED_PREFIX_CANDIDATES_20261013.md)
- [DART 实现](sampling/DART_IMPLEMENTATION_20261016_ZH.md)、[DART 策略成功率扫描](sampling/DART_POLICY_SUCCESS_SCAN_20261016_ZH.md)

## 任务与场景

- [自定义 handover_to_tray 全流程](tasks/HANDOVER_TO_TRAY_FULL_PIPELINE.md)：从官方任务扩展、数据、策略、接管与 D 的对应关系。
- [RoboTwin DAgger 任务筛选](tasks/ROBOTWIN_DAGGER_TASK_SCREENING_20261024.md)：候选任务、筛选协议与历史进展。

## 历史研究：Coverage / Tail Value

这些是按实验主题保存的设计、诊断与结果，**不是**当前统一复现流程：

- Coverage 设计与状态：[两层方案](experiments/coverage/COVERAGE_CHUNK_TWO_LAYER_PLAN_ZH.md)、[实现状态](experiments/coverage/COVERAGE_CHUNK_IMPLEMENTATION_STATUS_20261009.md)、[诊断待办](experiments/coverage/COVERAGE_DIAGNOSIS_TODO.md)、[倾斜采样理论](experiments/coverage/COVERAGE_TILTING_THEORY_NOTE.md)
- Coverage 离线与编码器消融：[离线结果](experiments/coverage/COVERAGE_CHUNK_OFFLINE_RESULTS_20261009.md)、[编码器消融](experiments/coverage/COVERAGE_ENCODER_ABLATION_20261009.md)、[判别器大指标](experiments/coverage/COVERAGE_ENCODER_DISCRIMINATOR_BIG_METRICS_20261010.md)、[KNN 消融](experiments/coverage/COVERAGE_ENCODER_KNN_ABLATION_20261010.md)、[分数 40 诊断](experiments/coverage/COVERAGE_SCORE_40_DEBUG_REPORT.md)
- Tail / Value：[入口](experiments/coverage/TAIL_VALUE_README.md)、[V2 消融](experiments/coverage/TAIL_VALUE_V2_ABLATION.md)、[专家 bootstrap](experiments/coverage/TAIL_COVERAGE_EXPERT_BOOTSTRAP_REPORT.md)、[alpha 消融](experiments/coverage/TAIL_CHUNK_ALPHA_ABLATION_20261009.md)、[100 动作多样性](experiments/coverage/TAIL_COVERAGE_BASELINE_100_ACTION_DIVERSITY.md)、[候选多样性诊断](experiments/coverage/TAIL_COVERAGE_CANDIDATE_DIVERSITY_DIAGNOSIS.md)、[配对动作诊断](experiments/coverage/TAIL_COVERAGE_PAIRED_ACTION_DIAGNOSIS.md)、[终点奖励消融](experiments/coverage/TERMINAL_REWARD_ABLATION.md)

## 项目与通用协议

[上游隔离与兼容运行目录](project/UPSTREAM_ISOLATION.md)；[历史测评协议备忘](project/测评协议.md)。遇到互相矛盾的历史命令时，以对应实验的 manifest、源码版本和逐 seed 原始记录为准。
