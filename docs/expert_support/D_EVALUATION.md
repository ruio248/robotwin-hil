# 专家支撑函数 D：测评流程

## 1. 冻结版本与同观测测评

先记录 D checkpoint/各折、图像编码器 SHA、标准化和校准文件、策略 checkpoint、任务与 scene split、动作 H、频率。`train_d_support_sft50_test20_20261030.py` 会对源 train50 给出 episode 级 OOF，并对完全未参与训练与校准的 test20 评一次；`train_screened_task_discriminators.py` 提供对应的五折筛查输出。核对所有 episode 都恰好落入一次 OOF 测试、split 没有重叠、每个锚点专家/候选动作共享 RGB/state/prompt。按 [指标定义](D_EVALUATION_METRICS.md) 报告源排序、D_B、ESS/K、候选重复率、加权动作位移与阶段分层。

## 2. 独立策略轨迹测评

在冻结、未用于 D 拟合的场景上，以真实策略 rollout 记录**每次策略查询时的**三路 RGB、state、实际预测 H10 和本轮终局成功。使用与 D 训练完全相同的图像编码器和动作表示对这些查询评分；`robotwin_hil/overlay/scripts/coverage_benchmark/score_outcome_traces.py` 提供发布策略 seed2、固定 H50 的实现，要求 launch/metadata complete、50 个唯一场景、无 infra error、D manifest/编码器 SHA 一致。自训 H10 策略应使用对应的 H10 轨迹评分协议，不把 H50 与 H10 当相同控制过程。

对每 episode 计算固定前缀均分和全轨均分，再用**本轮 rollout 的真实终局标签**计算 AUC；旧测评若没保存逐查询轨迹，必须重新跑并用新标签，不能借用旧成功列表。固定前缀不足的 episode 排除该前缀并报告两类剩余数；全轨结果同时报告查询数本身的 AUC、长度与阶段混杂。若仅有成功或仅有失败，AUC 不可定义。

## 3. 真实接管与下游检验

真实 HIL 检验需在同任务、同阶段记录人工 `i` 请求时间、未请求的对照窗口、X、恢复结果和终局。先评价 D 对请求事件是否有提前量和可接受误报，再做 K 候选均匀权重、D 加权和单候选等控制变量对照；最后才能讨论有效 HIL 采集效率。`diagnostic_only` 轨迹不得回流训练。报告协议、分母和不确定性，不在本流程文档中固定任何一次测评结果。
