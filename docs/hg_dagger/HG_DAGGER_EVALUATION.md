# HG-DAgger：测评协议

## 采集效率（真人接管）

末段采集的共同起点是脚本专家完成阶段 1–5 且 `stage6_handoff.ready=true`；前缀失败不进末段分母。`x` 放弃的 episode 不作为有效接管判断，但需单列其数量和整体工作量。分开报告：合格阶段 6 rollout 数、人工 `i` 请求数、专家恢复失败数、**已保存有效 HIL** 数、自主成功数、达到预设有效 HIL 数所需 rollout 与墙钟时间。有效 HIL 比例是 `有效 HIL / 合格且未 X 的阶段 6 rollout`；请求比例和有效保存比例不得混称“接管率”。

普通与增强组先固定任务/场景 seed、SFT checkpoint、专家前缀、人工按键规则，再按 baseline 完成报告配对。增强组重跑每个 baseline 合格 seed，并校验交接状态容差；超限记 `paired_prefix_excluded`。仅在两组同 seed 都合格且未 X 的集合上做配对差异，另外保留各自总尝试数与排除原因。K=1 普通组与 K=32 D 加权增强组比较的是**整个采样方案**；要单独归因 D，须补 K=32 均匀权重对照。操作者阈值与随机波动不能仅靠少量配对样本消除。

## 训练后策略成功率

新策略（普通 HIL、增强 HIL、Gaussian、纯 HIL-only 或复合数据）应在同一批冻结 seed 上做**无人工介入**的整任务自主测评。使用 [SFT 测评协议](../sft/SFT_EVALUATION.md) 的相同任务几何、提示词、`state` 配置、绝对关节 H10、频率字段、step limit 和无预筛口径；逐 seed 记录并检查 infra error。训练初始化、混合比例、stats、LoRA/全参及仿真补丁任何一个不同都要单列，不能只凭成功率归因采样方法。

还可用 `robotwin_hil/overlay/scripts/eval_one_stage_policy.py` 测脚本前缀后的阶段 6 **自主成功率**；它与真人接管概率、有效 HIL 保存率、整任务成功率是三个不同量。D 的同观测排序/轨迹 AUC 亦不能直接当策略成功率。文档只定义协议；具体结果应保留在各实验输出的 `summary.json`、逐 seed 记录和实验 manifest 中。
