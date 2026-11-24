# K32 训练候选覆盖消融

两项 train50/test20 均保留原 K4 并新增 28 个候选。原状态、特征、专家 chunk 与前四候选逐项核验一致。排序距离、按观察平均、lambda=1、10,000步、batch256、AdamW学习率1e-4与旧实验相同。K32阈值仅由训练候选距离差第25百分位计算，四个模型评测共用阈值及候选。

|任务|模型|K32 train 排序准确率|K32 test 排序准确率|原50 SFT轨迹AUC|
|---|---|---:|---:|---:|
|place_dual_shoes|bce_k4|62.07%|53.43%|0.758|
|place_dual_shoes|rank_k4_10k|64.16%|55.06%|0.471|
|place_dual_shoes|rank_k4_100k|76.56%|53.86%|0.918|
|place_dual_shoes|rank_k32_10k|65.75%|53.73%|0.611|
|place_empty_cup|bce_k4|62.71%|55.77%|0.996|
|place_empty_cup|rank_k4_10k|67.99%|58.76%|0.984|
|place_empty_cup|rank_k4_100k|71.97%|58.94%|0.796|
|place_empty_cup|rank_k32_10k|67.12%|57.59%|0.996|

K32 10,000步训练未显示独立场景候选排序提升。比较基于各任务同一批K32候选，不能与历史K4测试准确率直接相减。每个任务仅一次训练随机种子。本轮完成数据补齐及离线评测。更多候选带来更高计算量，batch观察数与更新步数保持一致。完整数据位于 matched_sft_k32，训练与报告位于 rank_k32_10000，各目录均位于 outputs/support_three_tasks_20261123。
