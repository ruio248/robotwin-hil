# shoes 视觉编码器不一致审计

Cached original train/test features match ImageNet ResNet18, while outcome and online scorers used R3M18.

Original shoes train50/test20 candidate ranking metrics use mutually consistent cached ImageNet features, and remain valid.

Historical shoes SFT-outcome AUC and online support-guided evaluations used wrong visual encoder. expert100 shoes mixed cached ImageNet and freshly encoded R3M features; its results are invalid for coverage conclusions.

ImageNet ResNet18 resnet18-f37072fd.pth

原测试排序指标保持可追溯。旧在线成功数记录了实际运行结果，但对应错误编码器输入，不能作为正确部署的支撑模型效果。

## 正确编码器重新评分

|模型|旧R3M输入AUC|匹配ImageNet输入AUC|
|---|---:|---:|
|matched_sft_models|0.758|0.973|
|rank_distance_v1|0.471|0.748|
|rank_distance_100000|0.918|0.869|
