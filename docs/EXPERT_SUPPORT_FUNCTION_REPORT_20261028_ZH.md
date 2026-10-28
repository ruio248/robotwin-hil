# 专家支撑函数训练与评测报告

日期：2026 年 10 月 28 日。范围：本轮公开 π0.5 三任务原生 H50 专家与策略动作判别器重训，以及与历史 handover_to_tray 方法的对应关系。

## 核心结论

本轮三个任务的专家数据准备、D 训练、独立策略轨迹评分和曲线生成均已完成。每任务 50 条有效专家 episode，其中 40 条用于训练、10 条用于来源判别留出验证；另有 50 条策略测试轨迹，逐任务均为 50 个唯一场景且记录中无基础设施错误。仅更新了判别器 D，公开策略 checkpoint 保持不变。

**来源判别有效，不等于失败识别有效。** 10K 模型的留出专家高于策略候选排序率分别为 88.9%、92.6%、81.9%，但三项测试的全轨平均 D 仍均表现为失败高于成功。控制为所有轨迹共有的查询前缀后，反向现象减弱，但六个新模型的共同前缀 AUC 置信区间均包含 0.5，尚不能认定具有可靠的成功预测或 HIL 预警能力。

本轮结果不支持“只要改成原生动作时间语义，成功与失败的分离就一定恢复”。同时改变了数据采样、训练划分和候选数，因此新旧版本比较是纠正后的流程对照，而不是单一因素的因果消融。

| 任务 | 运行位置 | 完成时间 CST | 有效专家 episode | 同观测锚点 | 测试成功与失败 |
|---|---|---|---|---|---|
| beat_block_hammer | new_power_3 | 20:53:42 | 50（训练 40／留出 10） | 668 | 35 成功／15 失败 |
| place_empty_cup | new_power_3 | 21:06:23 | 50（训练 40／留出 10） | 1249 | 23 成功／27 失败 |
| open_microwave | ubuntu | 21:23:26 | 50（训练 40／留出 10） | 7055 | 42 成功／8 失败 |

本报告的精确数字、95% 区间、训练及留出场景清单、测试完整性计数、checkpoint 哈希和远端来源哈希见 [机器证据快照](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/docs/evidence/expert_support_function_20261028.json)。完整测试逐场景记录仍在该快照注明的远端 evaluation_path 下。下表及正文的数字仅作显示舍入。

## 专家支撑函数的定义

当前实现将“专家支撑函数”具体定义为同一观测下的动作来源判别器：

- 正样本：任务脚本专家在观测 o 之后的完整动作 chunk，来源标签 y=1。
- 负样本：冻结的任务策略在同一个 o 上生成的完整动作 chunk，来源标签 y=0。
- 输出：D(o,A) 属于 0–1，数值越大代表越像本轮训练数据中的专家动作。

来源标签不是任务成败标签。即使策略给出的动作能成功完成任务，它在这个训练目标中依然是负类；脚本专家在自身轨迹上给出的动作是正类，但这不证明它可以从任意策略失败状态安全续接。

在相同观测分布、平衡正负类别、理想拟合的条件下，该目标对应的最优分类器可写成：

$$
D^*(o,A)=\frac{p_E(A\mid o)}{p_E(A\mid o)+p_\pi(A\mid o)}.
$$

这解释的是专家与当前策略的相对动作来源密度，而不是任务成功概率，也不是“从此状态是否能被专家救回”的概率。在训练没有覆盖的状态上，该公式不能赋予网络可靠外推能力。

## 数据如何构造

### 同一观测的完整 H50 配对

数据采集器执行原生专家脚本，在原生保存事件处记录三相机 RGB 和 14 维机器人状态。对每个有效锚点：

1. 正类动作取该锚点之后的 50 个原生保存事件关节状态，使用原生数据转换的 next-frame 约定。
2. 将该锚点已保存的同一 RGB、state 和任务提示词提交给冻结的公开策略，独立查询 4 次。
3. 得到 1 个专家 chunk 与 4 个策略 chunk，形状分别为 [50,14] 和 [4,50,14]。
4. 只有具有完整后续 50 帧的锚点才保留，不补齐尾部、不重复首动作、不做 ZOH 重排。

策略候选是在同一份已保存的观测上查询，并未把各候选先执行到不同状态再作比较。对动作、状态形状和有限值进行检查，三相机 R3M 特征为 1536 维。

**本轮训练锚点来自专家访问状态，而非策略访问状态。** 另采的策略 rollout 用于结局诊断，没有给这些策略状态补上经验证的专家续接标签后混入训练。因此“训练和测试同观测配对正确”与“覆盖了策略失败附近的状态分布”是两回事。

### 专家 episode 的数量和筛选

每任务从发布缓存 seed0、seed1 两个场景集合分别取得 25 条在本次重放中通过原任务成功判据的专家 episode，合计 50 条。seed0/seed1 是缓存集合标识，不是只运行数值为 0 或 1 的一个场景。未成功的专家场景及已识别的缺失抓取位姿异常有拒绝记录，不当作策略失败。

| 任务 | seed0 尝试／保留 | seed1 尝试／保留 | 被拒绝的专家场景 | 有效锚点总数 |
|---|---|---|---|---|
| beat_block_hammer | 27／25 | 29／25 | 6 | 668 |
| place_empty_cup | 25／25 | 26／25 | 1 | 1249 |
| open_microwave | 54／25 | 51／25 | 55 | 7055 |

微波炉为了获得 50 条有效专家轨迹共尝试了 105 个场景。因此这批数据不能用来声称专家脚本在任意初始状态、尤其策略偏离后的状态上有 100% 救回率。训练正类只纳入成功专家示范，与结局测试固定 50 场景、不删策略失败的分母是不同口径。

三个任务虽然各有 50 条 episode，有效锚点数相差较大。训练从训练侧所有锚点均匀有放回采样，所以更长、锚点更多的 episode 对训练贡献更大；评测指标则先在 episode 内汇总，再对 episode 等权。不能把“同样 50 条”理解为完全相同的数据量。

本轮是重新生成的原生示范，未证明其逐场景、逐动作等同于发布方训练策略时使用的原始专家数据。公开策略的训练 episode 数或发布说明，也不能替代数据身份校验。

### 动作时间语义

本轮跟随 RoboTwin 的原生 _take_picture 保存事件：子规划起点、周期控制采样、子规划终点。保留原生子规划边界可能出现的重复关节状态，但不人为插入重复动作。

- save_freq=15 是原生保存步距，不应直接解释为固定 15 Hz。
- 服务请求中的 frequency 字段、LeRobot 的 fps 元数据和仿真控制子步不是同一概念。
- 原生 H50 是 50 个动作目标，不能仅由这些元数据推断其固定物理持续时间。
- 采集器的 run_meta 中仍有继承的 sample_hz=50、physics_stride 字段；实际采样实现以原生保存事件回调及每 episode 的 native_save_events 记录为准。这些旧字段不能作为“专家按固定 50 Hz 抽样”的证据。

此前自训 stack_blocks_three 的 ZOH 对照已发现人为重复模式可泄露正负类来源；本轮三个公开任务之前的稠密物理网格采样是另一个时间语义差异，不能直接套用同一故障诊断。

## 网络和训练目标

### 网络结构

| 项目 | 本轮实现 |
|---|---|
| 视觉编码 | 冻结 R3M ResNet18，三视角各 512 维，拼接 1536 维 |
| 状态输入 | 14 维双臂关节与夹爪状态 |
| 动作输入 | 完整 H50 × 14，展平为 700 维 |
| 拼接维度 | 1536 + 14 + 700 = 2250 |
| 判别器主干 | 3 层 Linear 256 → LayerNorm → SiLU，再接 Linear 1 |
| 输出 | 主干输出 logit z，显式 Sigmoid 得 D |
| 归一化 | 仅训练 40 条 episode 的 state 与专家动作统计；标准差下限 0.05 |
| 任务模型 | 每任务独立 D，不共享任务参数；D 不直接接收文本特征 |

三项使用相同 SHA256 的冻结 R3M 资产：

`58d7bfcda80d81a4c54582e0dd3b0a73aaa2cafff111325eff729419ea989c1e`

### 平衡来源二分类损失

设每个训练锚点的专家 chunk 为 A_E，四个同观测策略候选为 A_π,k：

$$
D_\theta(o,A)=\operatorname{sigmoid}(f_\theta(o,A)),
$$

$$
\mathcal L_D=
-\frac{1}{2}\mathbb E_o\log D_\theta(o,A_E)
-\frac{1}{2}\mathbb E_o\frac{1}{4}\sum_{k=1}^{4}
\log\bigl(1-D_\theta(o,A_{\pi,k})\bigr).
$$

实现为 0.5 × [BCE(D_expert,1) + BCE(D_policy_K4,0)]。正负来源各占损失的一半，不会因为有 4 个负候选就把负类总权重变成正类的 4 倍。

显式 Sigmoid 加 BCE 与历史 softplus(-z_expert)、softplus(z_policy) 的平衡 logit BCE 在数学目标上等价；数值实现和饱和区稳定性可以不同。加入 Sigmoid 将显示值限制在 0–1，但不会自动把来源标签变成成功标签，也不保证概率校准。

| 超参 | 设置 |
|---|---|
| 优化器 | AdamW |
| 学习率／weight decay | 1e-4／1e-4 |
| batch | 256 个锚点 |
| 梯度裁剪 | 全局范数 1 |
| 随机种子 | 42 |
| 保存并评测 | 1,000、10,000 更新步 |
| 划分 | 50 episode 固定排列，40 训练／10 留出 |
| 五折或多训练随机种子 | 本轮没有；这是单次 40/10 划分、单训练种子的结果 |
| 归一化与特征 | 归一化仅来自训练侧；视觉编码器冻结 |

50 条独立策略测试轨迹的终局标签未进入上述损失；两个训练步数均完整报告，没有按测试 AUC 挑一个“最好断点”当作无偏最终结果。10 条来源留出 episode 是本轮 D 的留出集，不能据此证明公开策略从未见过相似场景。

## 来源判别和候选权重结果

### 来源排序与留出损失

专家排序统计同观测下 D_expert 高于各策略候选的比例，平分按 0.5 计；每条 episode 内取平均，再对 episode 等权。来源指标的区间使用 5,000 次 episode bootstrap，固定种子 42；完整区间见证据快照。

| 任务 | D 步数 | 训练集专家排序 | 留出专家排序 | 留出专家均分 | 留出策略均分 | 留出平衡 BCE |
|---|---|---|---|---|---|---|
| beat_block_hammer | 1000 | 91.0% | 81.6% | 0.665 | 0.433 | 0.588 |
| beat_block_hammer | 10000 | 98.0% | 88.9% | 0.788 | 0.138 | 0.491 |
| place_empty_cup | 1000 | 93.9% | 84.6% | 0.654 | 0.338 | 0.673 |
| place_empty_cup | 10000 | 99.1% | 92.6% | 0.641 | 0.235 | 1.558 |
| open_microwave | 1000 | 76.7% | 74.2% | 0.601 | 0.423 | 0.595 |
| open_microwave | 10000 | 90.2% | 81.9% | 0.673 | 0.381 | 0.529 |

杯子的排序率提高，但留出 BCE 从 0.673 恶化到 1.558。排序只要求正负相对次序，而 BCE 还惩罚错误且自信的预测，两者可以向相反方向变化。这是过拟合或置信度失真的警告，不是“训练越久所有方面都更好”。

锤子和微波炉的来源留出 BCE 有改善，但后面的轨迹结果表明，这并不保证成功／失败分离。

### 候选分数和权重诊断

本轮仅作固定 β=1 的离线诊断：

$$
w_k=\frac{\exp(-D_k)}{\sum_j\exp(-D_j)},\qquad
\mathrm{ESS}/K=\frac{1}{K\sum_k w_k^2}.
$$

低来源分数获较大权重，用于对应既有增强采样方向。加权动作位移是加权 chunk 与四候选均匀平均 chunk 之差，在 H50 和 12 个机械臂关节上求 RMS 后乘 1000，排除两个夹爪，单位 mrad。

| 任务 | D 步数 | 候选概率极差均值 | ESS/K | 加权关节位移 mrad |
|---|---|---|---|---|
| beat_block_hammer | 1000 | 0.0231 | 0.999975 | 0.0104 |
| beat_block_hammer | 10000 | 0.0278 | 0.999991 | 0.0149 |
| place_empty_cup | 1000 | 0.0294 | 0.999928 | 0.0186 |
| place_empty_cup | 10000 | 0.0278 | 0.999998 | 0.0146 |
| open_microwave | 1000 | 0.0407 | 0.999827 | 0.0263 |
| open_microwave | 10000 | 0.0616 | 0.999630 | 0.0469 |

表中 ESS/K 严格来说是“每 episode 的锚点 ESS/K 中位数，再对 10 条留出 episode 取均值”，不是把所有锚点合并后的总体中位数。候选极差是每个观测的 max(D)-min(D) 后作两层平均，未冒称与其他版本同名 D_B 指标完全一致。

ESS/K 接近 1，说明本轮这个固定温度下权重几乎均匀；同时动作位移很小。历史 handover_to_tray 使用原始 logit、冻结校准尺度和不同 β，不能拿本表直接判断其增强采样效果或复现历史 ESS 数字。本轮并未在线执行这些加权动作来测策略成功率。

## 独立策略轨迹的评测方法

### 测试集合

每任务 50 个发布缓存 seed2 场景，demo_clean、固定 H50，不启用 PACE 自适应 horizon；仍使用各自公开任务策略及其 norm_stats。

- beat_block_hammer、place_empty_cup：本轮重新运行并保存完整 H50 查询轨迹。
- open_microwave：复用已经完整记录的 seed2 H50 测试轨迹，经状态、任务、方法和 norm_stats 校验后评分。
- 所有任务保存实际查询的 RGB、state、完整 H50 建议动作和最终原任务成功标签。
- 后处理在每个已保存观测上补 3 个策略候选；候选 0 必须逐值等于原实际执行的建议 chunk。4 个候选中只有候选 0 来自实际执行路径，另外 3 个没有执行。
- 本轮每任务旧 D、新 1K、新 10K 都评相同轨迹及相同候选集合。
- 测试 scene_seed 与本轮 50 条专家训练／留出场景无交集。发布缓存本身经专家预筛，结果属于该条件分布，不与未预筛协议混算。

测试结局为锤子 35/50、杯子 23/50、微波炉 42/50。它们是冻结策略产生的诊断样本组成，**不是训练 D 带来的策略提升**。尤其微波炉 42/50 不是此前 seed0 筛选报告中的 31/50。

### 分数和 AUC 的定义

每条轨迹 i 有 T_i 次真实策略查询，每次查询四候选的概率均值为：

$$
s_{it}=\frac14\sum_{k=1}^{4}\operatorname{sigmoid}(z_{itk}),\qquad
S_i=\frac1{T_i}\sum_{t=1}^{T_i}s_{it}.
$$

先在轨迹内平均，再在成功／失败组内对 episode 等权。不是把全部帧混在一起平均，也不是先平均 logit 再做 Sigmoid。

始终以成功为正类、较高评分预测成功：

$$
\mathrm{AUC}=P(S_{\rm success}>S_{\rm failure})
+\tfrac12P(S_{\rm success}=S_{\rm failure}).
$$

AUC 小于 0.5 原样报告，不反转方向。95% 区间由成功、失败组内分别按 episode 重采样 10,000 次得到；查询点不是独立样本，不按帧 bootstrap。这些区间是探索性区间，未针对多任务、多个断点和多个窗口作多重比较校正。

另外同时保存 K1 实际建议动作和原始 logit 版本，避免将 K4 候选平均误称为实际执行动作的唯一评分。

## 成功与失败轨迹的结果

### 全轨概率均值

| 任务 | D 步数 | 成功平均 D | 失败平均 D | 成功减失败 | 全轨 AUC 与 95% CI |
|---|---|---|---|---|---|
| beat_block_hammer | 1000 | 0.501 | 0.542 | -0.041 | 0.417 [0.229, 0.610] |
| beat_block_hammer | 10000 | 0.395 | 0.578 | -0.183 | 0.290 [0.086, 0.526] |
| place_empty_cup | 1000 | 0.373 | 0.574 | -0.201 | 0.163 [0.061, 0.287] |
| place_empty_cup | 10000 | 0.271 | 0.637 | -0.366 | 0.074 [0.018, 0.155] |
| open_microwave | 1000 | 0.281 | 0.357 | -0.076 | 0.324 [0.185, 0.476] |
| open_microwave | 10000 | 0.313 | 0.376 | -0.063 | 0.348 [0.208, 0.494] |

三个任务两种新断点的全轨组均值均为失败高于成功。杯子反向最明显。锤子全轨 AUC 区间较宽，不能仅凭点估计断言总体反向关联。

### 轨迹长度对照

| 任务 | 成功平均查询数 | 失败平均查询数 | 更长预测成功的 AUC |
|---|---|---|---|
| beat_block_hammer | 3.49 | 8.00 | 0.014 |
| place_empty_cup | 3.39 | 10.00 | 0.000 |
| open_microwave | 15.67 | 30.00 | 0.000 |

失败轨迹普遍执行到更长的查询序列，成功轨迹提前终止。即使每条 episode 已等权，失败轨迹的均值仍包含成功轨迹不存在的后段状态和动作；这是时间窗口组成不同，不是简单的“长轨迹在组均值里权重更大”。

### 所有 50 条轨迹共有的查询前缀

共同前缀取本任务所有 50 条轨迹最短查询数，避免成功轨迹结束后退出组平均：

| 任务 | 所有轨迹共同查询数 | 1K AUC 与 95% CI | 10K AUC 与 95% CI |
|---|---|---|---|
| beat_block_hammer | 3 | 0.638 [0.478, 0.789] | 0.541 [0.358, 0.718] |
| place_empty_cup | 3 | 0.390 [0.237, 0.549] | 0.440 [0.283, 0.607] |
| open_microwave | 5 | 0.330 [0.161, 0.524] | 0.515 [0.330, 0.699] |

六个共同前缀区间都包含 0.5。锤子 1K 的 0.638 是一个待验证的正向点估计，不是已确认的失败预警效果；微波炉 10K 为 0.515，接近随机。杯子依然没有可靠的成功高分关系。

共同查询数仅对齐了记录数量，不保证语义阶段一致；前缀长度也是从本批轨迹最短长度得到的描述性选择。不能将本表直接称为“同阶段失败提前识别率”。

### 仅对实际建议动作评分

| 任务 | 1K K1 全轨／共同前缀 AUC | 10K K1 全轨／共同前缀 AUC |
|---|---|---|
| beat_block_hammer | 0.404／0.642 | 0.290／0.552 |
| place_empty_cup | 0.163／0.415 | 0.074／0.449 |
| open_microwave | 0.324／0.330 | 0.342／0.494 |

K1 与 K4 主结论相近，未显示“仅因补了三个候选，才出现反向分离”。K1/K4 不能混为一个指标；完整均值和区间都保存在证据快照。

### 相同测试输入上的旧版与新版 D

| 任务 | 旧 H50 D 全轨 AUC | 新 1K 全轨 AUC | 新 10K 全轨 AUC | 旧 D 共同前缀 AUC | 新 1K 共同前缀 AUC | 新 10K 共同前缀 AUC |
|---|---|---|---|---|---|---|
| beat_block_hammer | 0.356 | 0.417 | 0.290 | 0.503 | 0.638 | 0.541 |
| place_empty_cup | 0.095 | 0.163 | 0.074 | 0.316 | 0.390 | 0.440 |
| open_microwave | 0.208 | 0.324 | 0.348 | 0.438 | 0.330 | 0.515 |

这里“旧版 D”是前一轮各公开任务的 H50 判别器，不是 handover_to_tray 的历史 H10 disc_s42。以上各列统一先对候选作 Sigmoid 再聚合，输入测试轨迹完全相同。

旧版采用全部 50 条训练 episode、不同的动作采样和 K1；新版采用 40/10 划分、原生保存事件和 K4。不能把该表中任何变化全部归因于“原生采样”或单一网络修改。

## 曲线如何与历史 handover_to_tray 对齐

本轮已生成两种图，均在远端各任务 results 目录：

- normalized_progress.png：每条完整轨迹映射到 0–100% 查询进度，线性插值到 101 点，再分别对成功／失败组取平均。每个进度点保留全组 episode。
- common_prefix.png：仅画所有轨迹共有的查询前缀，全组 episode 始终参与。

这对齐的是历史绘图代码的“归一化整轨进度”和“共同前缀”两种横轴原则，不是将终止后的轨迹继续填零，也不是长尾逐步减少成功组样本。

还有两个不能混用的细节：

1. 本轮曲线阴影是 2,000 次 episode bootstrap 的组均值 95% CI；历史 report.py 的对应图使用轨迹之间的 25–75% 分位带。两种阴影含义不同。
2. 上述全轨 AUC 基于每条轨迹的原始查询均值，不是 101 点插值曲线的面积或插值后均值。显示横轴归一化，不会自动改写表格评分。归一化进度借用了最终轨迹长度，只能用于回顾性展示，不能作为部署时的预警时间轴。

完整对齐关系如下。

| 项目 | 历史 handover_to_tray 原 disc_s42 | 本轮三任务 |
|---|---|---|
| 策略 | 自训 SFT baseline | 发布任务专属 π0.5 |
| 动作 horizon | H10 | H50 |
| 视觉编码 | 原 ImageNet ResNet18；另有独立 R3M 消融，不能混名 | 冻结 R3M ResNet18 |
| 判别器主干 | 3 层 256 宽 MLP | 同类 ChunkCoverageCritic 主干 |
| 来源目标 | 平衡 logit BCE | 显式 Sigmoid + 平衡 BCE，数学来源目标相同 |
| 主展示分数 | 原始 logit | 本轮主表为概率，另保留 logit |
| 离线输入时刻 | 历史保存观测上的 K4 建议 | 实际 fixed-H50 策略查询点的 K4 |
| 增强权重 | 原始 logit + 校准尺度，历史默认 β=100 | 概率分数，固定 β=1 的离线诊断 |
| 轨迹横轴 | 归一化完整进度／共同保存帧前缀 | 归一化完整进度／共同查询前缀 |

因此准确说法是“沿用历史判别器主干和来源 BCE 目标，并对齐绘图横轴原则”，不是所有训练、输入、温度与评分口径都完全复刻。原始 logit 可以达到 ±10 或更大，不等于数值本身错误；Sigmoid 概率与 logit 的逐点排序相同，但先变换再跨候选／时间平均，可能改变 episode 的排序，不能默认它们有相同轨迹 AUC。

历史训练与图形定义见 [原 disc_s42 拟合代码](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/RoboTwin/scripts/coverage_benchmark/fit.py)、[历史绘图与统计代码](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/RoboTwin/scripts/coverage_benchmark/report.py)、[HIL 完整 chunk 判别器接入说明](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/docs/HIL_CHUNK_DISCRIMINATOR_20261012.md)。自训 stack_blocks_three 的 ZOH 诊断作为独立对照保留在 [原生动作与 ZOH 诊断](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/outputs/own_stack_blocks_three_d_verification_20261028/DIAGNOSIS_20261028.md)，其数值不混入本轮三任务表。

## 原因分析与下一步判定

### 当前证据支持的判断

- 同观测配对和完整 H50 输入已落实，来源留出排序并非随机。
- 输出已经在 0–1；数值是否像概率与能否预测成败是不同问题。
- 动作原生时序修正后，反向或弱分离依然存在，不能将所有异常都归因于 ZOH 或保存频率。
- 轨迹终止长度与阶段组成存在明显差异；共同前缀消除了部分时间窗口差异，但没有恢复可靠的正向分离。
- 杯子在 10K 的留出 BCE 恶化，提示更强的来源排序可能伴随更差的置信度。
- 训练只覆盖专家状态，对策略偏离状态缺乏正负配对约束，这是下一步需要直接检验的分布偏移，而不是已经被本轮定量证明的唯一原因。
- 完整未来 H50 不足的尾部锚点被排除，训练和测试的终局／停滞状态覆盖也可能不同；本轮尚未按阶段验证其贡献。

### 后续实验优先级

1. **先按阶段审计，不以最终成败替代局部动作标签。** 在同一真实任务阶段、相近时间窗内比较成功与失败片段，记录物体位姿、接触和阶段信息，避免把“还在操作”与“已经停滞”当作同一状态。固定查询编号不能冒称语义阶段。
2. **补策略访问状态上的可靠专家对照。** 从冻结策略实际 rollout 中抽取观测，只有专家能从同一完整仿真状态安全续接、且该未来动作可验证时才形成正类 chunk。单凭 RGB/state 文件不能直接臆造该状态的专家续接标签。
3. **将来源判别、成败关联和 HIL 事件识别分开验收。** 来源指标看 episode 留出排序和 BCE；成败指标看固定窗口／阶段的 AUC；HIL 指标需要真实接管请求和同阶段无请求负例，报告误报、召回与提前量。
4. **不根据测试图反复挑训练步数或取反分数。** 若需要选断点、窗口或温度，应另划选择集，并留一个不参与选择的外部测试集。本轮 1K/10K 全量报告作为探索性结果保留。
5. **重加权需单独校准和闭环验证。** 固定 β=1 的概率权重几乎均匀，不能直接拿它替代历史增强采样配置；之后应先验证候选分数动态范围、ESS、实际动作位移，再评估采集效率和救回质量。

以上是后续分析顺序，不表示已启动新采集、重训或 HIL 实验。现有证据尚不足以把这三个 D 部署为经过验证的自动接管触发器，也不能将专家来源高分等同于专家可救回率。

## 产物和复核入口

### 本机项目文件

- 本报告：[EXPERT_SUPPORT_FUNCTION_REPORT_20261028_ZH.md](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/docs/EXPERT_SUPPORT_FUNCTION_REPORT_20261028_ZH.md)
- 本报告证据快照：[expert_support_function_20261028.json](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/docs/evidence/expert_support_function_20261028.json)
- 任务配置与初始启动清单：[launch_manifest.json](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/outputs/three_task_native_h50_d_retrain_20261028/launch_manifest.json)
- 原生数据采集：[collect_native_h50.py](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/outputs/three_task_native_h50_d_retrain_20261028/code/collect_native_h50.py)
- D 训练与来源验证：[train_native_h50.py](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/outputs/three_task_native_h50_d_retrain_20261028/code/train_native_h50.py)
- 轨迹评分与口径：[evaluate_support.py](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/outputs/three_task_native_h50_d_retrain_20261028/code/evaluate_support.py)
- AUC 与 bootstrap 实现：[score_outcome_h50.py](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/outputs/three_task_native_h50_d_retrain_20261028/code/score_outcome_h50.py)
- 绘图：[plot_results.py](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/outputs/three_task_native_h50_d_retrain_20261028/code/plot_results.py)
- 主干结构：[ChunkCoverageCritic](/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling/RoboTwin/scripts/tail_value/chunk.py)

启动清单是启动时快照，其中 running_data_collection 不代表当前状态。当前完成结论来自远端 state.json、models/protocol.json、results/aggregate.json 及逐场景记录核验，相关值和哈希已放入本报告证据快照。没有为写报告而修改旧清单。

### 远端任务根目录

以下为服务器路径，不是本机副本。

| 主机 | 任务 | 任务根目录 |
|---|---|---|
| new_power_3 | beat_block_hammer | /data-training/robotwin-hil-lyt/outputs/three_task_native_h50_d_retrain_20261028/beat_block_hammer |
| new_power_3 | place_empty_cup | /data-training/robotwin-hil-lyt/outputs/three_task_native_h50_d_retrain_20261028/place_empty_cup |
| ubuntu | open_microwave | /media/ruio/hdd/three_task_native_h50_d_retrain_20261028/open_microwave |

每个任务根目录下：

- models/disc_s42_step_1000.pt、models/disc_s42_step_10000.pt：两个 D checkpoint。
- models/protocol.json、models/metrics_1000.json、models/metrics_10000.json：数据划分、模型哈希与来源指标。
- results/aggregate.json：结局统计和口径。
- results/native_1000_scores.json、results/native_10000_scores.json、results/old_h50_10000_scores.json：逐轨迹查询分数。
- results/normalized_progress.png、results/common_prefix.png：回顾性曲线。
- data/seed0_n25、data/seed1_n25：有效专家数据及来源记录。
- outcome_k4/manifest.json：测试同观测四候选数据校验信息。

微波炉复用的策略轨迹路径和各任务测试缓存哈希均记录在证据快照的 evaluation_path、evaluation_metadata 中。checkpoint 与原始 RGB／动作数据仍留远端；本次报告只新增本机文档和小体积 JSON 证据，没有下载策略模型，也没有修改、停止或重新启动任何训练及测评。
