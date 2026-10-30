# 覆盖价值引导的动作偏置与长尾数据采集理论笔记

更新日期：2026 年 10 月 1 日。

后续诊断：2026 年 10 月 3 日的[覆盖分数与接管率诊断 TODO](COVERAGE_DIAGNOSIS_TODO.md)补充了分数基准约为 40 的 EMA 解释、固定观测动作敏感性测试及后续实验优先级。

本文整理 RoboTwin HG-DAgger 当前增强采样方法的理论定义、实际实现和验证边界。方法从冻结策略生成候选，用学习到的覆盖价值构造指数偏置分布，并以自归一化 Monte Carlo 估计该分布的动作均值。研究目标是提高有效专家恢复数据的采集效率。

当前方法的核心假设是：较低的覆盖价值与更容易进入长尾状态的动作相关。指数偏置可以严格保证降低其分布下的期望覆盖分数；是否提高长尾状态访问率、人工请求接管率和后续学习收益，需要分别验证。

## 1 问题与符号

冻结基础策略记为 \(\pi\)。在第 \(t\) 次真实策略决策时，观测为 \(o_t\)，包含三视角 RGB 和 14 维双臂关节及夹爪状态。策略一次生成一个长度为 \(H\) 的动作 chunk：

\[
A_t=(a_{t,0},\ldots,a_{t,H-1})\sim\pi(\cdot\mid o_t),
\qquad a_{t,h}\in\mathbb R^{14}.
\]

当前 \(H=10\)，每个动作是物理单位下的绝对关节及夹爪目标，顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪。\(t\) 表示外层策略决策，\(h\) 表示 chunk 内部动作索引；两者都不能直接等同于秒、保存帧或物理仿真子步。

为了统一与代码的记号，本文定义

\[
\boxed{B_\phi(o,a)\equiv C_\phi(o,a).}
\]

即 \(B\) 是覆盖分数，约定高分对应较充分的覆盖，低分对应潜在覆盖不足。若另行定义“越高越长尾”的分数 \(R=-B\)，相同算法可写成 \(\exp(+\beta R)\)；本文始终使用 \(\exp(-\beta B)\)。

在线偏置作用于同一个观测下的动作选择。因此，\(B\) 仅能区分不同状态还不够：若 \(B(o,a)=b(o)\) 对动作不敏感，固定 \(o\) 后所有候选的权重相同，偏置分布就退化回原策略。

## 2 当前覆盖价值的训练

训练数据为专家示范中的相邻保存观测及对应的动作目标。用 \(\tau\) 表示训练数据的帧索引，避免与线上每次执行一个 chunk 的决策索引 \(t\) 混淆：

\[
(o_\tau,a_\tau^E,o_{\tau+1})\sim D_E.
\]

对每个专家观测，用冻结策略独立采样 \(K\) 个 chunk，只取各自首动作作为候选 \(\tilde a_\tau^{(k)}\)。当前已训练版本的 TD target 是下一观测的策略候选评分均值：

\[
\boxed{
y_\tau=
1+\gamma\frac1K\sum_{k=1}^K
B_{\bar\phi}(o_{\tau+1},\tilde a_{\tau+1}^{(k)}).
}
\]

\[
\boxed{
\begin{aligned}
\mathcal L_B(\phi)
={}&\mathbb E_{D_E}\left[
\left(B_\phi(o_\tau,a_\tau^E)-\operatorname{sg}(y_\tau)\right)^2
\right]\\
&+\alpha\mathbb E_{D_E}\left[
\frac1K\sum_{k=1}^K B_\phi(o_\tau,\tilde a_\tau^{(k)})
-B_\phi(o_\tau,a_\tau^E)
\right].
\end{aligned}
}
\]

\(\operatorname{sg}\) 表示停止梯度。target 网络以 EMA 更新：

\[
\bar\phi\leftarrow(1-\eta)\bar\phi+\eta\phi.
\]

TD 项沿示范转移传播分数，保守项倾向于降低同一观测下策略候选相对于示范动作的评分。后继候选先经过 target 网络评分，再对评分取均值；先平均动作再评分是不同的目标。

历史上试验过使用 \(B_{\bar\phi}(o_{\tau+1},a_{\tau+1}^E)\) 的专家 bootstrap。当前线上 \(\alpha=1.0\) checkpoint 的配置明确为 `policy_next_action_mean`，本文使用的是后来的策略 bootstrap 版本。训练目标中没有额外的输出平方正则，AdamW 的参数 weight decay 仍然存在。

| 项目 | 当前配置 |
| --- | --- |
| 数据 | 450 条 SFT 示范，按 episode 划分为 405 train 和 45 val |
| 有效训练及验证转移 | 119,956 和 13,388 |
| 训练候选数 | \(K=4\)，只取各 chunk 首动作 |
| 网络输入 | 冻结 ResNet18 的三视角特征 1536 维，加 14 维状态和 14 维动作 |
| 价值网络 | 三层宽度 256 的 MLP，LayerNorm、SiLU，实数输出 |
| 当前训练系数 | \(\alpha=1.0\)、\(\gamma=0.99\)、EMA \(\eta=0.005\) |
| 优化 | batch 256、AdamW、学习率 \(10^{-4}\)、weight decay \(10^{-4}\)、梯度裁剪 1.0 |
| checkpoint | 10,000 steps，seed 42 |

状态和动作标准化仅使用训练数据。验证集是 critic 的内部验证集，基础策略已经见过这批示范。示范动作采用下一保存帧关节目标构造，属于近似转移；仅使用同一 episode 内具有真实后继观测的帧，不把文件末尾制造成终止状态或自循环。

这里的 \(B\) 是学习到的覆盖代理分数，不能直接解释成动作概率、密度、成功概率或长尾概率。常数 TD 信号 \(1\) 与保守项也不构成对这些概率的校准。线性保守项本身不提供一般性的有界性保证，实验训练的稳定性需要由损失、分数范围和数值检查记录。

## 3 指数偏置分布的定义与性质

在一个固定观测 \(o\) 下，定义当前在线使用的 chunk 评分：

\[
F_o(A)=B_\phi(o,a_0).
\]

用它构造 tilted 分布：

\[
\boxed{
q_\beta(A\mid o)=
\frac{\pi(A\mid o)\exp[-\beta F_o(A)]}{Z_\beta(o)},
\qquad
Z_\beta(o)=\mathbb E_{A\sim\pi(\cdot\mid o)}
\left[\exp[-\beta F_o(A)]\right].
}
\]

假设 \(0<Z_\beta(o)<\infty\)，则这是一个合法的条件概率分布。\(\beta\geq0\) 控制偏置强度，\(\beta=0\) 时 \(q_0=\pi\)。对原策略支持范围内的两个候选，偏置引起的相对密度变化为

\[
\frac{q_\beta(A_i\mid o)/q_\beta(A_j\mid o)}
{\pi(A_i\mid o)/\pi(A_j\mid o)}
=\exp\{-\beta[F_o(A_i)-F_o(A_j)]\}.
\]

因此低覆盖分数得到更大的相对权重。该构造保留原策略的支持范围，不能产生原策略概率为零的动作模式。

对于 \(\beta>0\)，它也有明确的优化解释：

\[
\boxed{
q_\beta=\arg\min_{q\ll\pi}
\left\{\mathbb E_q[F_o(A)]
+\frac1\beta D_{\mathrm{KL}}(q\Vert\pi)\right\}.
}
\]

证明来自恒等式

\[
\mathbb E_q[F_o]+\frac1\beta D_{\mathrm{KL}}(q\Vert\pi)
=\frac1\beta D_{\mathrm{KL}}(q\Vert q_\beta)
-\frac1\beta\log Z_\beta.
\]

它表示在降低期望覆盖分数的同时，以相对原策略的 KL 散度约束偏移。这是对采样分布的设计，不是由前述 TD 损失自动推出的唯一分布。

在允许交换微分和积分且相关矩存在时，对任意不随 \(\beta\) 改变的统计量 \(f(A)\)，有

\[
\frac{\partial}{\partial\beta}\mathbb E_{q_\beta}[f(A)]
=-\operatorname{Cov}_{q_\beta}(f(A),F_o(A)).
\]

特别地，

\[
\boxed{
\frac{\partial}{\partial\beta}\mathbb E_{q_\beta}[F_o]
=-\operatorname{Var}_{q_\beta}(F_o)\leq0.
}
\]

这是偏置机制本身可以保证的性质：增大 \(\beta\) 会降低偏置分布下的期望评分。该结论针对分布内候选的评分期望；它不直接约束平均动作执行后的评分。

令 \(p_L(o,A)\) 表示执行 chunk \(A\) 后、在预先约定的时间范围与后续控制规则下访问真实长尾状态的概率，则

\[
\frac{\partial}{\partial\beta}\mathbb E_{q_\beta}[p_L(o,A)]
=-\operatorname{Cov}_{q_\beta}(p_L(o,A),F_o(A)).
\]

因此，当真实长尾访问概率与覆盖分数在相关偏置分布下呈负协方差时，增大偏置才会提高该期望概率。在原策略上发现负相关只支持 \(\beta=0\) 附近的局部判断，不能直接推广到任意大的 \(\beta\)。

## 4 Monte Carlo 近似与当前执行规则

在线从同一观测的原策略分布中独立抽取候选：

\[
A^{(1)},\ldots,A^{(N)}\overset{\mathrm{iid}}\sim\pi(\cdot\mid o),
\qquad b_i=F_o(A^{(i)}).
\]

定义自归一化重要性权重：

\[
\boxed{
w_i=\frac{\exp(-\beta b_i)}{\sum_j\exp(-\beta b_j)}.
}
\]

候选已经由 \(\pi\) 抽取，所以权重中无需再乘一次策略概率，也不需要计算 flow 策略的显式密度。代码计算 \(\exp[-\beta(b_i-\min_j b_j)]\)，减去共同常数仅用于数值稳定。

对应的两个经验分布为

\[
\widehat\pi_N=\frac1N\sum_i\delta_{A^{(i)}},
\qquad
\widehat q_{\beta,N}=\sum_iw_i\delta_{A^{(i)}}.
\]

对统计量 \(f\)，自归一化 MC 估计为

\[
\widehat\mu_f=\sum_iw_if(A^{(i)}).
\]

在独立采样、有限且非零的归一化常数及相应可积条件下，\(\widehat\mu_f\) 随 \(N\to\infty\) 收敛到 \(\mathbb E_{q_\beta}[f]\)。有限样本下该比值估计一般有偏；常见的 \(N^{-1/2}\) 误差阶还需要相应二阶矩条件。

当前代码取 \(f(A)=A\)，执行整段加权平均：

\[
\boxed{
\bar A_t=\sum_{i=1}^Nw_{t,i}A_t^{(i)},
\qquad
\bar a_{t,h}=\sum_iw_{t,i}a_{t,h}^{(i)}.
}
\]

同一个候选权重用于该 chunk 的全部时间步和关节。当前方法可准确命名为“覆盖偏置分布下的 MC 动作均值控制”。增大 \(N\) 可以改善这个均值估计，与加权平均的设计目标一致。

若目标改成实际执行一次 tilted 抽样，则需要额外做

\[
I\sim\operatorname{Categorical}(w_1,\ldots,w_N),
\qquad A_t^{\mathrm{exec}}=A_t^{(I)}.
\]

这种重采样保留完整候选，属于另一种执行规则，当前线上版本没有采用。两者都能围绕同一 \(q_\beta\) 定义，但输出行为不同。尤其在多峰分布中，均值可能处于两个动作模式之间，未必是策略常产生的动作。

一般也有

\[
B\left(o,\sum_iw_i a_0^{(i)}\right)
\neq\sum_iw_iB(o,a_0^{(i)}).
\]

所以“候选加权评分降低”不能直接作为“实际平均动作降低覆盖”的证据。平均动作需要根据真实观测重新评分，其执行后的状态和任务结果还需另行记录。

当前在线路径只用首动作计算 \(F_o\)，没有对候选执行假想仿真分支。TD 训练赋予分数延续价值的设计意图，但这不等于已验证的整段 chunk 预测。若两个 chunk 首动作相同而后续不同，它们获得相同偏置因子；在理想分布中，有

\[
q_\beta(A_{1:H-1}\mid o,a_0)
=\pi(A_{1:H-1}\mid o,a_0).
\]

## 5 沿真实时间轴比较分布

研究动作分布差异时，应固定一组真实参考观测 \(o_t\)，在每个相同观测下比较

\[
\pi(A\mid o_t)\quad\text{与}\quad q_\beta(A\mid o_t),
\]

或比较它们的首动作边缘分布。横轴 \(t\) 是参考轨迹的真实策略决策顺序，而不是同一个 chunk 内的第 1 至第 10 个动作。

不同状态下的动作直接混在一起，会把任务阶段变化误认为同状态的多峰性。反过来，真正分别运行两种控制器之后，它们到达的 \(o_t\) 会逐渐不同，曲线差异同时包含动作选择与状态访问分布的变化。固定观测实验适合检验偏置本身，闭环实验适合检验实际采集收益。

偏置后的分布可能变宽、变窄，或只改变几个峰之间的概率质量。它的形状取决于 \(B\) 在动作空间中的分布，不能预先规定为“增强分布一定更宽”。例如，若两个对称侧峰都具有低覆盖分数，偏置可以增强两侧而保持均值不变；执行均值就无法保留这种双峰变化。

在小 \(\beta\) 且满足展开条件时，

\[
\frac{q_\beta(A\mid o)}{\pi(A\mid o)}
=1-\beta\left(F_o(A)-\mathbb E_\pi F_o\right)+O(\beta^2),
\]

\[
\mathbb E_{q_\beta}[A]-\mathbb E_\pi[A]
=-\beta\operatorname{Cov}_\pi(A,F_o)+O(\beta^2).
\]

这表明：分数需要在候选之间有变化，动作本身也需要与分数有协方差，偏置均值才会明显移动。分布发生变化时，均值仍可能因对称性抵消而几乎不动。

## 6 候选数与偏置强度的作用

训练系数 \(\alpha\)、在线偏置系数 \(\beta\) 和候选数 \(N\) 控制不同环节。\(\alpha\) 改变学习到的分数，\(\beta\) 改变给定分数下的理想偏置分布，\(N\) 改变对该分布的有限样本近似。当前分别为 \(1.0\)、\(10\) 和 \(4\)。

若原策略中某一动作区域的概率为 \(p\)，则 \(N\) 次独立候选中至少出现一次的概率是

\[
P_{\mathrm{hit}}=1-(1-p)^N.
\]

增加 \(N\) 可以提高发现稀有候选的概率，但不能产生原策略不支持的动作，也不能让对动作恒定的 \(B\) 获得区分能力。若所有候选的分数范围为 \(\Delta b\)，最大与最小权重之比为

\[
\frac{w_{\max}}{w_{\min}}=\exp(\beta\Delta b).
\]

当 \(\beta\Delta b\ll1\) 时，权重接近均匀。增大 \(N\) 是否能找到分数明显不同的候选，需要实测；理想 \(q_\beta\) 本身并不因 \(N\) 变大而变得更偏置。

共同分数平移不会改变权重，分数缩放则会等价地改变有效 \(\beta\)。因此不同 \(\alpha\) checkpoint 若分数尺度不同，直接共用一个 \(\beta\) 不代表偏置强度相同。当前选择 \(\alpha=1.0\) 是已有离线诊断支持的实验选择，尚不等于已证明最优的线上采集配置。

对同一组候选，可以记录

\[
\operatorname{ESS}=\frac1{\sum_iw_i^2},
\qquad
\operatorname{TV}(\widehat q_{\beta,N},\widehat\pi_N)
=\frac12\sum_i\left|w_i-\frac1N\right|.
\]

跨 \(N\) 比较时报告 \(\operatorname{ESS}/N\)。ESS 接近 \(N\) 表示权重接近均匀，较低 ESS 表示概率质量集中在较少候选上；两者本身都不是采集效果好坏的标签。这里的 TV 比较的是同一组候选上的两个经验分布，不是经验分布与连续真实分布的 TV。

理论上 \(N\to\infty\) 时，当前执行动作趋向 \(\mathbb E_{q_\beta}[A]\)。只有另行采用按权重重采样，执行候选的分布才会逐渐接近理想 tilted 分布。因此，“增大 \(N\) 能否解决问题”取决于目标是估计均值还是保留分布中的不同动作模式。

## 7 当前记录支持的结论

对 2026 年 9 月 30 日恢复采集目录中的全部 430 次 active enhanced 决策重新统计，配置为 \(N=4\)、\(\beta=10\)、\(\alpha=1.0\)，评分依据均为当前观测的候选首动作，执行规则均为完整 chunk 加权平均。

| 指标 | 中位数 |
| --- | ---: |
| 同观测候选最大减最小覆盖分数 | 0.0004444 |
| 最大候选权重 | 0.25053 |
| ESS | 3.999987，最大可能值为 4 |
| 加权与均匀经验分布的 TV | 0.0007618 |
| 首动作候选两两关节 RMS 距离 | 0.001509 rad |
| chunk 第 10 步候选两两关节 RMS 距离 | 0.007977 rad |
| 加权均值与均匀均值的首动作关节 RMS 距离 | 约 \(1.24\times10^{-6}\) rad |

关节 RMS 只使用 12 个机械臂关节，排除两个夹爪维度。候选距离统计包含每次决策的 6 对候选，共 2,580 对；第 10 步距离仅是 chunk 内候选差异的补充诊断，不是沿外层时间轴的分布比较。

这些记录说明当前四候选上的重加权很弱，实际 enhanced 均值非常接近 vanilla 均值。它们尚不能区分“原策略首动作分布较窄”和“critic 对已有动作变化不敏感”，也不能排除增大候选数后发现不同动作模式的可能性。四个样本不足以可靠恢复同状态的多峰密度。

## 8 实验验证与采集收益

首先验证固定观测下的偏置。沿不同任务阶段选取一组共同观测，缓存原始候选和评分，使用嵌套候选集合比较 \(N=4,16,64\)。对每个 \(t\) 报告动作分离度、分数范围、TV、ESS 比例、偏置评分期望和动作均值位移，同时画相同动作坐标上的加权与未加权分布。坐标投影和距离标准化应在比较前固定。

上述更大 \(N\) 是待做实验。当前在线接口仅支持 \(2\leq N\leq8\)，更大候选池需要在同一冻结观测上分批收集独立样本，并验证策略状态和随机性语义，不能直接视为现有线上配置。

其次验证 \(B\) 与真实结果的关系。对同一参考状态下不同分数的完整候选，通过独立、经过核验的仿真分支执行，观察预先定义的后续长尾访问事件或专家恢复需求。这里的仿真分支是离线验证手段，当前线上评分路径没有执行这些分支。不能仅用“低 \(B\)”给长尾事件打标签，再用同一个 \(B\) 证明检测准确。

最后进行 HG-DAgger 闭环对照：

| 条件 | 候选及执行 | 比较用途 |
| --- | --- | --- |
| Off | 原策略单个 chunk | 现有 HG-DAgger 采集基线 |
| Vanilla | \(N\) 个候选的均匀平均 | 测量多候选平均本身的影响 |
| Enhanced | 相同 \(N\) 下按 \(B\) 加权平均 | 相对 Vanilla 检验覆盖偏置贡献 |

三组使用相同策略 checkpoint、任务参数、初始 seed 列表、观测处理和执行配置。提前确定增强触发阶段或决策窗口，固定已启动 rollout 的数量，并保留中止和取消记录。人工自行决定何时按 `e` 的探索数据不能直接作为严格的随机对照证据。

主指标为“已启动且满足预定纳入规则的 rollout 中，至少一次人工按键请求接管的比例”。请求后是否取消阶段确认、是否实际执行专家恢复、是否最终保存有效 HIL，应分别记录。还应报告每条有效 HIL 的墙钟时间、实际恢复时间、有效保存率和固定训练预算下的后续策略成功率。

提高接管请求率只有在获得可用恢复数据且时间成本可接受时，才支持采集效率提升。当前方法的研究假设是：覆盖引导的动作偏置能增加有学习价值的恢复机会；该假设的闭环检验仍是方法成立的重要环节。

## 9 实现与证据来源

本文以本地及 Ubuntu 工作区的 `codex/coverage-enhanced-sampling` 分支、代码提交 `25593d3` 为语义依据，结合实际采集日志和加载出的 checkpoint 配置核验。已有 `ENHANCED_SAMPLING.md` 的部分段落仍描述旧版仿真分支评分，当前首动作评分以代码和日志中的 `current_observation_first_action` 为准。

- [训练公式及配置](TAIL_VALUE_V2_ABLATION.md)：策略 bootstrap、四组 alpha 消融、数据划分。
- [critic 与 TD 损失实现](../../../robotwin_hil/overlay/scripts/tail_value/model.py)：`critic_loss`、`loss_from_scores`、`POLICY_BOOTSTRAP`。
- [在线权重与动作聚合](../../../robotwin_hil/overlay/scripts/coverage_sampling/core.py)：`coverage_weights`、`DecisionSampler.select`。
- [当前桌面启动预设](../../../robotwin_hil/local_serving/run_hg_dagger_smooth.sh)：alpha 1.0 critic 路径及手动增强配置。

实际加载的 inference checkpoint 位于 Ubuntu：

```text
/hdd/robotwin-hil/outputs/sft_policy_eval_100/coverage_v2_alpha_ablation_step10000_20260924/inference_checkpoints/alpha_1p0.pt
```

其训练步数为 10,000，`bootstrap=policy_next_action_mean`，采集日志记录的 SHA-256 为：

```text
e925f81c4b84537930ecbdfbfaa69089c6622e444b1e80ef5f739a174420c055
```

第 7 节统计来自以下目录的 `sampling/episode_*.jsonl`，统计对象是全部 `active=true` 的 decision 记录：

```text
/hdd/robotwin-hil-enhanced-sampling/outputs/hg_dagger_collection_smooth_20260930_163549
```

本笔记中的 KL 变分解释、协方差恒等式和 MC 收敛表述是围绕上述明确定义的分布给出的数学分析；训练出的 \(B\) 是否满足长尾代理假设，须由第 8 节实验提供证据。
