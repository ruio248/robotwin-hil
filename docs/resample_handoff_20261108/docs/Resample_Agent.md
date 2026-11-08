# RESample Research & Engineering Handoff

> 项目：RESample —— 基于专家支撑估计的主动长尾态增强采样与 HG-DAgger。
> 编写日期：2026-11-07（Asia/Shanghai）；本地源码审计及远端只读快照约 13:05–13:10 CST。
> 本文只交接事实与后续工作，不授权接手者继续旧会话的训练、清理、停止进程或覆盖实验。
> 证据原则：当前本地实现 > 对应运行目录/manifest > 实验记录 > 通用旧文档；运行结果以其冻结版本为准。

状态约定：**VERIFIED**＝指定范围内有源码/产物/日志证据；**IMPLEMENTED**＝已写代码但缺充分执行验收；**PROPOSED**＝研究方案；**UNKNOWN**＝本次未确认。
“VERIFIED（源码）”仅证明分支/公式存在，不等于仿真成功；本次没有重跑单元测试、仿真或训练。文末 E 编号是证据索引。

## 1. Project Identity

- **VERIFIED / E01** 本地主工作树 `R=/Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling`。
- 本文中仓库相对路径均相对于 R；`P=robotwin_hil/compat/XPolicyLab/policy/Pi_05_RobotTwin`，`O=P/openpi`。
- Git remote：`ssh://git@ssh.github.com:443/ruio248/robotwin-hil.git`；网页：https://github.com/ruio248/robotwin-hil 。
- branch：`main`；HEAD：`11ddf23c150c171bfd26029886ab89f2ae0328a9`（2026-11-05 15:52 CST）。
- **当前工作树不等于 HEAD**：50 个 tracked 文件有差异，约 +1897/-1932 行；另有大量 untracked 实验目录。
- 审计时 `git diff --binary` 的 SHA256 为 `10b36b62d359d26041aebb44e58d79d1752dd7ab3d536d200d6aeed8b41ff2ba`；不包含 untracked 文件。
- 本文未 fetch；远端 GitHub 最新状态为 UNKNOWN。仅 clone HEAD 不能恢复本文描述的全部实现。
- `RoboTwin/` 是固定 upstream submodule：`4c5c753d66c54c3f6ce855ddf426bcaf3e327250`；扩展放在 `robotwin_hil/` 与顶层 `dagger/`。
- 当前 HIL 主入口是 `dagger.runner`；任务注册表实际包含 tray、stack、QR、cup、handover 共 5 项，不是 README 中的“仅 tray”。
- 当前科研四任务：`handover_block`、`rotate_qrcode`、`place_empty_cup`、`stack_blocks_three`。
- `handover_to_tray` 是自定义任务/历史方法验证平台；它的 v2_promptfix、v2_interior、H10/H50、state 设置必须分别标识。
- 用户口头的 hand_over_block、stack_block_three、place_cup_empty 等不是可直接用于 CLI 的正式任务名。

关键 Git 演进（commit 只描述当时状态，后续 dirty diff 仍需查看）：

| Commit | 意义 |
| --- | --- |
| `5df8e1f` | 抽出共享 DAgger runner 与逐任务 recovery adapters |
| `4b103a0` | 新任务 adapter、可恢复的 HIL 收集与进度 |
| `48ba992` | 修正 HIL 分段导出、统一接管入口 |
| `11ddf23` | 最短 HIL 段统一为 2 transitions，并加入真实 HDF5 集成测试 |
| `cc6bc18` / `3269bc1` / `06f0398` | 转向最小记录官方测评、隔离 upstream、检查仿真 assets |

历史入口边界：`scripts/eval_policy_xpolicylab.py` 当前只是 import compatibility，直接运行会拒绝；旧综合 ES evaluator 已退役。
当前工作树还删除了 `robotwin_hil/eval/ubuntu_dump_eval/` 下旧 queue/milestone 等脚本；不要从旧文档复制它们的命令。

## 2. Research Motivation and Objective

**PROPOSED（研究假设，不是已证明结论）**：普通策略 rollout 主要重复高概率、已熟悉行为；有信息价值的长尾状态不易在有限人工预算下出现。
RESample 希望在同一观测下产生多条策略 action chunks，偏向专家支撑较低但仍有可能纠正的动作，主动增加有价值的 HIL 机会。
这里的“长尾”应落到观测—动作分布、关键任务阶段、人工请求事件与最终再训练收益；不能只以动作离专家远或接管率高定义成功。

研究链条应分别验收：

1. 候选是否真实互异，是否来自同一 observation/state/prompt/checkpoint？
2. D 是否具有同观测动作分辨能力，而不只依赖图像/阶段或时间尺度捷径？
3. 增强采样是否到达有用的偏离状态，而不是纯粹破坏抓持、延长等待？
4. 专家/操作者能否从这些状态恢复，能否产出有效短纠偏片段？
5. 固定人工时间、HIL transitions 或 episode 预算后，再训练的独立成功率是否提高？

**VERIFIED / E02–E08** 当前四任务跑的是 `es-mode off` 的 HIL/DAgger 基线；尚无“四任务 H50 在线 RESample 带来性能提升”的证据。
D 是专家来源支撑评分，不是任务成功概率，也不是专家从当前状态恢复成功的概率；“低 D → 可恢复且值得采”仍待验证。
增强采样的用途是采数据；最终 policy 的无人工、无 ES 正式测评必须单独进行，不能拿接管后成功率代替自主成功率。

## 3. Algorithm Design

### 3.1 必须隔离的算法版本

| 版本 | 输入/目标/运行方式 | 当前定位与证据 |
| --- | --- | --- |
| PiperX 早期 RESample-style B pilot | 真机数据；Coverage C/尾部 B，TD 与保守项；事件检出 | 历史离线研究，不能混入 RoboTwin 实验；E21 |
| RoboTwin Tail Coverage v2 | 3-view features + state14 + 单个 action14；TD bootstrap | 保留研究实现；`tail_value/model.py::critic_loss`，E09 |
| 仿真分支 lookahead | 同观测 K chunks；分支执行并取 coverage 曲线最小值 | 历史 backend；不是当前 H50 D 或官方测评；E10 |
| H10 完整 chunk source discriminator | H10×14；balanced source BCE；在线输出 raw logit | 已有历史 sampler/HIL 接入；目前仍严格 H10；E06/E11 |
| 自训 SFT 同源 H50 单 D | 原 SFT train50 / 独立 test20；R3M；对外 sigmoid | handover/stack 有离线完成产物，尚未接入四任务在线 ES；E12/E13 |
| 当前四任务 HG-DAgger | H50/state=True；人工阶段选择；短 HIL 数据再训练 | 已训练 step299，测评部分完成；没有 D/ES 参与本轮采集；E16/E17 |

Tail Coverage v2 的代码目标为 `y=1+γ mean_k C_target(o_next,a_pi_next[k])`，默认不假造 terminal；可选 record-end reward 是另一显式设置。
损失 `MSE(C(o,a_E),y)+α E[mean_k C(o,a_pi[k])-C(o,a_E)]`；下一观测来自专家记录，并非执行候选得到的后继。
旧 first-action critic 不等于完整 chunk D；历史 B/C 命名不能据字母推断正负方向，须看具体 loss 和 sampler。

### 3.2 Observation、动作、horizon 与时间

- **VERIFIED（源码）/ E03/E06/E14** 策略使用 head/left-wrist/right-wrist RGB、机器人 state 和语言 prompt；恢复 adapter 另读 privileged pose/contact。
- Privileged 恢复状态不得注入策略 observation；D 当前 MLP 输入没有独立语言 embedding，prompt 通过配对策略候选与数据来源固定。
- 外部动作是有限值 `[left joint6, left gripper, right joint6, right gripper]`，共 14 维绝对关节/夹爪目标。
- 不要把驱动目标 `joint_action.vector` 无条件叫作传感器实测 qpos；官方 trace 还保存其他 state 字段，需按 schema 分别解释。
- OpenPI 内部 `use_delta_joint_actions=True`：12 个 arm 关节相对当前 state，2 个 gripper 维保持绝对；输出变换还原绝对动作。
- 导出端和执行端不得再次做 delta/反归一化；D 对物理绝对动作使用其自己由训练集拟合的 mean/std。
- 当前四任务 policy chunk 必须是 H50×14；H 是动作步数，K 是候选条数，二者不可混淆。
- `--frequency 50` 被传入策略 observation 的 `additional_info.frequency`；不是“50 Hz 仿真”或“每秒查询 50 次”的保证。
- `--save-freq 15` 是底层保存步距设置；LeRobot 导出标注 fps=15，实际物理时间仍要查 timestep/执行插值/帧间隔。
- 专家原生 15 Hz 的相邻 H50 约覆盖 3.33 s；在线执行 H50 的时间语义未证明与它相同，D 来源分类可能利用此捷径。

### 3.3 H50 单 D 的真实实现与数据

**VERIFIED / E12–E13** `ChunkCoverageCritic` 拼接 1536 维冻结三视角特征、标准化 state14、展平 H×14，3 层 256/LayerNorm/SiLU，输出标量 logit。
state/action std 下限 0.05；共享基类 `forward()` 返回 logit，实验 `SigmoidChunkCoverageCritic.forward()` 才返回 sigmoid。
R3M 与 ImageNet ResNet18 不可互换，encoder SHA、特征 cache、normalization、策略 checkpoint 和动作频率必须绑定。

平衡来源判别：
`L=-0.5 E log D(o,A_E)-0.5 E_k log(1-D(o,A_pi,k))`，`D=sigmoid(z)`。
数值实现用 `0.5*(softplus(-z_E).mean()+softplus(z_pi).mean())`，不是 sigmoid 后再 BCEWithLogits。
负例是成功专家观测下同一冻结策略生成的候选；不是失败 rollout，也不能来自另一任务/公开 checkpoint。
官方同源实验使用 SFT 原 train50、独立专家 test20、K4/H50、10k steps、batch256、seed42；test20 不拟合/校准/早停。
旧五折平均 D 的 outcome AUC 不属于全 train50 单 D；必须用同一个最终 D 重新计算。
来源排序率：比较同一观测专家分数与候选分数；不等价于 P(任务成功)、P(安全)、P(专家可救回)。

### 3.4 同一观测生成 K 个 chunks

实际调用：`dagger.runner → sample_absolute_joint_candidates → RPC get_action_candidates → Pi_05_RobotTwin/model.py → OpenPI`。
`update_obs` 固定同一 observation；候选的随机 Flow 噪声独立。不能把单条 chunk 的不同时刻伪装成 K 候选。
普通路径复制同一观测为 batch；客户端按 microbatch 切分，尾批大小 1 时调用 `get_action`。
`get_action_batch` 是不同环境观测的另一接口，不能与候选批混用。
**IMPLEMENTED + 局部实测 / E07** `OPENPI_SHARED_PREFIX_CANDIDATES=1` 才启用 `Policy.infer_candidates → Pi0.sample_actions_shared_prefix`。
共享前缀只编码一次视觉/语言并预填充 KV，再广播给 K 个 suffix；10 步去噪与普通路径共用积分器。
默认开关为 0；候选能力上限由服务报告，旧 JAX fraction=.30 路径通常限制 microbatch≤8，opt-in 路径允许 32。
H10/K32 单观测 benchmark 通过不证明 H50/K32 的显存、延迟或整个 HIL 回路已验收。

### 3.5 当前重加权公式、ESS 与动作聚合

**VERIFIED（源码）/ E06** 对低专家支撑评分 b_k：
`w_k=exp[-β(b_k-min b)/s] / Σ_j exp[-β(b_j-min b)/s]`。
`A_exec[h,d]=Σ_k w_k A_k[h,d]`，同一个 w_k 作用于所有时刻和 14 个维度（含夹爪）。
β≥0；β=0 或退化校准时均匀。有效温度可写 `T=s/β`（β>0）；不是概率置信度 temperature calibration。
这实现的是 weighted mean，不是 `Categorical(w)` 抽一个候选，也不是选择最低分 argmin。
非线性系统中平均动作可能不属于任何可行模式；`ΣwD(A_k) ≠ D(ΣwA_k)`，实际 blended chunk 会单独重新评分。
`ESS=1/Σw²`；`ESS/K` 衡量权重集中度，不衡量候选几何多样性，更不是最终策略性能。
日志还有候选 joint RMS、gripper RMS、score range/std、max_weight、相对均匀动作偏移、评分/候选耗时。
`Σw log(Kw)` 只是候选索引分布到均匀分布的 KL，**没有**输出动作分布相对策略的硬 KL 保证。
`coverage_benchmark/common.py::calibrate` 从训练/校准侧估计 scale，目标 median ESS/K=.5，β cap100；可能达不到目标。
历史 H10 logit 校准为 β100、s=1.5994237859，K4 ESS/K≈.88484（未达 .5）；不可直接搬到 H50 sigmoid。
stack H50 离线 train50 校准为 β≈12.35286、s≈.185987；只是该冻结 D 的离线分析参数，不是线上四任务设置。
`vanilla` 在窗口内是 K 候选均匀平均，并非普通单采样；`off` 才走普通单 chunk。
`gaussian` 对单候选/均匀均值的 H×12 arm entries 加各向同性噪声，夹爪不加噪；须单独控制候选数量。

### 3.6 触发机制与研究约束

已有 fixed decision window 与 `ManualSamplingWindow`（按 e 开启有限决策窗口）；索引是 policy decision，不是 physics step。
历史 tray one-stage 是专家前 5 阶段后，仅末段 policy/ES；不是整任务 autonomous rollout。
当前 adapter 的阶段判断用于恢复建议；“按阶段自动启用 D/ES”尚无四任务闭环验收，列 PROPOSED。
建议后续冻结阶段触发、β/ESS 下限、动作偏移预算、人工分钟数、可恢复率和技术异常率，再做配对消融。
至少分开 single-policy、K-uniform、K-low-support-weighted、Gaussian；不能把 K 平均与 D 评分的作用混在一起。
最终验收以独立再训练收益/数据效率为主，不以“制造更多失败/接管”为目标。

## 4. Current Implementation

### 4.1 Canonical runner 的实际状态机

**VERIFIED（源码）/ E02–E05** `parse_args → get_adapter → prepare_screened_scene → policy query → take_action`。
按 i 记录请求；在动作/查询边界丢弃剩余策略 chunk；`choose_recovery_stage` 确认后才切 source=hil。
构造 `recovery_stage_iterator` 不移动场景；每次 next 执行一个专家 primitive，返回 stage/done 事件。
按 r 交还策略，重新取当前 observation；不 reset 初始场景。按 x 作废当前局，按 q 退出并按实际流程处理缓存。
接管请求与真正进入专家模式分别统计；阶段确认取消也属于已发生请求，不应删除。
RPC 中轮询 viewer/键盘，但要等待 outstanding RPC 排空；不是硬实时取消，不能保证按键即刻中断底层运动。
运动规划失败会暂停、允许重选/交还/作废，不应当成基础设施崩溃后自动重启。

### 4.2 当前逐任务行为，不要照旧截图理解

| 任务 | 人工恢复/专用阶段 | 初筛默认 / 数据门槛 |
| --- | --- | --- |
| stack_blocks_three | 红绿蓝各 grasp/lift/place 共9段；手选优先；每 grasp 开满夹爪；预规划失败后同臂 world-z +5cm，最多5次再评估 | official-seen；仍是整局成功/关节合法门槛，不是 schema2 |
| place_empty_cup | OperatorPickPlaceAdapter 4段；grasp 预形0.6、左右抓取点2/0、arm-z抬杯；额外姿态/接触只诊断 | official-seen；schema2 连续 HIL 片段验收 |
| rotate_qrcode | 4基本段 + 手选5：固定臂位姿张爪→闭爪→抬起→转正放下；仅适用于牌已在指间，不是远处重抓 | official-seen；schema2 |
| handover_block | 6基本段 + 手选7：右臂重抓放偏红块并放蓝垫；手选4可重试接取；接触分类仅建议 | 仍继承 legacy-fixed；正式采集必须显式 official-seen；schema2 |
| handover_to_tray | 历史自定义前缀/恢复，支持 one-stage 与旧 ES；与四任务不同 | legacy-fixed / frequency30；保留专属历史门槛 |

stack 的抓取预规划检查候选抓取点与路径代价，不是已经切换成一种全新的 motion planner。
stack/cup/QR/handover 对新增规划路径检查 finite 与 absmax≤π，避免旧失败路径直接污染重试；fallback 仍可调用累计检查。
`planned_joints_legal` 不是完整碰撞/物理安全证明；不能由“几何上看似可达”推断 IK、起始碰撞、抓取姿态都可行。
stack 不再以 ambiguous/接触/稳定性否决手选，也不先 park_idle_arm；但 handover 阶段6仍保留官方的左臂 back_to_origin 与右臂 place 联合动作。
cup/QR/hand 的 unobservable_state、无效阶段、规划异常仍能中止；不能写成“所有保护代码都删了”。
`LiveObjectAdapter/PickPlaceAdapter` 的严格旧实现和 QR 文件底部 legacy factory 仍存在，但 canonical registry 选择新的类。
**关键**：四任务均 `supports_sampling=False`；runner 拒绝非 off；`SamplingConfig` 与 `OnlineChunkDiscriminator` 仍约束 H10。

### 4.3 初始专家预筛、指令与 seed

`screen_expert_scene`：关闭保存/视频/渲染，运行官方 expert play_once；接受条件是 plan_success ∧ check_success ∧ planned_joints_legal。
通过后重建同一 scene seed，再按官方 seen 生成/冻结 prompt；`scene_screening.jsonl` 保存 provenance，预筛不产生训练帧。
官方 min-trace evaluator 的专家筛选只有 plan_success ∧ check_success；它未额外做 joints gate，故 DAgger/SFT 并非完全同筛选。
runner 默认 seed-start=50000，seed-mode 默认 random；要连续场景必须显式 `--seed-mode sequential`。
`--episodes` 是已进入 policy 的 rollout 上限，预筛拒绝不计；max-scene-attempts 默认10×episodes。
续跑依赖 `inspect_collection`、session报告、screen history、raw元数据及目录锁；不按“已保存15条”推算下一个 seed。
同一任务训练/独立测试/采集 seed 必须查 manifest；不同任务同 seed 不代表同场景，也不构成跨任务数据泄漏的充分证据。

### 4.4 Recording 与训练资格的真实边界

raw-first：保存每帧 pkl、`episode.json`、control_mask、stage事件、prompt、seed；不在交互循环编码视频。
`saved HIL` 是保存且 hil_frames>0；`training-eligible HIL` 是通过任务 gate 的保存局；两者不是同一个计数。
默认最短连续 HIL＝3帧形成2 transitions；不得跨 policy/HIL 边界取 `state_t → action_{t+1}`。
schema2 只导出与 control_mask 精确匹配且 accepted 的段；失败/未知 motion 所在整段目前不部分放行。
`training_eligible()` 对旧无显式字段记录仍兼容放行；schema1/2 必须明确 accepted，diagnostic marker 会拒绝。
**当前遗留问题**：`assess_operator_hil_segments` 只认 grasp/lift/place_and_release/release/retreat，遗漏 handover 专用 substage 名。
**当前遗留问题**：stack 没设 segment_training_gate，runner 仍要求 final_task_success、整局 joints_legal、supervisor_success。
四任务都不再要求完整 recovery_ok；不能因此推断 stack 已取消所有整局门槛，或回写旧数据的 flags。
已完成的 all15 转换使用实验专用 diagnostic_all 路径；不是通用严格 exporter 默许所有保存数据训练。

接管率当前公式（E02）：
`N=记录中(policy_steps>0 或 manual_takeover_requests>0)的 rollout 数`；
`requested_rate=至少一次人工请求的 rollout 数/N`，`triggered_rate=至少一次进入专家模式的 rollout 数/N`。
接管失败和按 x 中止在满足上述有效条件时仍入分母；零步且无请求排除；中断缺失记录需单列，不能补造结果。

## 5. Code Architecture

下表路径中的 P/O 定义见 §1；状态只覆盖本次核查范围。

| 模块路径 | 核心职责 | 关键函数/类 | 调用关系 | 当前状态 |
| --- | --- | --- | --- | --- |
| `RoboTwin/envs/`、`task_config/` | 官方任务/机器人/场景 | play_once/check_success/setup_demo | evaluator/runner→env | VERIFIED upstream固定；运行副本另核SHA |
| `robotwin_hil/prepare_runtime.py` | upstream+compat+overlay+dagger 生成隔离runtime | build/_source_fingerprint | enter_ubuntu→build | IMPLEMENTED；不自动更新运行中进程 |
| `robotwin_hil/overlay/envs/_base_task.py` | HIL控制来源、保存、viewer与运动扩展 | take_action/take_dense_action | runner/adapter→env | 不等于正式测评base_task |
| `P/model.py` | Pi0.5适配、观测更新、候选RPC | get_action/get_action_candidates | websocket→OpenPI | VERIFIED源码及局部GPU诊断 |
| `O/src/openpi/policies/policy.py`、`models/pi0.py` | 输入/输出变换、Flow采样与共享前缀 | infer_candidates/sample_actions_shared_prefix | P/model→policy→model | IMPLEMENTED；H10局部实测 |
| `overlay/scripts/eval_policy_xpolicylab.py`（位于robotwin_hil） | import-only动作解码/候选辅助 | sample_absolute_joint_candidates | runner→RPC | VERIFIED；不是测评CLI |
| `robotwin_hil/overlay/scripts/tail_value/` | 历史TD、chunk模型、缓存 | CoverageCritic/ChunkCoverageCritic | offline fit / scorer | 历史版本并存 |
| `robotwin_hil/overlay/scripts/coverage_sampling/` | 在线评分/权重/窗口/日志 | DecisionSampler/create_online_scorer | runner→scorer→select | H10/tray，不是四任务H50 |
| `robotwin_hil/overlay/scripts/coverage_benchmark/` | 来源判别、校准、候选诊断 | calibrate/train_screened_task_discriminators | cache→fit→offline | 历史五折与单D需分开 |
| `dagger/runner.py`、`episode_protocol.py` | rollout、初筛、prompt、gate/进度 | main/prepare_screened_scene | top-level调度 | 当前主线 |
| `dagger/adapters/` | 任务阶段、当前scene恢复 | TaskAdapter/recovery_stage_iterator | runner→registry→adapter | 5任务，行为见§4 |
| `dagger/takeover.py`、`recording.py` | 输入、阶段确认、raw与资格 | choose_recovery_stage/finalize_episode | runner→helpers | 有严格/兼容路径差异 |
| `robotwin_hil/overlay/scripts/{export_hg_dagger_dataset,prepare_hg_dagger_dataset}.py` | HIL-only HDF5/LeRobot、norm | hil_runs/extract_source | raw→segments→datasets | schema2严格导出 |
| `experiments/dagger_hil_4task_lerobot_20261107/convert.py` | 本轮all15转换、段选择 | main | raw→LeRobot v3 | VERIFIED产物，诊断例外 |
| `O/scripts/train_iwr_balanced_5050.py` | SFT/HIL平衡batch、优化与保存 | 脚本训练入口 | run_train→balanced trainer | 本轮H50/64:64已运行 |
| `experiments/dagger_four_h50_5050_300_20261107/` | 四任务冻结编排、restore、验收 | run_train/run_eval/validate_checkpoint | 实验control→隔离runtime | 训练完成，测评未全完成 |
| `robotwin_hil/overlay/scripts/{eval_policy_xpolicylab_official_trace,official_policy_trace}.py` | 官方控制逻辑+被动trace | eval_remote_policy/记录器 | isolated env+policy→manifest | 当前正式评估入口 |

所有 `overlay/scripts` 都应在**生成/隔离 RoboTwin runtime**中解析 imports/cwd，不能仅从 Mac 仓库根直接执行。
旧 `hg_dagger_handover.py` 是兼容入口；不要绕过 canonical registry 去调用 QR 的 legacy factory。
`prepare_runtime` 发现 fingerprint 不一致会要求 --rebuild 并备份旧目录；`SKIP_RUNTIME_PREPARE=1` 只复用，不能让旧代码自动生效。
构建器现保留已有 assets/objects 与 assets/embodiments 有效软链；仍应查 objaverse/list.json 等资产是否可读。

## 6. Experiments and Verified Results

### A. 已完成、可追溯的结果

| 实验 | 设置与真实结果 | 证据及解释限制 |
| --- | --- | --- |
| H10共享前缀诊断 10/13 | RTX4090，K32，同一观测；4×batch8 1.3709s→共享0.1067s，32条互异 | E07；仅两次warm中位数/单观测，非H50端到端速度 |
| handover H50单D | train50/test20，专家排序率0.89396；K4概率分差中位数0.10508 | E12；来源排序，不是成功率 |
| stack H50单D | 排序0.91122，95%CI[.89298,.92998]；概率分差中位数.0594 | E13；候选RMS约5.91mrad，非强探索证据 |
| stack D真实轨迹评分 | 21成功/29失败；成功均值.31859、失败.38131；AUC=.42693 | E13；差CI跨0，未支持“成功分数更高” |
| handover D条件分析 | 29成功/21失败；均值.03028/.03887；AUC=.46798 | E12；排除专家异常seed100044，原manifest仍infra_error，非干净正式评测 |
| tray历史HIL-only | step99/stateFalse/H10，seed31000–31049，无专家预筛：32/50 | E22；与H50四任务不可混比 |
| tray历史Gaussian复合数据再训练 | step99/stateFalse/H10，同号测试范围：26/50 | E22；训练数据构成不同，不归因于采样单因素 |
| tray200演示SFT4999 | v2_interior自定义task/H50/stateTrue，正式45/50，专家拒绝3 | E23；不是官方handover_block，也不代表9999结果 |
| seed范围诊断 | stack从48400起28/50；QR从50100起41/50 | E24；不同seed池，不可与原50局直接作因果比较 |

### B. 本轮四任务 HIL 数据与 DAgger 训练

**VERIFIED / E16** 每任务最终15个 LeRobot HIL episodes（连续纠偏段），非15条完整专家轨迹。
Ubuntu 根 `/media/ruio/hdd/dagger_hil_lerobot_20261107/<task>_all15/dagger`；小manifest在本机 `outputs/dagger_hil_lerobot_20261107/<task>/`。

| 任务 | HIL episodes / transitions | 旧资格true源局 | 本轮处理 |
| --- | --- | --- | --- |
| handover_block | 15 / 899 | 4 | 11局主要为substage名字检查漏配，不能据false认定动作失败 |
| rotate_qrcode | 15 / 1165 | 15 | 资格与格式通过不代表训练一定提升 |
| place_empty_cup | 15 / 1211 | 14 | seed51007连续段含失败grasp后成功重试，实际质量混杂需保留 |
| stack_blocks_three | 15 / 1159 | 2 | 13局旧recovery_ok失败，其中11局官方终局成功；另外2局还有终局失败 |

空杯/交接块原有17段，依用户要求每源局择1段后变15；旧17版在 `superseded_multi_segment_17/`，raw不改。
LeRobot v3 为按相机打包的3个MP4，不是每episode各3个文件；视频数量不能推算episode数量。
manifest保留 `diagnostic_all/training_usable=false`；用户明确批准本轮训练all15，不得把这项例外改写成全局规则。
**文档/代码张力**：convert.py 的 warning 仍写 NEVER train，但本轮实验README明确记录用户例外；审计字段没有篡改。

**VERIFIED / E17** 本轮各任务 SFT50+HIL15、有效batch128（64:64，两个64样本梯度累积微批）、FSDP2、seed42、300更新。
Pi0.5 H50/state=True，全参；warmup25、LR1e-5→1e-6、EMA.999；从各自SFT4999 params初始化，非旧优化器原地续跑。
混合数据重新计算norm_stats；H50窗口不跨episode，末尾重复该episode最后动作。不要静默换回旧SFT normalization。
A800根记为 `A=/data-training/robotwin-hil-lyt/outputs/dagger_four_h50_5050_300_20261107`。
checkpoint为 `A/checkpoints/<task>/dagger_5050_h50_state_300/299`，norm资产为 `assets/sft_hil_5050_h50_20261107/norm_stats.json`。
四项 `A/control/<task>/checkpoint_validation.json` 都complete，独立恢复51参数leaves及156 train-state arrays。
当前4个299/train_state均已缺失：已按此前该批次授权在restore后删除；params/assets保留，不能精确恢复优化器。
params manifest SHA256（完整数据文件仍须以完整传输清单校验，单manifest不替代全包SHA）：

| 任务 | params/manifest.ocdbt SHA256 |
| --- | --- |
| handover_block | `c7223986e1e486d5eeedfe6e5048402a71cabf95d6288331e326abc18b3049d3` |
| rotate_qrcode | `0416f0ea2be7a34ad730089d57a78aee70472310bc7ef86467c842406ac72cf2` |
| place_empty_cup | `53c73e18d2caaf6c209978b9f261218c811c53ca4235bffd645f1ca171deeb2d` |
| stack_blocks_three | `245888882c42dc749fac00f1081e4b799d32890f7a951ec67784e39cae79a3c1` |

### C. 正在进行的测评 / 未完成事项

快照来源：远端 `A/evaluations/<task>_retry1/{control/state.json,formal/summary.json,formal/trace/manifest.json}`，约13:07 CST。

| 任务 | 旧SFT独立结果（仅参照） | 新DAgger299在A800的状态 |
| --- | --- | --- |
| handover_block | 24/50，PR510官方50数据4999 | complete：0/50，50正常失败，专家拒绝9，seed100000–100058，manifest零infra/d_ready=true |
| rotate_qrcode | 39/50，官方50数据4999 | formal running，快照8局其中1成功；不是最终成功率 |
| place_empty_cup | 32/50，官方50数据4999 | retry1 state/summary尚不存在 |
| stack_blocks_three | 21/50，官方50数据4999 | retry1 state/summary尚不存在 |

新hand结果已由state/summary/manifest交叉确认；本次未重新逐NPZ最终审计，不能宣称完成全部artifact验收。
旧handover的base_task SHA不同且逐局prompt不完整；另3项仍需逐局scene/prompt配对。0/50须重视，但不能直接断言DAgger算法导致退化。
旧SFT的4个summary路径在E18；它们不等于官方公开π0 Single/Easy表，本文不混入无法同协议配对的公开榜单数字。
首轮缺msgpack_numpy在smoke前失败，0正式局；retry1在隔离server_deps修复，旧目录保留。
formal protocol：固定upstream隔离运行副本、demo_clean、expert_check、seen、seed0派生100000、frequency字段50、完整H50、无人工；1smoke不计50。
成功终验：50唯一计数seed、0infra、非空查询、finite H50×14、执行数及3RGB/state一致、success+policy_failure=50、summary与官方_result一致。
失败视频working notes只描述采样帧所见，不是精确掉落因果；常见末态有源侧丢失、接取/交接掉落、偏离蓝垫、可能未松右爪。
主对话还在向Ubuntu直传4个299并准备Ubuntu测评；本次看到handover rsync运行，没有验收传完或Ubuntu正式结果。
Ubuntu目标根 `U=/media/ruio/hdd/robotwin-hil-github/outputs/dagger_four_h50_5050_300_20261107`；新 `run_all_evals_ubuntu.sh` 已实现，部署/完成状态仍须核实。
不得重复启动现有传输/测评，也不要把A800和Ubuntu局数合并。

## 7. Known Issues and Failed Attempts

1. **VERIFIED负面 / E21** 早期PiperX pilot：固定校准FPR约5%，heldout人工事件召回0/3；paper-style/TD-only AUROC .590/.574；小样本且离线，不是复现成功。
2. **VERIFIED负面 / E20** ImageNet TD20k同观测排序≈.5356，β到100仍ESS/K≈.9889，实际均值偏移≈0.000108rad；分数“有数值”不等于对动作敏感。
3. **VERIFIED负面 / E12/E13** H50单D来源排序较好，但两项整轨AUC均未支持高分更成功；成功/失败查询数差异极强，不能把事后长度效应当预警。
4. **VERIFIED限制 / E19** Gaussian探索pilot的sigma选择随有效seed交集改变（.01 vs .003），不是稳定HIL率校准；failure proxy≠真人接管率。
5. **VERIFIED运行故障 / E12** handover旧trace运行漏设JAX显存比例：同seed smoke约6374s，对照约96s；fraction=.55后三步取图7.204→.0384s，根因定位到渲染资源竞争而非D训练。
6. **VERIFIED中止记录 / E12** 修正显存后的旧D轨迹重测被用户停止于48/50（26成功）；manifest running/d_ready=false，不能当完整正式率或与其他批次拼50。
7. **VERIFIED基础设施差异 / E17** A800仿真cuRobo依赖及YAML中的Ubuntu绝对资产路径曾需隔离修正；同checkpoint跨机器也必须对齐环境。
8. **VERIFIED源码问题 / E04/E05** handover片段substage覆盖漏配；stack仍整局验收；旧schema兼容与新schema2规则不同，saved数不等于训练段数。
9. **IMPLEMENTED但未充分验收 / E04** 宽松恢复取消启发式拦截，并不保证抓取规划可行；5×5cm抬升有上限，失败应保留诊断，不能无限循环。
10. **UNKNOWN根因 / E17** DAgger handover 0/50可能涉及训练、norm变化、数据监督、执行/环境协议；目前没有单一因果证据。
11. **VERIFIED文档矛盾** 根README仍称仅tray；旧HG训练页的H10/stateFalse只适用历史实验；旧chunk文档否认共享前缀，当前代码已opt-in实现。
12. **UNKNOWN科研结论** 增强采样是否在同人工/数据预算下提升最终策略，当前四任务没有完成对照，不能宣传已证实。

## 8. Compute Resources and SSH

地址来自本机 `/Users/liaoruihao/.ssh/config`；13:05–13:09只读SSH验证。无密码、私钥或token写入本文。
**资源状态会变化；可SSH/显存空闲不等于获准占用。** Linux节点只查到相关进程不等于穷尽所有用户任务。
用户口头“new_power”在本机没有同名Host；本项目通常指 new_power_3，不能自行改连new_power_lrh。

| 机器 / SSH | 地址、账户、端口 | 当前已核硬件 / OS | 主要角色 |
| --- | --- | --- | --- |
| Mac本地 | 当前用户liaoruihao；无项目SSH服务假设 | macOS26.2，Apple M5/Mac17,2，10CPU，16GiB RAM；GPU统一内存容量未单查 | 编辑/文档/小JSON与选定视频 |
| Ubuntu / ubuntu | ruio@100.85.128.29:22；ubuntu-local→192.168.101.242 | Ubuntu22.04.5；i7-13700K，24线程，1NUMA；62GiB；1×RTX4090 24564MiB | SAPIEN交互、HIL、数据、策略推理与测评 |
| JG-ts / JG | ruihao@100.99.60.71:22；LAN192.168.101.11；JG-test另用test账户 | Ubuntu24.04.3；Ultra7 265K，20CPU；62GiB；1×RTX5090 D v2 24455MiB；NUMA未单查 | 历史PiperX真机/OpenPI；本轮RoboTwin环境未验收 |
| new_power_3 / node3 | lyt@61.49.53.31:60120 | Ubuntu22.04.5；双8458P，176线程/88核，2NUMA；约1TiB；8×A800-SXM4-80GB 81920MiB/卡 | 本项目训练与A800顺序测评 |
| new_power_lrh / node2 | lrh@61.49.53.31:60121 | Ubuntu22.04.5；同型号双8458P/176线程/2NUMA；约1TiB；8×A800 81920MiB/卡 | 其他QAM/Context-QGF工作；未查到RESample部署证据 |

node3/node2同公网IP不同端口、主机名与资源；不是共享同一工作目录的可互换账号。
两A800节点NUMA0 CPU0–43,88–131；NUMA1 CPU44–87,132–175；GPU↔NUMA拓扑未查，不能据编号猜绑核。

资源快照（不是预约）：

- Mac数据卷可用约36GiB；不要下载整模型/大query轨迹。
- Ubuntu GPU空闲23359MiB/util0%，RAM available57GiB，根61GiB；/media/ruio/hdd可用529GiB，实挂/dev/sda1。
- Ubuntu本次未见目标dagger runner/policy server匹配；有handover checkpoint rsync PID589762/589764。此快照不同于旧“QR常驻服务”记录。
- JG GPU空闲23768MiB/util0%，available RAM59GiB，根166GiB；本次进程筛查未见目标任务。
- node3 GPU0/2/4/5空闲约81138–81153MiB/util0%；GPU1空闲29928MiB/util77%，正在QR测评；GPU3空闲23822MiB。
- node3 GPU6/7虽空闲约76300MiB但util100%，不可当空卡；根分区0可用、/data-training仅34GiB，磁盘是关键阻塞。
- node2 GPU0–2空闲约68173MiB但util100%；GPU3–7空闲约36–38GiB且有负载；根23GiB、数据233GiB；不可抢占。
- A800队列PID3714195；QR runner3857291、server3857298、sim3861780，GPU1/18520。PID只供定位，接手必须核对完整命令。

路径与环境：

| 主机 | 仓库/数据/checkpoint | Python/Conda及限制 |
| --- | --- | --- |
| Mac | R；目标文档在Desktop/deepseek_task；与同目录QAM的Agent.md分开 | 本次只静态检查，没有安装依赖 |
| Ubuntu | /hdd/robotwin-hil-github；生成runtime=outputs/robotwin_hil_runtime/RoboTwin；数据盘物理路径/media/ruio/hdd | sim=/media/ruio/hdd/miniconda3/envs/robotwin_hil；policy=/hdd/Openpi-PACE-eval-20261022/.venv |
| Ubuntu数据 | HIL raw在仓库outputs/dagger/<task>/；all15与U见§6 | 每次写前findmnt确认/dev/sda1；不覆盖旧raw与结果 |
| JG | /home/ruihao/PiperX-ToolKit、/opt/openpi均存在 | Conda有piper/piper_lyt_test/lerobot等；本项目checkpoint、仿真assets与cuRobo可用性UNKNOWN |
| node3 | /data-training/robotwin-hil-lyt；A/data、A/checkpoints、A/evaluations | policy/train使用outputs/handover_block_official_sft_5000_20261101/env/openpi/bin/python；sim=A/eval_env/bin/python；代码A/runtime_openpi、A/eval_runtime |
| node2 | /data-training/lrh/context-qgf-*为其他项目记录 | RESample路径、Conda和模型均UNKNOWN，不复用他人环境 |

连接：本次 `ssh -o BatchMode=yes -o ConnectTimeout=12 new_power_3` 成功；Ubuntu跳板偶有banner timeout。
备用已在历史使用：`ssh -o BatchMode=yes -o ConnectTimeout=15 -J ubuntu -p 60120 lyt@61.49.53.31`。
alias是Mac配置；登录Ubuntu后再次 `ssh ubuntu` 不保证存在该alias，不代表Ubuntu本身坏了。
其他配置但本次未验证：new_server_4090=lrh@111.2.199.31:31024；5090=wy@100.70.178.32:22；new_server_rh_2=lrh@ssh-cn-huabei1.ebcloud.com:30634（ProxyJump JG-ts）。
以上额外机器的可用权限、OS、硬件、数据路径均UNKNOWN，不列作可立即调度资源。

## 9. Reproduction Guide

命令语法来自当前argparse/shell代码；本次未执行下面的写数据/训练命令。先取得当前授权、确认主线程任务、空闲GPU和未占用端口。
正式评估使用pristine upstream隔离副本+最小trace，不使用完整HIL overlay；记录task/base_task/config/params/norm/seed/prompt指纹。

### 9.1 先做只读核实

```bash
cd /Users/liaoruihao/Desktop/ruihao/robotwin-hil-enhanced-sampling
git status --short
git rev-parse HEAD
git submodule status
ssh -o BatchMode=yes ubuntu 'findmnt -no SOURCE,TARGET /media/ruio/hdd; df -h /media/ruio/hdd; nvidia-smi'
ssh -o BatchMode=yes new_power_3 'df -h / /data-training; nvidia-smi'
```

只读检查还需完整PID命令、ss监听、control/state和manifest；端口可连接不证明加载的是目标checkpoint。
`--ckpt-name` 是客户端标识/结果标签，不会替你切换已运行的policy server权重。

### 9.2 交互式 DAgger（Ubuntu，手动启动模板）

先确认现有服务确为handover_block对应模型，并给出端口和**新的**输出目录；不要在传输/正式测评占用时运行。
runtime必须已核对源码fingerprint；若不一致，退出旧runner后另行授权重建，不能盲用SKIP掩盖版本差异。
以下用环境参数防止误写历史目录；必要时用户设置 `RESAMPLE_POLICY_PORT` 和 `RESAMPLE_NEW_OUTPUT`。

```bash
cd /hdd/robotwin-hil-github
: "${RESAMPLE_POLICY_PORT:?填写已核实的handover_block服务端口}"
: "${RESAMPLE_NEW_OUTPUT:?填写新的采集目录绝对路径}"
eval "$(PYTHONPATH="$PWD" python3 -m dagger.desktop --target remote)"
ROBOTWIN_HIL_ROOT="$PWD" ROBOTWIN_HIL_SKIP_RUNTIME_PREPARE=1 \
HIL_VIEWER_BACKEND=preview HIL_VIEWER_IMAGE_SOURCE=head_wrist \
HIL_VIEWER_RENDER_RESOLUTION=720x405 HIL_VIEWER_WINDOW_RESOLUTION=800x450 \
HIL_VIEWER_MAX_FPS=10 \
bash robotwin_hil/enter_ubuntu.sh python -u -m dagger.runner \
  --task handover_block --episode-protocol official-seen \
  --host 127.0.0.1 --port "$RESAMPLE_POLICY_PORT" --policy-name Pi_05 \
  --ckpt-name handover_block_pr510_official50_4999 --task-config demo_clean \
  --frequency 50 --seed-mode sequential --seed-start 50000 \
  --episodes 100 --target-saved 15 --target-mode hil \
  --render-freq 10 --save-data true --es-mode off --output-dir "$RESAMPLE_NEW_OUTPUT"
```

续跑：原目录、原配置、原seed-start，增加 `--resume`；异常退出才按审计使用 `--resume-interrupted`，不要另建目录又要求沿用进度。
上述target15是当前训练资格计数，受§4遗留gate影响；不保证保存15局就导出15段。Windows App显示选择remote，不硬编码DISPLAY=:10。
runtime重建入口是 `python -m robotwin_hil.prepare_runtime --rebuild`（会备份/写目录）；本交接未执行，也不允许热替换运行中副本。

### 9.3 数据、训练、评估入口

- 通用HIL HDF5：runtime中 `python scripts/export_hg_dagger_dataset.py --raw-root RAW --output-dir NEW --mode hil --save-video true`；RAW/NEW须换实际路径。
- 通用LeRobot：`prepare_hg_dagger_dataset.py --raw-root RAW --output-dir NEW --baseline-root SFT --action-horizon 50`；还需匹配config-name/norm资产，不能用默认历史H10配置。
- 本轮转换：`experiments/dagger_hil_4task_lerobot_20261107/convert.py` 参数为 --repo-root、--raw-root、--output-dir、--mode eligible|diagnostic_all、可选 --one-segment-per-source；diagnostic_all不默认获准训练。
- 四任务训练：`experiments/dagger_four_h50_5050_300_20261107/run_train.py {prepare|smoke|train} --task TASK --root ROOT`；ROOT必须具备该脚本的data/input_checkpoints/runtime_openpi布局。
- A800实际Python见§8；原299/train_state缺失，脚本resume=False，不能用这条命令冒充从299精确续训。
- 单任务评估：`run_eval.py --root ROOT --task TASK --gpu GPU --port PORT --attempt NEWID`；负责启动专用server、1smoke+50formal及退出清理；不会覆盖已存在runner.pid。
- 现有批次不要重跑。A800顺序入口 `run_all_evals_a800.sh retry1` 已在跑；Ubuntu入口 `run_all_evals_ubuntu.sh ubuntu1` 已实现但须先验收全部checkpoint/runtime与资源。
- 最终NPZ审计入口 `audit_eval_npz.py --run-dir FORMAL --expected-count 50`；实际目录语义及可选输出参数以脚本main为准，禁止只看文件名就运行。
- 场景对照：`compare_formal_scenes.py --baseline OLD_FORMAL --candidate NEW_FORMAL`；有缺失prompt必须明确UNKNOWN。
- D入口：`experiments/handover_block_own5k_20261031/run_single_sigmoid_d_h50.py --experiment-root ROOT --source-manifest FILE --r3m-asset FILE --output NEW --steps 10000 --batch-size 256 --seed 42 --device cuda:0`。
- 最后一条是实验专用布局，不是四任务通用D命令；官方同源handover/stack各有自己的实验脚本，不能仅换task名复用旧own5k数据。

## 10. Next Tasks (P0 / P1 / P2)

下列均为 **PROPOSED**，不是本交接已修改/执行的事项；冻结当前批次，不中途更改协议。

### P0-1：完成现有批次验收，建立同机同协议 SFT→DAgger 对照

- 问题：hand299 A800为0/50；其余未全完；Ubuntu传输/评测未验收；旧baseline协议不完全相同。
- 代码/建议位置：E17的run_eval、audit_eval_npz、compare_formal_scenes、eval_server；新增独立对照目录，不改已有结果。
- 最小验证：先全文件SHA核对params/assets；用同机同scene/prompt比较4999与299的一组已冻结失败case，并核对输入变换/norm。
- 指标：50唯一场景、0infra、完整H50动作/执行数、匹配scene/prompt比例、独立成功数、逐阶段可观察失败；报告缺失。
- 完成条件：四任务两类模型有可比结果与CI；差异无法控制时明确不作因果结论，失败案例/版本证据可追溯。

### P0-2：统一短 HIL 片段资格，但不篡改本轮数据

- 问题：handover的substage覆盖漏配、stack整局gate、cup失败后重试同span，saved与eligible长期混淆。
- 代码/建议位置：`dagger/recording.py::assess_operator_hil_segments`、adapter事件schema、runner gate、两条export管线。
- 最小验证：用冻结raw回放资格判定；正常短纠偏后r、之后policy失败、规划失败后重试、x中止、多段一局各建fixture。
- 指标：跨controller边界0转移、所有合格段≥2transitions、失败motion不误放行、短有效段不因未done/后续policy失败误丢；报告拒绝原因。
- 完成条件：各任务使用显式动作事件语义而非名字白名单；新schema版本与迁移审计报告齐全；旧all15/flags不改写。

### P0-3：冻结可部署代码身份与资源占用边界

- 问题：大量dirty/untracked不在HEAD，运行副本可能滞后；node3根满/数据仅34GiB，不能继续无预算启动。
- 代码/建议位置：`prepare_runtime.py`指纹、实验protocol/command/validation manifest；部署清单。
- 最小验证：只读比较本机/运行目录关键文件SHA、模型+norm、资产链接；核对队列/rsync完整命令与剩余空间。
- 指标：每个run能绑定唯一源码快照和完整资产清单；无重复进程；保存峰值有明确空间预算。
- 完成条件：接手者可复现当前实现而非只clone11ddf23；用户确认资源与代码交付方式后才开展新run/提交。

### P1-1：验证 D 的用途，排除来源/时长捷径

- 问题：来源排序好但outcome区分无证据；H50时间语义与阶段组成有混杂。
- 位置：E12/E13的train/score脚本、官方trace、D缓存schema；不要直接改loss掩盖问题。
- 最小验证：核对实际dt与H50跨度；按episode/阶段分层、匹配查询时刻/长度；使用人工请求前的固定窗口标签。
- 指标：独立test20来源rank/候选分差；事件级recall@预定FPR、PR-AUC、提前量、分阶段CI；不以整轨AUC代替事件检出。
- 完成条件：一套冻结数据/encoder/policy/D/split/calibration manifest，结果可复算；若仍无分辨率，先停止上线ES假设。

### P1-2：受控接入 H50 RESample 与关键阶段触发

- 问题：当前sampler/scorer硬限制H10，四任务ES禁用；不能只删assert就上线。
- 位置：coverage_sampling/{core,critic,options}、P/model候选接口、runner sampling路径、adapter触发schema。
- 最小验证：保存观测离线K4/H50，检查online/offline分数与blended输出，再做单scene独立仿真；H50共享前缀另测内存/延迟。
- 指标：同观测一致、有限K×50×14、候选非重复、归一化只一次、同D-SHA与评分尺度；记录ESS/diversity/输出偏移/耗时。
- 完成条件：阶段窗口及β/预算预注册，off行为不变、故障可追溯；真实i/r/x/q与运动边界验收后才允许正式收集。

### P2：数据效率与最终收益消融

- 问题：更多接管不等于更好数据；K均值、额外算力、混合norm都可能混杂。
- 位置：新实验manifest/runner配置、balanced trainer、独立正式评估；不重用test挑β。
- 最小验证：single/K-uniform/weighted/Gaussian在固定同源policy与预算下比较，至少报告人工分钟与有效transitions。
- 指标：单位人工时间的有效HIL、恢复失败率、阶段覆盖、候选计算成本；统一SFT/HIL混合及训练步数后的独立成功率和CI。
- 完成条件：多随机种子、独立评测，结论区分效应与不确定性；负结果完整保留，不按好看的seed子集报告。

## 11. Rules for Future Coding Agents

1. 先核对用户当前授权和主线程负责人；本文历史命令/PID不是启动、删除、kill或提交的许可。
2. 不改官方upstream，不用完整HIL overlay冒充正式SFT evaluator；所有改动写入独立版本/manifest。
3. 保留dirty/untracked、raw、失败attempt和原flags；禁止git reset/checkout覆盖用户修改。
4. CPU单元测试、格式测试、headless smoke、GUI按键、50局仿真是五种不同证据，分别报告。
5. 修改adapter先测试动作与新路径合法性；不要重新引入完整suffix必须done的统一数据gate，也不默认取消所有技术校验。
6. 低D不是可恢复保证；不能用专家GT给policy增加未声明输入；人工阶段与自动建议分开记录。
7. 大数据/模型留远端，Mac只取小JSON和用户批准视频；下载、迁移、空间清理需明确范围。
8. 每个checkpoint必须带匹配norm与完整校验；train_state删除许可是特定批次例外，不永久继承。
9. 同任务/权重/seed/prompt/代码/预筛/H/state/动作时间未对齐，不直接比较百分比；infra不算策略失败。
10. 重建runtime、切换服务、清理空间前确认无人使用；不要热替换采集代码，也不占用“显存多但util100”的GPU。
11. 所有未来报告按VERIFIED/IMPLEMENTED/PROPOSED/UNKNOWN标注，引用具体产物，不能用计划填补结果。
12. 本交接编写仅修改本Markdown；不接手主会话automations，不对现有训练/测评发控制命令。

## 12. Evidence Index and Unknowns

索引是精简导航；路径均相对R（除标明主机/绝对路径）。代码行会变，优先函数名与SHA。

| ID | 一手来源 / 核查位置 |
| --- | --- |
| E01 | git remote/status/log/submodule；HEAD与diff SHA见§1；`README.md`、`docs/README.md` |
| E02 | `dagger/runner.py`：parse_args、begin_takeover（main内）、format_collection_progress、training_gate_checks、takeover_measurement |
| E03 | `dagger/episode_protocol.py`：screen_expert_scene、prepare_screened_scene、screening_qualifies_for_training |
| E04 | `dagger/adapters/{__init__,base,stack_blocks_three,live_object,rotate_qrcode,place_empty_cup,handover_block,handover_to_tray}.py` |
| E05 | `dagger/{takeover,recording}.py`；`robotwin_hil/overlay/scripts/hg_dagger_resume.py` |
| E06 | `robotwin_hil/overlay/scripts/coverage_sampling/{core,critic,options,calibration}.py`；`coverage_benchmark/common.py`在同scripts下 |
| E07 | `P/model.py`、`O/src/openpi/policies/policy.py`、`O/src/openpi/models/pi0.py`；`outputs/pi05_shared_prefix_diag_20261013/{legacy_frac030,shared_frac030,parity_frac030}.json` |
| E08 | `robotwin_hil/overlay/scripts/eval_policy_xpolicylab.py`（import-only）；`docs/sampling/ENHANCED_SAMPLING.md`退役声明 |
| E09 | `robotwin_hil/overlay/scripts/tail_value/{model,chunk}.py`；`docs/experiments/coverage/TAIL_VALUE_README.md` |
| E10 | `robotwin_hil/overlay/scripts/coverage_sampling/core.py::DecisionSampler.select` 的 legacy_branch_min 分支；同目录rollout模块 |
| E11 | `docs/sampling/{HIL_CHUNK_DISCRIMINATOR_20261012,PI05_SHARED_PREFIX_CANDIDATES_20261013}.md`；旧校准只用于对应H10模型 |
| E12 | `experiments/handover_block_official_d_20261101/README.md`；`outputs/handover_block_official_d_20261101/min_trace_exclude100044_20261102/{d_outcome_metrics,raw_trace_manifest}.json`；own5k下single_sigmoid脚本 |
| E13 | `experiments/stack_blocks_three_official_d_20261102/`；`outputs/stack_blocks_three_official_d_20261102/d_results_single/{manifest,source_metrics,resolution_metrics}.json`、`d_outcome_metrics.json` |
| E14 | `robotwin_hil/overlay/scripts/{eval_policy_xpolicylab_official_trace,official_policy_trace}.py`；`docs/sft/OFFICIAL_INTERFACE_TRACE.md` |
| E15 | `robotwin_hil/overlay/scripts/{export_hg_dagger_dataset,prepare_hg_dagger_dataset}.py`；相邻tests；`docs/hg_dagger/HG_DAGGER_TRAINING.md` |
| E16 | `experiments/dagger_hil_4task_lerobot_20261107/{convert.py,README.md}`；`outputs/dagger_hil_lerobot_20261107/<task>/{conversion_manifest,validation_report}.json` |
| E17 | `experiments/dagger_four_h50_5050_300_20261107/`；远端A/control/<task>/{mix_manifest,checkpoint_validation}.json及A/evaluations/*_retry1；本机同名outputs下analysis/review_notes.md |
| E18 | `outputs/handover_block_official_sft_5000_20261101/official_eval_20261101/formal/summary.json`；`outputs/{rotate_qrcode,place_empty_cup}_official_min_trace_eval_20261103/formal/summary.json`；stack对应20261102目录 |
| E19 | `outputs/gaussian_sigma_calibration_pilot_120000_120019_20261016/paired_calibration_report.json` |
| E20 | `outputs/coverage_encoder_ablation_20261009/summary.json`，scope明确internal diagnostics；`outputs/coverage_diagnosis_20261003/` |
| E21 | `/Users/liaoruihao/Desktop/ruihao/RESAMPLE_B_PILOT_EXPERIMENT_REPORT_20260911.md`：历史报告级证据，本次未重验原训练 |
| E22 | `outputs/iwr_one_stage_ordinary_hil_only_state_false_20261019/eval/formal/aggregate_summary.json`；`outputs/iwr_one_stage_gaussian30_cgroup100_state_false_fullparam_20261016/evaluation_31000_31049_20261017/formal/gaussian_fullparam_99/aggregate_summary.json` |
| E23 | `outputs/handover_to_tray_sft200_eval_20261106_retry1/summary.json`；`experiments/handover_to_tray_sft200_eval_20261106/README.md` |
| E24 | `outputs/stack_blocks_three_seed48400_eval_20261105/formal/summary.json`；`outputs/rotate_qrcode_seed50100_eval_20261106_retry2/formal/summary.json` |
| E25 | `/Users/liaoruihao/.ssh/config` 的Host/HostName/User/Port；本次ssh只读hostname/os-release/lscpu/free/nvidia-smi/findmnt/df/ps；无持久化资源日志，快照见§8 |

**UNKNOWN 汇总（接手先核实，不自行补齐）**

- GitHub当前HEAD、untracked实验的完整可移植快照；各远端runtime是否与当前本地dirty源码逐文件一致。
- 四个299传Ubuntu的最终退出码/全包SHA、Ubuntu队列是否已经开始；A800后续QR/cup/stack最终结果与严格逐NPZ终验。
- handover 0/50的因果根源；旧无prompt基线的完全配对条件；视频中未采到的精确抓持/掉落时刻。
- H50共享前缀K32真实GPU预算与端到端交互延迟；候选均值是否处于有效动作模式。
- D的事件级提前预警、专家可恢复性、可靠阶段标签；专家/在线H50的真实时间尺度等价性。
- 四任务H50增强采样闭环、同预算数据效率与再训练收益；目前都不是已完成成果。
- 历史tray9999后续状态不由本文继续跟踪；不能把4999的45/50归给9999。
- JG的NUMA与本项目SAPIEN/assets/cuRobo配置；node2的RESample目录/环境；其他SSH服务器实时资源及授权。
- Mac GPU细分规格、A800 GPU↔NUMA拓扑，以及快照之后的空闲资源；上线前重新只读核查。
- 本次未重跑测试和大数据校验；历史报告/validation与当前运行文件的完整一致性需按实验验收门禁确认。
