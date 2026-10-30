# RoboTwin DAgger 任务筛选与公开 checkpoint 验证

## 新增十任务公开 π0.5 测评队列（2026-10-25）

用户要求再增加十项任务测评。本轮选择尚无同协议正式结果的 `beat_block_hammer`、`click_alarmclock`、`click_bell`、`move_playingcard_away`、`press_stapler`、`place_empty_cup`、`place_object_stand`、`place_phone_stand`、`open_microwave`、`lift_pot`。官方 π0·Single 的 Easy 先验依次为 43%、63%、44%、53%、62%、37%、36%、35%、80%、84%；后五项中的 37/36/35 和 84 属于目标 40–80% 窗口的边缘/外侧探索，不能直接按官方率断言公开 π0.5 是否合适。按压类任务虽可能成功率合适，但阶段与可救回性也未验证，测评仅作筛选第一关。

Ubuntu `/media/ruio/hdd` 已核实挂载为 `/dev/sda1`，发布 manifest 包含十项 task-specific checkpoint，发布缓存的 seed0/seed2 场景各至少100条。独立队列脚本 `RoboTwin/scripts/run_pi05_ten_task_screen_20261025.sh` 部署至 Ubuntu `pace_pi05_eval_20261022/control/`，远端 `ten_task_screen_20261025/state.log` 和同名 orchestrator 日志记录完整过程。每任务先从 hf-mirror 在 Ubuntu 直接下载并逐文件 SHA256 验证，再做 seed2 一条独立技术 smoke，最后用发布专家筛选缓存的 seed0 前50个场景做正式测评：公开任务专属 π0.5、固定 H50、完整50动作、`demo_clean`、绝对关节动作、GPU0/18484、不启用 PACE 动态 horizon；无人工接管或训练数据保存。十项均有 metadata `complete`、各50个唯一 scene seed、与发布缓存列表逐项一致、无基础设施错误，合计500条正式 rollout。队列和自有策略服务器已退出。小体积 aggregate/metadata/逐场景记录已同步本机 `outputs/ten_task_pi05_screen_20261025/`，模型仍只在 Ubuntu。

| 新增任务 | 官方 π0 Easy（先验） | 公开 π0.5 实测，发布缓存场景 | DAgger 初筛备注 |
|---|---:|---:|---|
| `beat_block_hammer` | 43% | **32/50 = 64%** | 抓锤—敲击有阶段，但需验证工具抓持和撞击后能否安全接管。 |
| `click_alarmclock` | 63% | **34/50 = 68%** | 率合适，短时按压任务的可救回窗口可能很窄。 |
| `click_bell` | 44% | **23/50 = 46%** | 率合适，但短接触任务的专家纠错增益未证实。 |
| `move_playingcard_away` | 53% | **37/50 = 74%** | 抓持—移动可分段，下一步需做失败阶段分布与物理救回。 |
| `press_stapler` | 62% | **32/50 = 64%** | 率合适，仍须检查接管安全窗口而非只看成功率。 |
| `place_empty_cup` | 37% | **23/50 = 46%** | 实测进入软筛选窗，优先检查杯子掉落/抓持失败是否可救。 |
| `place_object_stand` | 36% | **23/50 = 46%** | 实测进入软筛选窗；物体类型/放置目标变化需纳入阶段门。 |
| `place_phone_stand` | 35% | **24/50 = 48%** | 实测进入软筛选窗；需检查手机姿态及支架接触质量。 |
| `open_microwave` | 80% | **31/50 = 62%** | 多阶段、潜在安全窗口较明确，但门/把手状态与规划可行性待测。 |
| `lift_pot` | 84% | **33/50 = 66%** | 实测落入软筛选窗；需验证抬锅任务的抓持和双臂状态门。 |

官方列是不同 π0 模型、100次测试的参考，不是同一 checkpoint 的直接差值估计。十项公开 π0.5 实测均在约40–80%的软筛选窗内，但**成功率无法证明失败集中于单一阶段、专家能从安全窗口救回、或30条有效HIL的采集成本**。按目前用户希望的“单任务、单一失败阶段”方向，下一步可优先对 `place_empty_cup`、`place_object_stand`、`place_phone_stand`、`move_playingcard_away` 与 `open_microwave` 做逐帧失败阶段定位，再决定其中哪一项值得编写/验证 HG-DAgger 专家接管脚本；短时按压任务留作对照。该缓存场景协议与此前 A800 无专家预筛 seeds31000–31049 **不可直接混算**。

## 本轮新增三项候选（2026-10-25，最新进度）

- 新增 `pick_dual_bottles`、`put_bottles_dustbin`、`stack_blocks_two` 的 HG-DAgger 任务适配器与独立入口；本地恢复逻辑/CLI/适配器测试20项通过，Ubuntu隔离副本恢复逻辑/CLI测试18项通过。修复了公共适配器文件中原 `rotate_qrcode` 返回逻辑不可达的回归，并修正 `put_bottles_dustbin` 投桶阶段把 `move()` 返回布尔值误当动作重复执行的问题。三任务均已完成真实公开 π0.5 策略接管诊断：pick seed40006可救回、seed40005重抓规划失败而被门控拒绝；put 原固定H50失败 seed100002在第100步接管后，逐瓶投桶的所有规划阶段成功，原任务与严格物理质量检查均通过；stack seed100000在第100步接管后重定位底块并放置顶块，原任务与严格接触/位置检查均通过。所有输出均标记 `diagnostic_only=true`、`accepted_for_training=false`，不进训练集；这只是单例救回证据，不是总体恢复率或正式采集数据。
- Hugging Face 发布 manifest 列出三项 task-specific π0.5 权重：每项约12.44GB、训练 checkpoint step 30000、50条 `demo_clean` 演示、训练 seed 42、EMA 0.99、预测 H=50、14维 `absolute_joint`，并各自带 norm stats。对应发布配置是 Pi0.5、`action_horizon=50`、从 `pi05_base` 初始化、batch 32、30,005步、FSDP 1、AdamW、warmup 1,000步后 cosine（peak LR 2.5e-5，30,000步衰减至2.5e-6）、EMA 0.99、seed 42。训练数据管线先将关节动作转为相对量，推理输出管线用 `AbsoluteActions` 还原为绝对动作；`adapt_to_pi=false`。已在 Ubuntu 发布版 OpenPI 环境逐项调用 `config.get_config()`，三项动态配置均可解析，运行时均返回H=50、30,005步、batch 32、seed 42、EMA 0.99及各自 `robotwin/<task>` 数据集。三项 checkpoint 已从 Ubuntu 端 hf-mirror 下载并按 manifest 逐文件 SHA256 校验，均保存在 `/media/ruio/hdd/pace_pi05_eval_20261022/checkpoints/`；不经本机中转。
- Ubuntu 固定 H=50 评测队列使用 GPU0 / 18480，正式场景为每任务发布缓存 seed0 的前50个专家筛选场景；三项依次执行 `fixed_h50`、`demo_clean`，不使用 PACE 自适应 horizon。首次队列在 `pick_dual_bottles` smoke 后因编排器读取错 `metadata.json` 路径退出，已改为读取 `evaluation/metadata.json` 与相邻 `episodes.jsonl`。最终 aggregate 标记 `complete=true`；每项50条唯一 scene seed、逐项匹配各自缓存列表，无技术/基础设施错误；H=50、14维 absolute_joint、`demo_clean` 均已核对。结果：`pick_dual_bottles` **16/50=32%**、`put_bottles_dustbin` **38/50=76%**、`stack_blocks_two` **32/50=64%**。总汇总在 `/media/ruio/hdd/pace_pi05_eval_20261022/control/three_dagger_candidate_eval_20261025/aggregate_three_tasks.json`；逐任务 metadata/episodes 路径分别位于 `/media/ruio/hdd/pace_pi05_eval_20261022/pace/runs/20261025-163018-443001/pick_dual_bottles/fixed_h50/seed0/evaluation/`、`/media/ruio/hdd/pace_pi05_eval_20261022/pace/runs/20261025-163810-739519/put_bottles_dustbin/fixed_h50/seed0/evaluation/`、`/media/ruio/hdd/pace_pi05_eval_20261022/pace/runs/20261025-165743-156563/stack_blocks_two/fixed_h50/seed0/evaluation/`。
- 当前未启动正式 HIL 采集或训练；所有冻结诊断输出均不进入训练集。正式成功率只在每项有50个唯一场景、元数据完成且没有未解决基础设施错误后更新。

状态：本轮新增三项筛选与任务专用接管 smoke 已完成。无专家预筛的既有公开 π0.5 正式50-seed测评包括 `handover_block` 25/50、`place_can_basket` 24/50、`move_can_pot` 28/50、`rotate_qrcode` 33/50、`place_burger_fries` 44/50、`put_object_cabinet` 14/50、`stack_bowls_three` 36/50；新增三项均使用发布 expert-screened cache，需单独解读：`pick_dual_bottles` 16/50=32%、`put_bottles_dustbin` 38/50=76%、`stack_blocks_two` 32/50=64%。按“约40–80%”的软筛选窗，put 与 stack 入窗，pick 低于窗；但 pick 有可救回的失败态，应保留为偏难备选而非机械淘汰。put 与 stack 各有一条原失败场景被专家接管后通过严格成功门的诊断证据；单例不能估计总体救回率。三项均已核实50个唯一 seed、元数据完整且无基础设施错误，冻结 smoke 不进入训练集。更新日期：2026-10-25。本轮三项 Goal 工作完成；本文档为持续更新的统一筛选记录。

## 当前 DAgger 候选与脚本验证关卡

| 任务 | 官方 π0 Easy | 公开 π0.5 实测（协议见证据列） | 当前脚本/救回证据 | 状态与剩余验证 |
|---|---:|---:|---|---|
| `move_can_pot` | 58% | 28/50=56% | task-specific collector；真实失败态 stage-1 重抓→安全 handback→策略完成单例；自动 `i/r` 混合帧68帧合法且已落诊断 raw；新增自动 `x` smoke 验证取消不保存 | **第一候选**。还需更多失败态和真人键盘验证；不能由单例推出总体救回率。 |
| `handover_block` | 45% | 25/50=50% | task-specific collector；两个真实失败 seed 的 stage-3 专家动作→安全 handback→策略完成；自动 `i/r/x` smoke 覆盖帧保存/取消排除 | **第一候选**。人工 GUI 按键与非 smoke 正式采集前核对仍待做。 |
| `stack_bowls_three` | 66% | 36/50=72% | task-specific collector；GPU4 绑定后的真实策略集成 smoke 在 step300 接受 `i`，stage-3 四段计划均成功，但最终原任务判据/严格几何检查失败，按规则未保存 | **条件候选，暂不正式采集**。当前中间态到末碗的救回未能产生成功样本；需先修正落点/质量验证并通过正样本保存 smoke，再测 GUI。 |
| `place_can_basket` | 41% | 24/50=48% | task-specific collector 已接通；step250 的恢复探针被 `can_placement_requires_separate_recovery` 门控拒绝；历史“重抓→放罐”规划及质量验证仍失败 | **储备**。不正式采集，先解决罐入篮的规划与释放问题。 |
| `put_object_cabinet` | 68% | 14/50=28% | 尚无 task-specific collector 或救回 smoke | 暂缓：公开策略低于首轮目标区间。 |
| `rotate_qrcode` | 68% | 33/50=66% | task-specific collector 已通过1条真实策略接管 smoke：step100 右手持板，stage-3任务原生放置成功；原判据、XY误差0.0033m、桌面支撑与关节合法门均通过 | **进入人工交互验证候选**；仍需验证真人 `i/r/x` 与 handback，单例不代表总体救回率。 |
| `pick_dual_bottles` | 57% | **16/50=32%** | 固定H50完成、缓存匹配；失败态 seed40006 接管后两瓶 lift/place 成功并通过严格物理门；seed40005 重抓规划失败，fail-closed | **偏难备选**。32%低于软目标窗，可能收集成本较高；单例表明至少一类双瓶状态可救回，仍需更多失败态验证。 |
| `put_bottles_dustbin` | 54% | **38/50=76%** | 固定H50完成；原失败 seed100002 第100步接管，逐瓶投桶与转交阶段均规划成功，原任务与严格物理门通过 | **可优先候选**。处于软目标窗上沿、仅12/50自主失败，30条有效HIL的采集成本仍需实测；救回证据目前为一个冻结 smoke。 |
| `stack_blocks_two` | 42% | **32/50=64%** | 固定H50完成；原失败 seed100000 第100步接管，底块重定位→顶块抓放→严格支撑/位置/双手释放检查均通过 | **可优先候选**。处于软目标窗中段、阶段边界较清楚；仍需跨 seed 验证救回率与有效HIL产出。 |

就本轮三项新增任务而言，优先顺序建议为 `stack_blocks_two`、`put_bottles_dustbin`、再到 `pick_dual_bottles`：前两者成功率落在软目标区间且各自有一次原失败 seed 的严格成功救回；pick 成功率32%，并同时存在救回成功和规划失败分支，适合作为偏难备选。三项均已完成逻辑/CLI测试、task-specific checkpoint 固定H50与一次任务专用恢复集成 smoke；这些 headless 冻结 seed 样本全部 `accepted_for_training=false`，尚未做真人键盘验证，也没有开始正式采集或训练。总体 DAgger 可行性仍需用新 seed 统计失败态可救回率，以及收集30条有效 HIL 所需有效 rollout 数后最终确认。

## 数据来源与不可混用的协议

- 官方逐任务数据：[RoboTwin 2.0 leaderboard](https://robotwin-platform.github.io/leaderboard) 的 [原始 JSON](https://robotwin-platform.github.io/data/robotwin_leaderboard.json)，`board.updated=2026-10-23`，条目 `sft:π0`。以下 Easy 是 `demo_clean` 训练到 `demo_clean` 测试；Hard 是到 `demo_randomized` 测试。榜单说明每任务训练数据为 50 条 `demo_clean`，每任务/环境 100 次测试。它们是**官方 π0 单任务 SFT 的先验**，不是将要部署的公开 π0.5 checkpoint 的实测率。
- 公开候选模型：[niejunnan25/pi05-robotwin](https://huggingface.co/niejunnan25/pi05-robotwin)，50 个任务各有独立的权重与 `norm_stats`；模型卡说明每任务 50 条干净演示、全参数微调、发布 30,000 步 EMA 权重，原生输出 50×14 绝对关节动作。实际部署率须另行测量。
- 第一轮适用性门槛是 **π0 Easy 大致 40%–80%**，不是机械淘汰线。阶段边界能否从物理状态判定、专家能否从中间态救回、以及有效 HIL/有效 rollout 的产出率同等重要。此前候选率使用无专家预筛；本轮新增三项为与发布工作流对齐、使用 expert-screened cache 的 seed0 前50场景。两类场景协议不能混成一列排名或直接做优劣归因；本表的每个新行已标注其场景来源。仿真初始化/服务失败单记基础设施错误。
- **已有结果不可遗漏**：此前 Ubuntu 已下载并测过公开 `handover_block` π0.5 checkpoint，固定 H50 在 50 个专家筛选缓存场景上成功 **30/50**（`outputs/pace_pi05_eval_20261022/formal/handover_block/fixed_h50/seed0/evaluation/metadata.json`，`expert_check_mode=cached_scenes`、`instruction_type=unseen`）。这说明它有测试过的模型与推理链路，但不是新工作流要求的无专家预筛成功率；当前 A800 下载是另一台机器上的副本。单条 smoke 仅检验基础设施，不用来挑任务或估成功率。
- 接管评估应分开记录：进入目标阶段的有效 rollout、接管请求、接管后成功并保存的 HIL、无人接管的自主成功、用户 `x` 取消；最终报告到 30 条有效 HIL 所需的有效 rollout 数，而不只报告接管请求率。

## 官方 π0·Single 全部 50 任务

| # | 任务 | Easy % | Hard % |
|---:|---|---:|---:|
| 1 | Adjust Bottle | 90 | 56 |
| 2 | Beat Block Hammer | 43 | 21 |
| 3 | Blocks Ranking RGB | 19 | 5 |
| 4 | Blocks Ranking Size | 7 | 1 |
| 5 | Click Alarmclock | 63 | 11 |
| 6 | Click Bell | 44 | 3 |
| 7 | Dump Bin Bigbin | 83 | 24 |
| 8 | Grab Roller | 96 | 80 |
| 9 | Handover Block | 45 | 8 |
| 10 | Handover Mic | 98 | 13 |
| 11 | Hanging Mug | 11 | 3 |
| 12 | Lift Pot | 84 | 36 |
| 13 | Move Can Pot | 58 | 21 |
| 14 | Move Pillbottle Pad | 21 | 1 |
| 15 | Move Playingcard Away | 53 | 22 |
| 16 | Move Stapler Pad | 0 | 2 |
| 17 | Open Laptop | 85 | 46 |
| 18 | Open Microwave | 80 | 50 |
| 19 | Pick Diverse Bottles | 27 | 6 |
| 20 | Pick Dual Bottles | 57 | 12 |
| 21 | Place A2B Left | 31 | 1 |
| 22 | Place A2B Right | 27 | 6 |
| 23 | Place Bread Basket | 17 | 4 |
| 24 | Place Bread Skillet | 23 | 1 |
| 25 | Place Burger Fries | 80 | 4 |
| 26 | Place Can Basket | 41 | 5 |
| 27 | Place Cans Plasticbox | 34 | 2 |
| 28 | Place Container Plate | 88 | 45 |
| 29 | Place Dual Shoes | 15 | 0 |
| 30 | Place Empty Cup | 37 | 11 |
| 31 | Place Fan | 20 | 10 |
| 32 | Place Mouse Pad | 7 | 1 |
| 33 | Place Object Basket | 16 | 2 |
| 34 | Place Object Scale | 10 | 0 |
| 35 | Place Object Stand | 36 | 11 |
| 36 | Place Phone Stand | 35 | 7 |
| 37 | Place Shoe | 28 | 6 |
| 38 | Press Stapler | 62 | 29 |
| 39 | Put Bottles Dustbin | 54 | 13 |
| 40 | Put Object Cabinet | 68 | 18 |
| 41 | Rotate QRcode | 68 | 15 |
| 42 | Scan Object | 18 | 1 |
| 43 | Shake Bottle Horizontally | 99 | 51 |
| 44 | Shake Bottle | 97 | 60 |
| 45 | Stack Blocks Three | 17 | 0 |
| 46 | Stack Blocks Two | 42 | 1 |
| 47 | Stack Bowls Three | 66 | 24 |
| 48 | Stack Bowls Two | 91 | 41 |
| 49 | Stamp Seal | 3 | 4 |
| 50 | Turn Switch | 27 | 23 |

## 第一阶段候选名单（仅供下一轮实测；不是已验证可救回）

优先验证的候选：

| 任务 | π0 Easy/Hard | 专家脚本提供的阶段线索 | 暂定判断与需要验证的风险 |
|---|---:|---|---|
| `handover_block` | 45/8 | 抓取、抬升、交接姿态、另一臂抓取、释放、目标放置（`RoboTwin/envs/handover_block.py`） | **优先**。双臂交接有明确阶段；需验证掉落后是否还能救回，以及不能仅按脚本行号推断实际进度。 |
| `place_can_basket` | 41/5 | 易拉罐抓放、另一臂抓篮、抬篮（`RoboTwin/envs/place_can_basket.py`） | π0.5 无预筛 **24/50=48%**，区间合适；但真实策略失败态恢复已试多次仍卡在罐的放置规划，暂列**储备/暂不采集**，先解决规划可达性。 |
| `put_object_cabinet` | 68/18 | 一臂抓物、另一臂拉抽屉、抬物、放入（`RoboTwin/envs/put_object_cabinet.py`） | π0.5 无预筛 **14/50=28%**，低于首轮目标区间；尽管任务有清晰阶段，暂缓投入，除非找到可靠救回且rollout成本可接受。 |
| `stack_bowls_three` | 66/24 | 三次依次放碗（`RoboTwin/envs/stack_bowls_three.py`） | π0.5 无预筛 **36/50=72%**，落在目标区间上沿；阶段清晰且脚本部分通过。可留候选，但正式采集前须核验原成功判据的物理质量，并通过有效正样本落盘与人工交互验证。 |
| `place_burger_fries` | 80/4 | 双臂分别抓取、汉堡放盘、薯条放盘（`RoboTwin/envs/place_burger_fries.py`） | **边界候选**。动作可分阶段，但 π0 已在 80%；若公开 π0.5 自主率更高，可能难以在有限 rollout 中收足 HIL。 |
| `move_can_pot` | 58/21 | 抓取、抬移、目标放置（`RoboTwin/envs/move_can_pot.py`） | **暂留优先候选**。无预筛公开 π0.5 为28/50；真实策略失败态已有一次“专家重抓→策略接手→原任务成功”的闭环，仍需多 seed 验证恢复率和人工交互。 |
| `rotate_qrcode` | 68/15 | 抓、抬、定向放置（`RoboTwin/envs/rotate_qrcode.py`） | **定向候选**。原任务成功判定检查姿态、高度与双手松开，**不检查目标 XY 位置**；另加目标位置门后，已有一条“右手持牌→任务原生放置”的策略失败态救回 smoke。仍需验证真人接管与 handback，且不可把单例解释为总体救回率。 |

需要特别核验的例外：`dump_bin_bigbin` 的官方 π0 Easy 83% 略超区间，但已有**不同于官方筛选缓存场景**的公开 π0.5 同 seed 无预筛测评：完整 H50 为 36/50，前 10 步重规划为 29/50；自训 5k H10 为 23/50。该任务的倾倒/多球物理和仿真 settling 修补增加了阶段救回与协议可比性风险，暂列次级候选。`turn_switch` 的官方 π0 Easy 27%，公开 π0.5 在专家筛选缓存场景中为 23/50；单阶段触碰容易定义但任务过短，暂列简化对照，不与优先长时任务直接比较。

暂缓原因：`stack_blocks_three` 官方 π0 Easy 17%，公开 π0.5 专家筛选缓存场景也只有 12/50；需要先证明中间层可稳定救回。`click_alarmclock` 与 `press_stapler` 虽在成功率区间，但主要是短接触/按压，阶段与可恢复偏离状态的丰富程度不足，暂不优先。其余低于 40% 或高于 80% 的任务不直接淘汰，但除非发现特别好的阶段/恢复结构或公开 π0.5 实测率与 π0 明显不同，先不下载其权重。

`stack_bowls_three` 的附加判定风险：原任务 `check_success()` 将三个碗按当前高度排序后要求相邻 XY 坐标差小于 0.04，但高度条件写作 `height - target_height < 0.02`，**没有取绝对值或设置下界**。因此正式结果中的 `success=true` 是原任务定义的成功，不自动证明三只碗均物理稳定堆叠；若其 50-seed 比例进入候选区间，仍需抽查最终姿态/接触并为有效 HIL 单独设质量核验，不能静默修改原 benchmark 分母。

`place_burger_fries` 也需额外核验物理放置：原任务 `check_success()` 只比较汉堡和薯条各自功能点到托盘两个功能点的 XY 距离（各 `<0.08`）并要求双夹爪张开，没有高度或食物—托盘接触条件。正式 50-seed 结果仍按原任务判定报告，但若模型因此被考虑用于 DAgger，专家救回/有效 HIL 的质量门槛应另查两件物体是否真的落在托盘上，不能把 XY 接近直接解释成稳定放置。

### 阶段检测与专家救回的待验证设计

这些是从各任务 `play_once()` 与 `check_success()` 的代码推得的**状态量与禁入条件**，不是已实现或已通过中间态 smoke 的方案；最终只对实测入选任务实现。阶段以物理状态识别，不拿专家脚本行号冒充策略 rollout 的阶段。

| 任务 | 可观测阶段线索 | 暂定接管入口与必须拒绝的情形 |
| --- | --- | --- |
| `handover_block` | 红块位姿/长轴姿态、两手抓持与接触、蓝色目标位姿 | 左手抓持可续交接（已有一条成功 smoke）；双手持有待检验松左手；右手持有待检验放置。无人抓持且侧躺已两次重抓失败，现有后缀必须拒绝自动执行。 |
| `place_can_basket` | 易拉罐与篮子位姿、罐—篮接触、罐—桌接触、双手抓持、篮子相对初始高度与朝向 | 罐稳定在篮内时才可只救篮子抓起/抬升阶段；罐未入篮不能误判为末段，只能在罐可达且静止时尝试重新入篮，失败/碰撞须单记。 |
| `put_object_cabinet` | 抽屉把手/功能点相对初始位移、物体与双手抓持、物体相对柜内功能点位姿 | 抽屉已开且物体在手中才考虑直接放入；抽屉未开需先从当前关节与物体状态开抽屉。物体掉入不可达位置、抽屉卡滞/碰撞时拒绝脚本盲重播。 |
| `stack_bowls_three` | 三只碗的身份、XY/高度、姿态与接触、夹爪抓持 | 只有下层已稳定、上层仍可达时才进入“补最后一只碗”；下层倒塌不能把原任务宽松的高度单侧判定当成稳定层数，需从底层修复或排除。 |
| `move_can_pot` | 罐—锅相对位置、罐俯仰/滚转、罐是否在手/已落桌、双夹爪状态 | 持罐可续搬运或放置；罐已落桌且静止、可达时才尝试重抓；侧倒/被锅遮挡不能仅按 XY 距离宣称可救回。 |
| `rotate_qrcode` | QR 牌位姿四元数、目标面朝向、夹爪接触/开合、是否落桌 | 持牌可续旋转放置；已在目标附近但姿态不对时需重新定向，掉落且不可抓时拒绝自动恢复。判定必须复用原任务姿态阈值。 |
| `place_burger_fries` | 汉堡/薯条各自到托盘功能点的 XY 与高度、真实接触、左右手抓持 | 一件已稳定放在托盘、另一件仍可抓时可只救缺失的放置阶段；两件都偏移或仅 XY 接近而未接触托盘时不得按原宽松成功判定简化为末段。 |

### `stack_bowls_three` 恢复与 HG-DAgger 采集器

`RoboTwin/scripts/stack_bowls_three_recovery.py` 的入口要求 bowl1、bowl2 分别稳定在目标底层/中层附近，bowl3 在桌面可恢复范围或被可靠抓持；下层错位、越界、抓持不确定及已完成状态均拒绝动作。stage-3 suffix 将最后一只碗分为 grasp/lift/place/retract 四个语义步骤，每步核对规划；抬升后再看真实高度/抓持，末尾同时核验原始 `check_success()` 与额外的绝对几何质量门槛。质量门槛不替换 benchmark 原成功率。5项 stack 测试与 handover、move-can、place-can 三个 helper 测试合计20项直接调用测试通过；运行环境无 pytest，因此是逐个调用测试函数，不代表 pytest runner 全套。

新增 `RoboTwin/scripts/hg_dagger_stack_bowls_three.py`：策略 chunk 内逐动作监听 `i/x/q`；`i` 时按物体位置、抓持和下层稳定性分类，拒绝危险/不确定状态；获准后丢弃 chunk 余量并运行 stage-3 suffix。`r` 只在重新分类仍可继续、规划有效且任务未完成时交还策略；最终只有发生真实接管、存在 HIL 帧、未取消、未失败、原任务成功、严格几何质量检查通过且关节合法才落 raw 样本。所有自动按键和“专家前缀构造两碗状态”都被限制为单条 `--headless-smoke` 诊断；冻结 seed 永远设置 `accepted_for_training=false`。

此前 collector 的 step300 `i/r/x` 混合流 smoke 在未限定 GPU 的 server 下完成了控制流验证，但不作为 GPU 隔离证明；之后发现 SAPIEN 可用 `sapien.Device("pci:0000:98:00.0")` 显式绑定 GPU4，且在 `CUDA_VISIBLE_DEVICES=4` 下 native `SapienRenderer` 探针通过。新增采集器 CLI 逻辑允许 smoke 只触发 `i`、由专家完成后缀；自动 `r` 仍必须先有 `i`。4 项 CLI 单测通过。

最新 policy-backed collector smoke（A800 seed31000，GPU4 显式用于 server 与 renderer，端口18465）在 policy step300 接受 `i`，stage-3 grasp/lift/place/retract 的 `plan_success` 都为真，关节合法，得到168个 HIL 帧；但末碗实际中心为约 `(0.0528,-0.0788,0.8119)m`，相对中碗 x 偏差约4.9 cm，原 `check_success()` 与严格物理门槛均为假。记录 `expert_failed=true`、`valid_hil=false`、`saved_valid_hil=0`、`raw_dir=null`；缓存被清理，没有 raw 训练轨迹。结论：**规划器报告成功不等于物体放置成功**，采集器的拒绝保存是正确 fail-closed 行为；需先针对末碗落点偏差修复/复测，不得放宽质量阈值。该 smoke 明确为诊断数据，不能作为训练样本或正式成功率。

此次 server 和 collector 都设 `CUDA_VISIBLE_DEVICES=4`，SAPIEN native renderer 绑定 GPU4 PCI BDF `0000:98:00.0`；server 退出后 GPU4 显存回到约1 MiB、18465 端口释放，没有触碰其它 GPU 任务。此前目录不正确及 Renderer 初始化失败的尝试均在策略动作前退出，不计为策略失败；GPU 选择问题现已绕过，但 stack 的有效样本关卡仍未通过。

有界专家前缀的 collector smoke 暴露了一条重要安全情形：即使前两步 `move_bowl` 报告规划成功，真实 bowl2 仍可能没有放稳；分类器返回 `lower_bowls_not_stably_aligned`，采集器拒绝启动末阶段专家动作并清掉缓存。最新隔离重放（seed31000，run1）记录 bowl1=(-0.0015,-0.0983,0.7400)m、bowl2=(-0.2098,-0.0752,0.8278)m，左夹爪闭合、仍未形成下层堆叠；247帧中 policy 1、HIL 246，`valid_hil=false`、`raw_dir=null`，没有训练样本落盘。另一次不带前缀的同 seed replay 到 policy step340仍未进入可接管分类，随后仅用于观察、在 step637由我们停止；不作成功率或脚本通过证据。原生帧**正样本**落盘/episode 导出与真人 GUI `i/r/x` 仍未验证，因此 stack 采集器**部分通过而非完整验收**。这次不是仿真崩溃，而是物理状态门槛正确地 fail-closed；不应为“通过 smoke”而放宽分类阈值。

随后尝试用公开策略在相同冻结 seed 进入已有的末碗恢复分支，以覆盖“专家完成后正样本写盘”：一次诊断仅因启动目录不对、在资产导入前退出；修正目录后的隔离尝试在 `sapien.SapienRenderer()` 创建 Vulkan device 时抛 `ErrorInitializationFailed`，尚未开始场景或 rollout，也没有 `rollouts.jsonl` / raw 样本。故不能把它记成策略失败或采集器失败。A800 `/` 当时为 0 可用。跟进查明本机 SAPIEN 3.0.0b1 的 `_vulkan_tricks.py` 只检查 `/usr/share/vulkan/icd.d/nvidia_icd.json`，而主机的 NVIDIA ICD 位于 `/etc/vulkan/icd.d/nvidia_icd.json`；官方 ICD 路径和 `Device("cuda:0/4")` 的隔离探针仍分别以不支持设备/`invalid PCI string` 失败，故 Vulkan 根因与可用的绑卡方案**尚未解决**。诊断 server PID `2482883` 已停止，18465 已释放；该次 server 未设置 GPU 可见性限制，运行期间多张卡的显存占用升高、退出后 GPU4/7 回到 1 MiB。今后必须先验证严格的服务端和渲染端 GPU 绑定，再重做正样本写盘 smoke；没有清理磁盘，也没有停止其他用户进程。

## 下一阶段待办与判定记录

1. 为优先候选逐任务核对发布 manifest 的任务名、权重文件 SHA256、`assets/robotwin/norm_stats.json`、训练/推理配置；禁止将 π0.5 原生 H50 发布协议静默改成自训策略的 H10。执行频率还需与具体评测代码核对，不能仅凭模型卡假定。只在远端服务器直接下载，先检查 Ubuntu、A800、JG 的 GPU/磁盘/SSH/仿真身份；机器不可达时记录，不伪称并行。
2. 先每任务少量 smoke，再用统一的固定 seed、`demo_clean`、无专家预筛协议正式测自主成功率；必要时记录合法的替代 seed 与仿真初始化错误。记录阶段失败分布和逐 seed 结果。初筛区间以**实际公开 π0.5**而不是官方 π0 率判定。
3. 对最终入选任务，基于物体位姿、接触、关节状态与任务成功条件定义可复查的阶段检测与救回动作；复用既有 HG-DAgger 的人工 `i` 接管、`x` 排除和有效 HIL 保存口径。先做中间态恢复 smoke，不自动承诺所有策略失败都可救回。
4. 保留入选模型；若淘汰，仅在核实下载来源、精确目标路径、结果与校验清单均已保存后清理**本轮下载的副本**。绝不触碰先前实验或他人文件。每次清理记录对象及可恢复性。

| 任务 | 发布权重已校验 | 统一协议 smoke | 正式 π0.5 成功率 | 可救回阶段验证 | 保留/淘汰 | 证据路径 |
|---|---|---|---|---|---|---|
| `handover_block` | Ubuntu 旧副本已测；A800 新副本已校验 | A800 无预筛 smoke seed31000 成功，281步 | 旧缓存场景 H50 30/50；A800 新无预筛 25/50，50 个唯一 seed31000–31049，`complete=true` | 策略左手持有旧一键后缀成功 1 条；侧躺掉落重抓失败 2 条并拒绝；JG 专家中间态分阶段接口完成/暂停各 1 条；Ubuntu 两个原本无干预失败的 seed31002、31012 均在第150步经两次专家 stage-3 动作交还策略后成功；Ubuntu 自动 `i/r` smoke 已保存真实策略接管混合帧，自动 `x` smoke 已验证排除；人工键盘操作与正式新 seed 采集仍未验证 | 暂留；无预筛成功率适中，但可救回域仍需扩展验证 | `outputs/pace_pi05_eval_20261022/formal/handover_block/fixed_h50/seed0/evaluation/metadata.json`；A800 `outputs/robotwin_dagger_screen_20261024/results/handover_block_h50/aggregate_summary.json`；本机 `outputs/handover_block_staged_smoke_20261025/` 与 `outputs/robotwin_dagger_screen_20261024/diagnostics/handover_seed31002_step150_handback2/`、`handover_seed31012_step150_handback2/`、`handover_native_frames_seed31002_v2/`、`handover_hil_collector_auto_seed31002/`、`handover_hil_collector_auto_x_seed31002/` |
| `place_can_basket` | A800 新副本已完成下载器校验 | 首次 smoke 因缺 `071_can`、`110_basket` 资产而在策略前失败；补齐后 seed31000 产出有效记录但策略失败（700步） | 无专家预筛固定 seeds31000–31049，原任务成功 **24/50=48%**；50唯一seed、无基础设施排除、逐seed终态一致 | JG 专家构造的 seed31002 后段抬篮可完成；真实策略 seed31000 第250步曾出现“罐仍在桌、篮由另一臂持有”的状态。恢复 helper 能放下篮并重抓罐，但直接放置与之后尝试的 grasp-compensated fallback 均规划失败；失败时不会释放罐，原成功判据为假。篮倾倒分支按门槛拒绝 | **储备候选，暂不进入正式采集**：成功率合适，但现有策略失败态恢复未通过；下一步应诊断碰撞/可达性/末端姿态，不再靠盲调目标点 | A800 `outputs/robotwin_dagger_screen_20261024/results/place_can_basket_h50/`；本机同名目录含汇总、协议及50条逐seed JSON；诊断见下方时间线 |

`place_can_basket` 原任务 `check_success()` 同时要求篮子相对初始位置抬高、罐子抬高、篮子轴向正确、罐篮空间接近、罐不接触桌面且接触篮子。因此 24/50 比仅看“抓住罐”或“抬起篮”严格；这也给接管脚本提供明确的状态门槛。无干预正式分母已审计；失败态恢复目前只对少数隔离状态做过诊断，不能据此估计26个失败的总体可救回比例。

最初新增的 `place_can_basket_recovery.py` 针对罐已稳定在篮内的后段；之后为覆盖真实策略状态“罐仍在桌、篮由另一臂持有”而加入独立重抓/放罐分支。4项逻辑测试（含拒绝态绝不动作）通过。JG 专家构造中间态 seed31002 的篮抓取与抬升可执行；这只证明该中间态的后缀可执行，不是无干预失败态救回率。JG 旧版帧无 `control_mask`，该诊断不作为训练数据。

2026-10-25 后续验证：候选 evaluator 新增仅诊断使用的 `--trace-task-state`，在每个动作 chunk 边界记录 `place_can_basket` / `move_can_pot` 的物理快照及保守阶段判断；`--recover-at-step` 对这两项也只允许在独立 `--smoke-only` 运行中触发专家后缀，不能进入正式成功率统计。相关脚本通过语法检查。A800 上正式结果仍为24/50。公开策略对同 seed31000 的重复 rollout 呈现随机差异：完整无干预诊断曾到700步仍失败；另一次在250步进入“罐仍在桌面、抓取臂空、另一臂持篮”的状态，专家重抓罐成功，但原始直接放置规划失败；再一次在250步时篮子已明显倾倒，分类器按姿态门槛拒绝动作。为覆盖前一种失败，恢复后缀加入原任务的备用放置动作；A800 同隔离 GPU2/端口18467 的 attempt3 正在重放该策略失败场景以验证备用动作，尚无通过结论。诊断目录分别为 `outputs/robotwin_dagger_screen_20261024/diagnostics/place_can_basket_real_policy_seed31000_trace/`、`..._recovery_step250_attempt2/`、`..._recovery_step250_attempt3/`；这些都不是正式成功率记录，也不与24/50混算。
| `put_object_cabinet` | A800 本轮任务专属权重下载完成，发布 manifest SHA256 校验通过；runner 二次校验通过 | 无预筛 seed31000 smoke 有效，策略 700 步未成功，无基础设施错误 | **14/50=28%**；31000–31049 50个唯一 seed，`complete=true` | 尚无任务专属 DAgger 恢复采集器验证 | 暂缓；成功率低于首轮目标区间，暂不投入人工接管开发 | A800 `outputs/robotwin_dagger_screen_20261024/results/put_object_cabinet_h50/aggregate_summary.json` |
| `stack_bowls_three` | A800 任务专属公开权重下载校验完成；runner 二次校验通过 | 无预筛 seed31000 smoke 有效并成功，453 步 | **36/50=72%**；31000–31049 50个唯一 seed，`complete=true`，无基础设施排除 | helper 的专家构造 stage-3 smoke 通过；真实策略 collector 在 step300 接受 `i`，四段专家计划成功但最终原任务/严格质量判据失败，168 HIL 帧不保存；CLI 单测4项通过；真人 GUI 未测 | **条件候选，不宜现在正式采集**；先修末碗落点偏差并通过成功保存 smoke，保留严格物理质量门槛 | A800 `outputs/robotwin_dagger_screen_20261024/results/stack_bowls_three_h50/aggregate_summary.json`；本机 `outputs/robotwin_dagger_screen_20261024/results/stack_bowls_three_h50/aggregate_summary.json`；本机隔离复测 `outputs/robotwin_dagger_screen_20261024/diagnostics/stack_bowls_three_collector_gpu4_pinned_20261025/attempt2/` |
| `place_burger_fries` | A800 任务专属下载器与 Ubuntu runner 均已逐文件 SHA256 验证 | Ubuntu 无预筛 seed31000 技术 smoke 230步成功，不计入正式分母 | Ubuntu 正式固定 31000–31049：**44/50=88%**；50 唯一 seed、无技术错误、每条 `success=final_check_success` | 待办；原成功判定只看 XY/双手张开，物理放置质量仍需独立核验 | 暂留模型待物理有效性复核；按原判定仅 6/50 失败，可能不利于 30 条有效 HIL 采集 | Ubuntu `/media/ruio/hdd/robotwin_dagger_screen_20261024/results/place_burger_fries_h50/`；本机 `outputs/robotwin_dagger_screen_20261024/results/place_burger_fries_h50/` |
| `move_can_pot` | A800 任务专属下载已逐文件 SHA256 验证 | 技术 smoke seed31000 有效完成，不计入成功率 | 无专家预筛 **28/50=56%**；seed31028 在策略执行前因场景初始化不稳定而排除，补入31050，实际50个唯一seed为31000–31050去掉31028；`complete=true`，逐seed原终态与记录一致 | JG 专家构造持罐中间态阶段2→3已成功。真实策略 seed31000 第50步由专家只做 stage-1 重抓，复核持罐后交还策略；策略继续66步，原任务 `check_success()` 通过。相同策略 seed 第100步介入太晚、重抓失败；第150步罐已出工作区。自动 `x` 取消 smoke 已验证不保存 raw。仍需更多失败 seed、真人键盘 `i/r/x` 和成功救回样本保存，不能据单例估计总体救回率 | **暂留优先候选**：56%基线在目标区间，且存在真实失败态救回与策略 handback 的单例证据；仍需更多失败态及真人操作核验 | A800 `outputs/robotwin_dagger_screen_20261024/results/move_can_pot_h50/`；本机同名目录含汇总/协议/逐 seed JSON；诊断在本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/move_can_pot_real_policy_seed31000_step50_attempt6/`、`..._step50_handback1_attempt7/`、`stack_bowls_three_collector_gpu4_pinned_20261025/move_can_pot_x_only_seed31000/` |

`move_can_pot` 的被排除 seed31028 原始 `eval.log` 追溯到 `setup_demo → _init_task_env_` 抛出 `UnStableError: Objects is unstable ... 105_sauce-can`，发生在任何策略动作之前；不是策略失败。独立审计核对汇总、协议及50个逐seed文件，得到 28 次原任务 `check_success`，且无重复 seed。原任务成功条件同时要求罐在锅的指定侧、罐锅 Y 对齐、罐的姿态与高度满足阈值、双夹爪张开，故不只是宽松的 XY 靠近；但 22 个失败的可救回比例尚未实测。

已新建 `move_can_pot_recovery.py` 的保守持罐分支：根据任务原始左右臂分配、夹爪闭合与罐接触/距离、锅相对目标漂移、罐子工作空间，以及罐相对目标高度/水平距离决定阶段2搬运抬升或直接进入阶段3放置；不持罐、锅已移位与越界均拒绝自动动作。3项逻辑测试通过。JG 原环境缺 `060_kitchenpot` 和 `105_sauce-can`，仅从 A800 经 Ubuntu 中转服务器间同步这两个物体资产（28MB+36MB，不经本机）；随后专家构造的 seed31000 左臂抓罐后中间态，阶段2→3原任务成功；seed31001 右臂抓罐后同样成功；seed31000 已抬罐后用改进分类器直接执行阶段3，原任务成功。结果只说明**专家构造态的后缀可执行**，不证明真实策略失败能被救回，也不估计有效 HIL 率；未改动正式 28/50 统计。JG 旧版 `Base_Task` 无 `control_mask`，这些 smoke 不当训练数据。脚本 `smoke_move_can_pot_recovery.py` 与诊断 JSON 已保存，尚未接到人工 `i/r/x` 采集器。
| `rotate_qrcode` | A800 任务专属下载器已逐文件 SHA256 验证，runner 二次校验通过 | 技术 smoke seed31000 有效成功，不计入成功率 | 无专家预筛固定 seeds31000–31049，原任务成功 **33/50=66%**；50唯一seed、无基础设施排除、逐seed终态一致 | 新增采集器已在 seed31000、step100 真实策略状态下完成右手持牌→任务原生放置；严格质量门通过，冻结诊断帧不进训练。手动键盘及 handback 仍待验证 | **可进入人工交互验证候选**：适用率合适，但保留 XY/桌面支撑质量门，单例不足以证明稳定救回 | A800 `outputs/robotwin_dagger_screen_20261024/results/rotate_qrcode_h50/`；本机同名目录含汇总、协议及50条逐seed JSON；新增诊断在 `outputs/robotwin_dagger_screen_20261024/diagnostics/hg_dagger_two_task_collectors_20261025/rotate_qrcode_auto_i100_v1/` |

`rotate_qrcode` 的33次成功来自原任务 `check_success()`：QR牌四元数逐分量距目标小于0.05、落桌高度满足阈值、两手松开；源码没有对 `target_pose` 的 XY 距离进行检查。与官方 π0 Easy 68% 的任务定义相符，但不说明牌被放到了目标区域。若作为 DAgger 主任务，需额外报告最终 QR 牌到目标的 XY 偏差，且不可事后把此辅助指标冒充官方成功率。
| `dump_bin_bigbin` | 已有独立验证 | 已有独立验证 | H50 36/50；H10 29/50（修复版无预筛场景） | 待办 | 待定 | `outputs/public_pi05_dump_bin_bigbin_matched_eval_20261024/results/` |

## 2026-10-24 资源与发布清单核对

- **候选任务的发布清单身份（不是本地文件完成证明）**：以下由 A800 上的 `outputs/public_pi05_dump_bin_bigbin_matched_eval_20261024/control/manifest.json` 逐条读取；清单对应公开仓库 `niejunnan25/pi05-robotwin`，任务名、配置名与任务专属 `norm_stats` 不可互换。每项的完整权重仍由 runner 在启动时逐文件复核 SHA256；实际下载/测评状态见上表与下方时间线。

  | 任务 | 发布配置名 | `norm_stats.json` SHA256 | 清单文件数 / 总字节 |
  | --- | --- | --- | ---: |
  | `handover_block` | `pi05_robotwin_handover_block` | `8a1f114a8873dfa0788c7f512851ee3063224ef2b8af270421076a2e4e544960` | 27 / 12,441,552,737 |
  | `place_can_basket` | `pi05_robotwin_place_can_basket` | `0e66c99edee31889518c8e40d06208d103c0c86fb8f23e860508a32d20d4a10f` | 28 / 12,441,564,513 |
  | `put_object_cabinet` | `pi05_robotwin_put_object_cabinet` | `2fc6901bf06e8e64a38086cbd218673c503fff4d45e69c9ef3fc0c445201f7cc` | 32 / 12,441,566,271 |
  | `stack_bowls_three` | `pi05_robotwin_stack_bowls_three` | `d11355f1ce641b41851775ca7256cd0699ecd13b89ff10b11f2cb0a5f3b5a925` | 31 / 12,441,549,382 |
  | `move_can_pot` | `pi05_robotwin_move_can_pot` | `c0f1e03ebc7f55467870acc05fedbea253b5441600c1b9e40d8dcf9a90e617ed` | 28 / 12,441,559,410 |
  | `rotate_qrcode` | `pi05_robotwin_rotate_qrcode` | `c3d46734902c0857554d6ed8de34cc939423ce6c67dab455686fe57a3ef099c5` | 30 / 12,441,557,654 |
  | `place_burger_fries` | `pi05_robotwin_place_burger_fries` | `00d67952fa59db921fc6be3f123cf12653bbd9031f759b499284c6e35fff469b` | 36 / 12,441,580,934 |

- 已在 A800 读取公开发布 manifest：共 50 个任务，以上 7 个优先候选及 `dump_bin_bigbin` 均有任务专属条目、配置名和逐文件 SHA256；例如 `handover_block` 的 `norm_stats` 指向 `handover_block/assets/robotwin/norm_stats.json`，SHA256 为 `8a1f114a8873dfa0788c7f512851ee3063224ef2b8af270421076a2e4e544960`。这一步仅验证**清单存在**，尚未宣称这些模型文件已下载或通过校验。
- A800 (`new_power_3`) SSH 可达，`/data-training` 约 297 GB 可用，GPU4 检查时显存使用 1 MiB；其他 GPU 有现存任务，不因空闲判断而停止它们。候选权重每个约 12.4 GB，须控制同时下载数量并预留测评输出空间。已启动首个候选 `handover_block` 的远端下载与逐文件 SHA256 校验，进程初始 PID `1877566`，本轮专用目录 `/data-training/robotwin-hil-lyt/outputs/robotwin_dagger_screen_20261024/models/`，日志在同根 `control/download_handover_block.log`；**完成状态仍待核实**，不算已验证模型。
- Ubuntu SSH 于 2026-10-24 恢复，可连接并运行 Codex CLI。JG 的局域网别名仍在握手阶段关闭，但 **`JG-ts` Tailscale 别名已可连接**：一张 RTX 5090 D v2，24,455 MiB 显存，检查时仅占 41 MiB；`/home` 剩余约 139 GB。JG 的 `/home/ruihao/robotwin_hil/conda` 可导入 SAPIEN 3.0.0b1，不过现有 `RoboTwin/assets/objects` 基本为空；公开权重、发布版服务代码与仿真资产未完成同协议校验，因此**不能把 GPU 空闲等同于测评环境已就绪**。模型下载不能经本地 Mac/手机流量中转。
- 已编写并部署候选任务共用的 `RoboTwin/scripts/eval_public_pi05_candidate.py` 与 `RoboTwin/scripts/run_public_pi05_candidate.py`：逐 seed 保存结果；smoke 与正式记录分开；技术错误立即停机、不计为策略失败；运行前重新核对发布 manifest 的每个权重文件与任务专属 `norm_stats` SHA256。脚本已通过语法检查和远端命令导入/参数检查，但尚未通过真实模型 smoke。第一候选拟使用 `handover_block`、固定提示词、原生 H50、无专家预筛、seed 31000–31049；真实频率字段及步数上限在启动时写入 protocol.json。
- 发布代码 `openpi_pace/pace/src/openpi_pace/configs.py` 动态生成 `pi05_robotwin_{task}`，其中 `Pi0Config(pi05=True, action_horizon=50)`、`prompt_from_task=True`、EMA 0.99、30,005 配置训练步数、batch32、AdamW/余弦学习率、FSDP1。**更正维度：**`pi0_config.py` 的模型内部 `action_dim` 默认是 **32**，`AlohaOutputs` 最终只取前 **14** 维作为 RoboTwin 双臂关节/夹爪动作；先前将模型内部维度写成 14 是错误的。`LeRobotAlohaDataConfig(use_delta_joint_actions=True, adapt_to_pi=False)` 把训练输入的关节动作变为相对当前 state 的增量、夹爪仍为绝对值；推理输出经 `AbsoluteActions` 逆变换后返回 14 维绝对动作，故仿真控制器执行绝对关节目标并不与训练增量表示矛盾。测评服务使用对应任务的发布 `norm_stats`，不能将本项目自训 checkpoint 的预处理配置替代进去。代码中的 30,005 步是发布训练**配置**，不是从每个 checkpoint 独立读取的实际最终步数；不能仅凭此行证明各权重精确训练步数。
- 已排队 `handover_block` 的 **1 条 smoke（不计正式分母）**，初始队列 PID `1880835`，日志 `control/handover_smoke_queue.log`；只有下载器退出且日志含 `all selected task files verified` 才会执行。启动器还会重新计算每个文件 SHA256、核查 GPU4 与端口 18458。smoke 使用 `demo_clean`、seed 31000、H50、频率字段 50、800 步上限。smoke 成功后仍需检查记录与日志，再启动 50 条正式测评；此时不能将排队等同于测评通过。
- A800 上第二个已预筛候选 `place_can_basket` 的独立下载进程初始 PID `1882037`，日志 `control/download_place_can_basket.log`，与 `handover_block` 写入不同任务子目录；当前仅是下载中，不能记作校验通过。启动前数据盘余量约 287 GB。两项下载均只使用远端网络，不经本机中转。
- 2026-10-24 11:52 CST：`handover_block` 新副本下载器输出 `all selected task files verified`；独立 runner 再做文件哈希校验并已启动 GPU4/18458 服务，smoke seed31000 正在仿真。仅当 smoke 进程退出、`control/state.json` 为 `smoke_complete` 且唯一 smoke 记录身份正确时，独立队列（初始 PID `1886502`，日志 `control/handover_formal_queue.log`）才会启动 50 个无预筛 seed。任何错误均不把 smoke 当作策略失败，正式成功数仍待结果。
- 2026-10-24 11:56 CST：`handover_block` smoke 产生唯一记录，seed31000、281步、`success=true` 且 `final_check_success=true`，没有专家预筛。该 1 条不代表成功率。队列已验证 `smoke_complete` 并启动正式 runner（初始子 PID `1887038`），在重新核验 checkpoint；正式 50-seed 成功数仍待结果。
- `place_can_basket` 下载器已输出 `all selected task files verified`，28/28 文件达到发布清单大小、总计 12,441,564,513 字节；独立 runner 将再校验 SHA256。A800 GPU2 在启动前利用率 0%、空闲约 72 GB，但有 root 所有的常驻 VLLM 服务约 8.6 GB；此实验只使用剩余资源，不停止或修改该服务。smoke 初始 PID `1892892`，端口 18459、原生 H50、频率字段 50、`demo_clean`、无专家预筛、700 步上限、seed31000；正式队列初始 PID `1893801`，只有 `smoke_complete` 和唯一记录验证通过才会启动。两组正式测评都使用 seeds31000–31049，但任务不同，不跨任务合并成一个成功率。
- 2026-10-24：`place_can_basket` 首次 smoke 在 `setup_demo`、策略动作开始前失败，日志明确缺 `071_can` 和 `110_basket`，继而 `basket.get_pose()` 报空对象。A800 原 `RoboTwin/assets/objects` 没有这两项；Ubuntu `/media/ruio/hdd/pace_pi05_eval_20261022/assets/objects/` 中有完整的网格、碰撞与元数据。仅将这两项约 84.46 MB 从 Ubuntu **直传** A800，不经本机数据中转；`rsync -acn` checksum 对比无差异。未覆盖或删除其他资产。smoke 重跑 PID 初始 `1963322`，同 seed、提示词、H50、GPU2/18459、频率和步数上限；新结果待核验。此前失败的正式队列已退出，不能当成正在跑的正式测评。
- 2026-10-24 16:20 CST：`handover_block` 无预筛正式测评 `control/state.json` 为 `complete`，`aggregate_summary.json` 为 `complete=true`，seed31000–31049 **50 个唯一 seed**，成功 **25/50=50%**。这是此批候选的直接成功率证据，不与 Ubuntu 专家缓存场景的 30/50 合并分母。还需逐 seed 分析失败阶段、验证专家中间态救回，才能确认实际 DAgger 可用性。
- `place_can_basket` 重跑 smoke 于 15:47 CST `smoke_complete`，唯一记录 `seed=31000`、`success=false`、`policy_steps=700`、无专家预筛；这证明仿真和服务链路已跑通，不是正式成功率。已另启正式 runner，初始 PID `2002368`，同提示词、H50、GPU2/18459、50 个固定 seed、无专家预筛；正式结果须等 `aggregate_summary.json` 完整后判定。
- 下一候选 `put_object_cabinet` 的任务专属公开权重在 A800 独立下载（初始 PID `2004472`，仅 A800 出站经 hf-mirror），未完成前不记为已校验。任务脚本会随机选取柜体 `036_cabinet/46653` 与十类物体；A800 此前缺少全部所需模型，已从 Ubuntu 现存公开测评环境直传对应 11 项共 448,161,962 字节、424 个文件，`rsync -acn` checksum 对比无差异；未覆盖或删除原资产。正式评测须待下载 SHA256、smoke 和配置核验。
- 为识别 `handover_block` 失败阶段而非只看最终二值成功率，独立诊断脚本增加默认关闭的 `--trace-handover-state`：每个完整 H50 chunk 边界记录方块/目标位姿、两臂末端距离、夹爪开合与接触布尔值，绝不回馈策略或改变正式分母。A800 使用新控制脚本路径与新输出目录回放正式失败 seed31002，GPU4/18460，初始 PID `2007042`；需验证 `handover_state_trace.jsonl` 与正常终止后才用这些状态校准阶段阈值和设计救回，不能从原正式 50 条二值结果臆测失败阶段。原正式评测脚本未被远端覆盖。
- 先从正式逐 seed 记录得到一个**诊断线索而非阶段结论**：25 条成功 episode 在 269–293 步结束（中位数 281），25 条失败 episode 全部运行到 800 步上限；只有轨迹状态追踪能区分“始终未抓住”“交接失败”“放置失败”等原因。
- `handover_block` seed31002 的独立状态回放在第 100 步左手抓持、第 200 步右手抓持，第 250 步方块脱离两手并跌回左侧，第 300 步后位姿稳定在约 `(-0.251,-0.073,0.770)`；它是**一条失败样本**，不能外推所有 25 条失败。保守阶段分类器 `RoboTwin/scripts/handover_block_recovery.py` 已区分左手持有、双手持有、右手持有、掉落待稳定、左侧稳定掉落可尝试重抓及不确定/不可达分支，另有动作后抓持核验和原任务 `check_success` 判定；本地仅完成语法及 3 个纯逻辑单测，尚未通过真实中间态救回 smoke，不能称为可用 HIL 接管。
- `put_object_cabinet` 下载日志现已输出 `all selected task files verified`；其发布配置名 `pi05_robotwin_put_object_cabinet`、任务专属 `norm_stats` 为 `put_object_cabinet/assets/robotwin/norm_stats.json`。A800 任务自带 `_eval_step_limit.yml` 为 700，使用 `description/task_instruction/put_object_cabinet.json` 的 `full_description` 固定提示词、原生 H50、无预筛 seed31000，GPU4/18461 的 smoke 初始 PID `2015120`；尚待有效记录验证，再决定正式 50 seed。不得把下载成功或服务启动当成策略成功。
- `stack_bowls_three` 的任务脚本只依赖 `002_bowl` 模型（模型 ID 3）；26,810,012 字节从 Ubuntu 既有资产目录直传 A800，`rsync -acn` checksum 对比无差异。A800 任务专属公开权重下载初始 PID `2019715`，仍待逐文件 SHA256 结果；任务自带步数上限为 1200。其三次放碗的阶段结构清晰，但未完成模型 smoke/无预筛成功率/中间态救回，不因结构直觉就列为最终入选。
- `put_object_cabinet` 的 seed31000 smoke 已 `smoke_complete`，唯一记录 `success=false`、`policy_steps=700`、无技术错误；它只验证了环境与服务，不能估算成功率。正式 50-seed runner 初始 PID `2021788`，GPU4/18461，复用 smoke 记录但正式输出为独立 `formal/`，待 `complete=true` 与 50 唯一 seed 核验。
- `handover_block` 独立失败 seed31002 状态回放完成，和原正式结果一致：800 步超时、未成功。针对其第 250 步掉落、第 300 步后稳定在左侧的具体情形，新增**诊断专用**专家后缀执行器：先检查当前方块仍在可达工作区及连续快照稳定，再依真实抓持关系从左手重抓/续交接/右手放置等分支启动，动作后重新核验抓持并以原任务 `check_success()` 判定。GPU7/18462、全新输出目录 `results/handover_block_rescue_seed31002_step300/` 的单条中间态救回 smoke 初始 PID `2025421`；它会在策略第 300 步停下尝试专家救回，**不属于正式自主成功率分母**，结果待验证。
- 上述第 300 步救回 smoke **没有触发专家**：同一个场景 seed31002 在该次独立回放中第 280 步由策略自主成功，记录 `expert_intervention=false`，没有 `rescue_result.json`。原正式与首次追踪回放均为 800 步失败；三次初始方块位姿完全相同，但第 50 步起轨迹出现微小分歧，之后放大。公开 PACE `Policy` 默认从 JAX `key(0)` 初始化 RNG，但这里只能证明**场景 seed 不足以保证整条策略+仿真轨迹严格重放**，不能把原因未经验证地归咎于某一模块，也不能把这次当救回成功。另启动诊断专用 `seed31002_step150`（初始 PID `2031497`，GPU7/18462）在第 150 步、通常为左手持有阶段接管，以先验证专家后缀执行链路；仍不纳入正式自主成功率。
- `stack_bowls_three` 下载器已输出 `all selected task files verified`，发布配置为 `pi05_robotwin_stack_bowls_three`、归一化统计 `stack_bowls_three/assets/robotwin/norm_stats.json`。A800 GPU1 上保留 root 所有 VLLM 服务（PID `754968`，约 9.25 GB），只用余下显存；原生 H50、任务自带 1200 步上限、`full_description` 固定提示词、无预筛 seed31000 smoke 初始 PID `2037179`，端口18463。尚待唯一有效记录与无基础设施错误验证。
- `handover_block` 的 seed31002、第 150 步接管诊断已完成：分类器判为 `resume_handover`（左手持有、右手未持有），专家后缀执行阶段 `[3,3,4,5,5,6]`，规划标志为真，原任务 `check_success()` 为真，记录 `expert_intervention=true`、`rescue_success=true`。这是**单条中间态 smoke**，不进入公开模型的 25/50 自主成功率，也不证明所有失败状态可救回。下一条预先选定的正式失败 seed31004 在第 300 步做掉落分支独立 smoke，初始 PID `2042905`，GPU7/18462；若回放提前自主成功或状态不符合，必须照实记为未触发，而不是挑成功轨迹替代。
- `stack_bowls_three` 的 seed31000 smoke 已 `smoke_complete`，唯一记录 453 步、`success=true`、原任务 `final_check_success=true`，无基础设施错误；这仍不是成功率估计。正式 50-seed runner 初始 PID `2046699`，GPU1/18463，与 smoke 结果目录分开，待完整核验。
- 下一候选 `move_can_pot` 官方 π0 Easy/Hard 为 58/21，任务原始上限 400 步；其 `060_kitchenpot`、`105_sauce-can` 两项资产已从 Ubuntu 直传 A800 共 60,146,645 字节、1472 个文件，checksum 空差异复核。任务专属公开模型下载初始 PID `2049846`，尚未宣称已校验或启动测评。
- `handover_block` 预定 seed31004、第 300 步的**自然掉落分支** smoke 已结束且**救回失败**：方块在约 `(-0.034,0.147,0.770)` 稳定，双手均未抓持，分类器选 `restart_source_grasp`，执行 `[1,1]` 后 `plan_success=false`、原任务 `check_success=false`，原因记录为 `stage_1_planning_failed`。这说明仅凭“方块仍在大工作区内”不能保证专家规划能从该姿态重抓。已在重抓前增加双臂回初始位姿的显式准备阶段，并把动作构造失败与运动规划失败区分开；本地 4 个纯逻辑单测通过。新的独立重试目录为 `results/handover_block_rescue_seed31004_step300_home/`，初始 PID `2053012`，GPU7/18462；由于同 seed 回放不严格相同，必须记录它实际遇到的物体状态和分支，不能把跨运行结果视为同一状态的严格 A/B 对照。
- 上述 seed31004、第 300 步的双臂回初始位姿重试已完成：这次回放的方块位姿同样约为 `(-0.034,0.147,0.770)`，实际触发 `restart_source_grasp`，`expert_intervention=true`，执行阶段 `[1,1,1]` 后仍为 `stage_1_planning_failed`、`rescue_success=false`。故不能把掉落但处于大工作区的状态视为自动可救回；下一步需核实动作规划失败的具体几何/碰撞原因，或将该状态保守标为不可自动恢复。左手已抓持状态的一次成功与此失败分支须分别报告，均不得混入自主 25/50。
- 已核实两次掉落重抓探针中方块的长轴近乎水平（四元数换算的长轴竖直分量约为 0），而原任务 `grasp_actor` 规划假设从正常姿态抓取。为避免再把这种状态错判为自动可恢复，`handover_block_recovery.py` 对**无人抓持、已静止**的方块新增姿态门控：缺失/无效四元数则拒绝；长轴竖直分量绝对值低于 0.75 则返回 `fallen_sideways_requires_specialized_recovery`，不执行动作。它不是“永久不可救回”的结论，只表示现有原任务重抓后缀未经证实；该分支需要专门的侧躺重抓方案。新增单测后本地 5 项纯逻辑测试、语法检查通过，A800 诊断模块与本地 SHA256 一致（`fc98e279319452598b85fb27e865a1f8984b20416b0b63c41365999e3f7be69c`）；尚未用侧躺中间态做新 smoke，因此不把这项代码门控冒称真实救回成功。
- 把 A800 已保存的**真实**快照重新喂给上述诊断模块核验：seed31004 第300步掉落侧躺（长轴竖直分量约 `3.24e-6`）现在返回 `ready=false`、`fallen_sideways_requires_specialized_recovery`；seed31002 第150步左手持有仍返回 `ready=true`、`resume_handover`。这是状态分类器的离线回放核验，并非新一条动作层救回 smoke。
- `move_can_pot` 模型下载日志已输出 `all selected task files verified`；独立 runner 在 GPU7/18464 使用发布的任务专属模型及 `norm_stats` 再核哈希。原任务上限 400 步，H50、频率字段50、`demo_clean`、无预筛 seed31000 的技术 smoke 初始 PID `2062259`，`control/state.json` 到 `smoke`。只有 smoke 产生唯一有效记录且无基础设施错误后，才启动正式固定 50 seed；smoke 不用于选择任务或估计成功率。
- `rotate_qrcode` 的 `070_paymentsign` 和 `place_burger_fries` 的 `008_tray`、`006_hamburg`、`005_french-fries` 已从 Ubuntu 挂载数据盘 `/dev/sda1` 直传 A800，共 167,682,590 字节；`rsync -acn` 校验无差异，未覆盖或删除既有资产，传输不经过本机手机流量。两项仅完成资产准备，公开权重与正式测评尚待进行。
- A800 又开始从发布 manifest 独立下载 `rotate_qrcode`（初始 PID `2065237`，日志 `control/download_rotate_qrcode.log`）和边界候选 `place_burger_fries`（初始 PID `2069501`，日志 `control/download_place_burger_fries.log`），各任务写入独立子目录，待逐文件 SHA256 通过后才可做技术 smoke 与正式测评；目前不得称为已验证 checkpoint。选择它们是依据此前官方 π0 与阶段可救回性预筛，而不是 smoke 成败。
- `move_can_pot` 技术 smoke 已 `smoke_complete`：唯一记录 seed31000、执行至原任务上限400步、`success=false`，但无技术错误。这不是正式成功率。原 smoke 进程退出、18464 端口释放、GPU7 空闲后，独立正式 runner 初始 PID `2070760` 启动，仍按固定 seed31000–31049、H50、无预筛；启动时重新 SHA256 校验任务权重与 `norm_stats`，目前尚无正式成功率结论。
- `rotate_qrcode` 下载器已退出并输出 `all selected task files verified`；原任务步数上限为400。A800 GPU0 在原有 root 进程占约30 GB 显存且有约50 GB 空余时，独立任务 smoke 初始 PID `2095699` 用剩余显存启动服务（端口18466、原生 H50、频率字段50、`demo_clean`、无预筛 seed31000），不触碰该 root 任务。技术 smoke 的有效记录与无 OOM/仿真错误尚待核验；即使 smoke 成功或失败，也不作为候选选择依据。
- `place_burger_fries` 下载器也已退出并输出 `all selected task files verified`；其原任务上限为500步，资产已校验。候选模型文件在 A800，但还需独立 runner 复核、技术 smoke 与正式50 seed；目前不因下载成功就给出成功率，也不抢占仍在执行 `rotate_qrcode` 的 GPU0。
- `rotate_qrcode` 的技术 smoke `smoke_complete`，唯一无预筛记录 seed31000、143步、原任务 `final_check_success=true`；它不进入正式分母。原 smoke 进程退出、GPU0/18466 端口恢复后，经 Ubuntu→A800 控制连接启动正式 runner，初始 PID `2105910`，仍在 A800 GPU0 与既有 root 任务共用剩余显存；固定 H50、seed31000–31049、原任务上限400，当前 `control/state.json` 已到 `formal`。Mac 直连 A800 SSH 当时连续关闭，但 Ubuntu 中转控制连接成功；不能因单条 SSH 失败擅自重启已运行任务。
- `place_burger_fries` 的公开权重以 A800→Ubuntu **服务器间直传**（Ubuntu rsync 初始 PID `399317`，由本机 SSH agent 仅代理认证，大文件数据不经本机）准备独立测评；传输尚未完成，不能声称 Ubuntu checkpoint 已验证。Ubuntu `/media/ruio/hdd` 挂载为 `/dev/sda1`、剩余约1.1TB，GPU0 检查时约23.3GB空闲；`008_tray`、`006_hamburg`、`005_french-fries` 从同机发布资产目录补到 RoboTwin 对象目录约130MB，`rsync -acn` 校验无差异。两机 `place_burger_fries.py` 与 `demo_clean.yml` SHA256 完全相同，发布 `openpi_pace/configs.py` SHA256 相同；Ubuntu XPolicyLab `model.py` 仅多了本项目候选采样 API，不影响单候选 `get_action` 路径。已把通用 runner/evaluator 的小体积代码和 manifest 复制到 Ubuntu 隔离控制目录，语法/导入检查通过；Ubuntu 发布版服务 venv 缺少 `msgpack-numpy`，已补与 A800 相同的 `0.4.8` 并验证导入。通用 runner 新增可配置 `--min-free-gpu-mib`，默认 40000 保持 A800 原门槛，Ubuntu 需显式设置适合24GB卡的门槛并写入 `protocol.json`；正式 smoke 尚未执行。
- **更新**：上述 A800→Ubuntu 模型 rsync 已正常退出，36 个清单文件、12,441,580,934 字节完整传输；Ubuntu 独立 runner 重新逐文件 SHA256 核验通过，`pi05_robotwin_place_burger_fries` 配置可解析为 H50、内部动作维32、关节增量训练与 `adapt_to_pi=False`。Ubuntu GPU0/18472、显存预检门槛18,000MiB、`demo_clean`、固定说明词、频率字段50、500步上限、无专家预筛的技术 smoke 为唯一 seed31000、230步原任务成功、`smoke_complete`；它不是正式成功率。smoke 进程/服务退出且端口释放后，独立正式 runner 初始 PID `404292` 启动，seed31000–31049；目前正在二次权重校验/准备阶段，须等 50 个唯一正式 seed 和 `aggregate_summary.json complete=true` 才能给出选题结论。
- 2026-10-25 00:06 CST 只读复核：六个正式 runner 的 PID 与命令均仍在运行，Ubuntu `place_burger_fries` 已产生 14 个正式记录；A800 `place_can_basket`、`put_object_cabinet`、`stack_bowls_three`、`move_can_pot`、`rotate_qrcode` 分别产生 29、17、11、25、12 个正式记录。它们均**尚无完整 aggregate**，这些中途数量不是成功数、也不能据此排名；不重启或干扰现有任务。
- `handover_block_recovery.py` 新增 `iter_expert_suffix()` 分阶段接口：每次推进至多执行一次脚本动作并返回阶段事件，控制器可在动作边界停止专家后缀、交还策略；旧 `run_expert_suffix()` 继续消费完整后缀以保持诊断调用方式。2026-10-25 本地语法检查与 7 项逻辑测试通过，包括阶段边界可暂停和一次性包装器阶段顺序。**尚未将这个新迭代接口接入 HG-DAgger 的 `i/x` 人工控制，也尚未对新接口做真实仿真 smoke**；此前 seed31002 左手持有状态的一条成功 smoke 使用的是旧一次性执行器，不能替代新接口验证。
- 2026-10-25 JG 只读资源复核：SSH 可达，RTX 5090 D v2 的 24,455 MiB 显存中约 41 MiB 使用，`/home` 尚有约 139 GB；现有 `handover_block.py` 与 `demo_clean.yml` 的 SHA256 均与本机工作树一致。JG 的 `/home/ruihao/robotwin_hil/RoboTwin/assets/objects` 仅有基础对象目录，未部署本轮公开模型/候选测评脚本；非交互 SSH 默认 `python` 不在 PATH。进一步定位到独立环境 `/home/ruihao/robotwin_hil/conda/bin/python`，已验证可导入 Sapien 3.0.0b1 和 NumPy 1.26.4，且可导入现有 evaluator 的任务配置加载器。故 JG **可用于无需公开模型的本任务专家阶段 smoke**，但尚未验证为可直接运行公开 π0.5 测评的完整环境；不复制 12 GB 权重来赌推理链路。
- 为后续 `handover_block` 交互接管复用同一观测口径，诊断 evaluator 原内嵌的 `handover_snapshot()` 已提取到独立 `handover_block_recovery.py`，不依赖模型包或 NumPy；它采集方块/目标与末端位姿、夹爪开合、真实接触、相对距离和原任务成功标志，并对非有限位姿直接报错，不把状态缺失算策略失败。evaluator 仅在开启诊断追踪时延迟导入此接口，不改变其它候选的默认测评路径。本地语法检查与 8 项逻辑测试通过，其中新增测试验证快照字段可直接驱动左手持有阶段分类；这仍不是新接口的真实仿真 smoke。
- 新增模型无关的 `smoke_handover_block_recovery.py`，在 JG 的匹配版 `handover_block`/`demo_clean` 环境中真实执行专家抓取和抬升后，从左手持有中间态调用分阶段后缀。seed31002 的阶段事件为 `[3,3,4,5,5,6]` 后 `done`，每步 `plan_success=true` 且原任务 `check_success=true`；seed31003 配置只执行 1 个阶段动作，实际仅返回首个 stage-3 事件、`handback_ready=true`，此时 `final_check_success=false`，没有误报任务成功。两条详细 JSON 已同步本机 `outputs/handover_block_staged_smoke_20261025/`。**这两条由专家构造中间态，没有执行公开策略，也没有验证交还策略后的闭环；不能算 HIL 成功率或替代真实策略失败态接管验证。**
- `move_can_pot` 原 runner 在正式 seed31028 的 `setup_demo` 阶段报 `UnStableError: 105_sauce-can`，仅 31000–31027 的 28 条有完整记录；该 seed 在**任何策略动作之前**被仿真稳定性门槛拒绝，不能计作策略失败。单 seed 隔离诊断（A800 `/data-training/.../diagnostics/move_can_pot_seed31028_settling_retry_scratch.json`，小体积 JSON 与原错误日志已复制到本机 `outputs/move_can_pot_diagnostics_20261025/`）用任务专属临时/缓存目录绕开已满的根分区后，按原代码原阈值连续做两次 `check_stable()`，两次都报 `105_sauce-can` 不稳定；故不采用“多等一轮就算有效”的假设，也未修改物理场景。A800 根分区 `/` 当时 100%（0 可用），`/data-training` 仍约 215 GB 可用；其它候选 runner 当时仍活动，不能清理他人文件。为完成 50 个**有效场景**且不靠专家成功率预筛，`run_public_pi05_candidate.py` 新增默认关闭的严格替换模式：仅识别 `setup_demo` 抛出的 `UnStableError`（策略前），逐 seed 留 `excluded_infra_seeds.json`，顺延最多 10 个 seed；任何其它错误依然停止，原 28 条记录经身份复核后复用。此模式的实际 seed 集与其它任务的固定 31000–31049 不完全相同，最终必须同时报告排除与替换，不能做伪配对比较。本地分类逻辑单测和语法检查通过；独立远端 runner 初始 PID `2208873` 于 2026-10-25 00:28 CST 启动，GPU7/18464，重新验证公开权重和 norm_stats；00:29 状态已至 seed31029，`excluded_infra_seeds.json` 明确记录 31028，结果仍待完成。
- Ubuntu `place_burger_fries` 正式 runner 已以 `control/state.json phase=complete` 正常退出；独立 `audit_public_pi05_candidate.py` 将小体积 aggregate、protocol 与逐 seed JSON 同步本机后复核：固定 seeds31000–31049 恰好 50 个、全无重复，原任务成功 **44/50=88%**，每条 `success` 与 `final_check_success` 相同，正式目录每 seed 一条、无技术错误，冒烟目录不计正式分母。此为**原任务代码的判定成功率**，而任务 `check_success()` 只核汉堡/薯条与托盘目标功能点的 XY 距离及双手松开，不核高度/接触，因此尚不能称为 44 次物理有效放置。即便全为有效，也只有 6/50 失败，按当前协议收集 30 条 HIL 可能需很多 rollout；最终保留/清理须待质量与其它任务比较，不现在删除已校验权重。
- A800 `handover_block` 的 aggregate、protocol 和 50 条逐 seed 小 JSON 也已同步本机，用同一独立审计脚本复核为 50 唯一 seed、**25/50**、逐 seed `success=final_check_success`，不再只依赖 runner 自报汇总。Ubuntu 隔离目录的 `place_burger_fries` seed31000 物理有效性诊断（PID `421927`，`smoke_complete`，不改动正式 44/50）显示终止瞬间及额外 250 次物理静置后，汉堡与薯条都保持托盘接触；对应功能点分别比托盘目标高约 0.0099 m、0.0182 m，原任务成功仍为真。小体积 JSON 已复制到本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/burger_seed31000/`。因此**这一条**没有发现“XY 判定成功但食物悬空”的假阳性；它不能证明另外 43 条判定成功都物理有效，也不改变 88% 原判定成功率偏高、有效 HIL 可能稀缺的选题判断。

### `handover_block` 的接管设计与未完成验证

原任务脚本的语义顺序是：左臂抓红块 → 抬升到双臂交接区 → 右臂抓住 → 左臂松开 → 右臂放到蓝色目标。阶段检测不能读取脚本的程序计数器，因为策略 rollout 没有执行该脚本；应根据红块位姿与姿态、两臂夹爪开合与接触、末端到红块距离、目标位置推断。现有保守分类器已区分左手持有、双手持有、右手持有、掉落待稳定、左侧静止且直立的候选重抓，以及未知/不可达状态；侧躺掉落会拒绝自动执行。每阶段的专家恢复必须从**当前**物体和夹爪状态规划，不能简单重播原 `play_once()`。现有专家后缀逐动作核验规划与抓持，最终运行原任务 `check_success()`；单条策略左手持有状态已由旧后缀在真实仿真中救回，侧躺掉落分支两次失败。新分阶段接口另在 JG 专家构造的左手持有中间态通过一次完整后缀及一次单动作暂停 smoke；Ubuntu 还在两条真实策略失败 seed 上完成两动作专家介入后策略交还成功。自动 `i/r/x` 诊断路径已接入并验证，仍需验证更多真实失败态、人工键盘操作与正式新 seed 的有效 HIL 采集，因此不能称为完整可用的 DAgger 接管脚本。

- 2026-10-25 新增并实测隔离的策略 handback 诊断入口：`eval_public_pi05_candidate.py --recover-at-step 150 --handover-handback-stage-actions 2` 在真实策略 chunk 边界分类，最多执行两次分阶段专家动作，重新检查物体/抓持状态；仅当原任务尚未成功、规划有效且阶段分类仍可接续时重置策略服务并交还控制。runner 仅允许 `--smoke-only` 使用该入口；本机语法检查和 8 项恢复逻辑测试通过。Ubuntu 使用公开任务专属 `handover_block` 权重的隔离 smoke（GPU0/18473）已 `smoke_complete`、进程退出、端口释放；protocol 记录 `demo_clean`、seed31002、H50、频率字段50、800步上限、无专家预筛。此 seed 在 A800 无干预正式测评中跑满800步失败；诊断中策略运行150步后进入 `resume_handover`，专家执行两次 stage-3 动作且规划成功，重新分类仍为可接续的 `resume_handover`，成功交还策略；交还后策略再执行134步，最终原任务 `check_success=true`，总计284次策略动作。`rescue_result.json`、单条 `eval_records.jsonl` 和 protocol 已同步本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/handover_seed31002_step150_handback2/`。这是真实策略失败 seed 的**单例**接管-交还闭环证据，不是 HIL 成功率估计；介入记录标 `expert_intervention=true` 且 smoke 不进入无干预的 25/50 正式统计。仍未接入 HG-DAgger 的 `i/x` 人工控制、有效 HIL 保存和多失败态验证。
- 第二个先前未做救回诊断的无干预失败 seed31012 复用相同协议与公开 checkpoint：A800 正式无干预记录在800步失败；Ubuntu 隔离 smoke 的策略第150步进入 `resume_handover`，两次 stage-3 专家动作规划成功、交还门控为真，后续策略146步后原任务成功，总计296次策略动作。`control/state.json=smoke_complete`，诊断进程退出、18473端口释放；protocol、`rescue_result.json` 与单条记录已同步 `outputs/robotwin_dagger_screen_20261024/diagnostics/handover_seed31012_step150_handback2/`。两条真实策略失败态的同分支复现仍不能估计总体 HIL 成功率，也未覆盖侧躺掉落等被拒绝分支；两条介入结果均不进入无干预 25/50。
- 接入 HIL 原生数据接口的前置核对：本机与 Ubuntu 当前版本的 `Base_Task._take_picture()` 将 `current_stage_id` 写入每帧 `task_metadata.stage_id`，并将 `current_control_source` 写入 `control_mask`；`take_action()` 本身不保存策略帧，既有 HG-DAgger 在策略端显式按 `save_freq` 调用 `_take_picture()`，专家 `move()` 会在动作轨迹中保存。`handover_block_recovery.iter_expert_suffix()` 现于每次 `move()` 前设置 `current_stage_id` 和 `current_control_source="hil"`；诊断策略端在执行动作前恢复 `current_control_source="policy"`。纯逻辑测试已逐次验证执行时的阶段序列 `[3,3,4,5,5,6]` 与全部专家来源标签，语法检查通过。**JG 当前 `_base_task.py` 是旧版本，没有 `control_mask`；首次在 JG 跑原生帧 smoke 虽完成专家动作并生成225个带阶段标签的 pkl，却因缺逐帧来源 mask 而被验证器拒绝，不把它作为合格数据，也不为此改动共享仿真代码。**
- 改在 Ubuntu 当前版本做模型无关的专家前缀→分阶段后缀原生帧保存 smoke：隔离目录 `/media/ruio/hdd/robotwin_dagger_screen_20261024/diagnostics/handover_native_frames_seed31002_v2/`，seed31002 原任务成功；`seed31002_raw/frames/` 有217个连续 pkl（约200 MB），旁边 `episode.json` 各有217项 `control_mask` 与 `stage_ids`，来源全部为 `hil`，阶段依次为0、3、4、5、6；独立抽查索引0、1、80、140、200、216的 pkl 内阶段标签均与 sidecar 对齐。仅小体积 smoke 结果和 episode 元数据同步本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/handover_native_frames_seed31002_v2/`，原始帧留 Ubuntu。这证明专家阶段标签/来源能按帧保存，但中间态仍由**专家前缀**构造，并未保存真实策略失败前的混合 policy/HIL 轨迹；交互 `i/r/x`、有效 HIL 筛选和数据导出仍未验证。
- 新增独立 `hg_dagger_handover_block.py` 采集器：策略 chunk 内每动作可响应 `i/x/q`，`i` 后从当前物体/夹爪/接触状态保守分类并丢弃剩余 chunk；专家后缀每个语义动作可按 `r` 请求交还，须重新分类可接续、规划有效才交还，否则继续专家；`x` 立即排除。原生帧使用 `control_mask` 区分策略与专家，保存门槛为真实接管、有专家帧、原任务最终成功、规划未失败及关节合法；没有专家成功的自主轨迹不作为 HIL 保存。`--headless-smoke` 的自动 `i/r` 钩子只用于诊断，正式采集必须用新的非测试 seed 和人类输入。Ubuntu 以 seed31002（**冻结测试 seed，只诊断，不进训练**）在真实公开 π0.5 策略服务上跑一次自动 `i` 于策略第150步、两次专家 stage-3 动作后 `r`：`session.json` 完成，原任务成功、关节合法，54帧原生 pkl（策略19、HIL35），控制分段为 policy[1,11)、hil[11,46)、policy[46,54)；抽查边界帧来源与阶段标签吻合，每帧关节向量14维且含相机观测。服务进程已按 PID 身份终止、18473 端口释放。原始帧约49.9MB仅留 Ubuntu `/media/ruio/hdd/robotwin_dagger_screen_20261024/diagnostics/handover_hil_collector_auto_seed31002/data/raw/episode_0000000/`，小体积 session/rollout/episode 元数据同步本机同名 `outputs/.../diagnostics/handover_hil_collector_auto_seed31002/`。这验证**自动触发的混合记录/保存路径**，尚未证明人工键盘响应或正式新 seed 采集；自动 `x` 排除路径由下述独立 smoke 验证。为避免测试 seed 数据泄漏，新采集器把后续 headless smoke 标 `diagnostic_only=true`、`accepted_for_training=false`；本次已生成的原始诊断 episode 另加 `DIAGNOSTIC_ONLY.json` 明确排除训练。
- 对 `x` 排除也做了独立真实仿真 smoke：同一冻结 seed31002、自动 `i` 于策略第150步、专家两动作后自动 `r`，在再执行一条策略动作到151步时触发测试专用 `x`。结果有35帧 HIL 记录但 `aborted=true`、`valid_hil=false`、`saved_valid_hil=0`、`raw_dir=null`；原生 `.cache/episode0` 已由采集器清除，输出目录只有 rollout/session JSON，**没有 raw 训练轨迹**。小体积结果在本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/handover_hil_collector_auto_x_seed31002/`；该次独立服务 PID 核对后已停止、18473 端口释放。这验证自动 `x` 的数据排除路径，仍不等于人用键盘操作已通过。
- 已将改进后的 `handover_block_recovery.py` 与 `hg_dagger_handover_block.py` 部署到 Ubuntu `/hdd/robotwin-hil-enhanced-sampling/RoboTwin/scripts/`，两文件原先在共享仓库均不存在，远端语法检查通过；SHA256 分别为 `b22725b34996018e0330b9abc17381d8b5ece27548ddebf1f3edb363b7fbae29`、`eb532a7a7a3445ced573312cf77676613add381444545acca2b6878ed947f561`。部署不等于人工采集已运行；正式运行仍需有人在 GUI 控制 `i/r/x`，并用非测试 seed 与独立输出目录。
- 2026-10-25 检查离线 `export_hg_dagger_dataset.py` 时发现它原来会遍历全部 raw episode，即使 episode 标了 `diagnostic_only` 仍可能导出。现已在导出入口排除 `DIAGNOSTIC_ONLY.json`、任一层 `diagnostic_only=true` 或 `accepted_for_training=false` 的 episode；旧版无此字段的正式数据仍可导出。新增 4 项轻量测试覆盖测试元数据、旧诊断标记、正式数据及导出主流程的诊断/正式混合目录（用桩替代 HDF5 编码器），语法检查及测试均通过。Ubuntu `/hdd/robotwin-hil-enhanced-sampling/RoboTwin/scripts/export_hg_dagger_dataset.py` 原文件 SHA 与本机 Git HEAD 相同后，已原子部署改版 SHA256 `a7df1d0e21b5140c4c351239e940a614dcb30911f3035d616e3b3cbc4a5ecb08`，远端语法检查通过。这是避免冻结测试 seed 混入后续训练的独立防线；尚未用真实帧做 HDF5 编码验证。

### 2026-10-25 真实策略失败态诊断续记

- 最新无预筛公开 π0.5 汇总已完成并复核：`put_object_cabinet` 为 14/50=28%，`stack_bowls_three` 为 36/50=72%；两者的 `aggregate_summary.json` 均 `complete=true`、50条唯一 seed，分别位于 A800 `results/put_object_cabinet_h50/` 与 `results/stack_bowls_three_h50/`。`stack_bowls_three` 的原成功判据仍有高度单侧约束问题，72%仅按 benchmark 原判据报告，物理堆叠质量需另审。
- `place_can_basket` 的修正版重抓态分支在 A800 冻结 seed31000 上再做专家前缀构造 smoke（attempt3–8）。每次均能放篮、释放篮、确认篮子直立稳定并重新抓罐；但 `place_actor` 的直接规划失败，改用 `get_place_pose()` 做抓持补偿并跳过预接近路点后仍在规划阶段失败（attempt8 `reason=alternate_can_placement_failed`，两次 stage-2 plan 均为 false，未释放罐）。最终罐仍接触桌面、未接触篮，原 `check_success=false`。因此接管分类门槛不是当前阻塞点；真正阻塞在恢复后“罐入篮”的运动规划/可达性。安全停止是预期行为。该候选降为储备；这些冻结 seed smokes 不计成功率、不进训练。
- `place_can_basket` seed31000 第250步的隔离恢复 attempt7：接管时罐仍在桌面、另一臂持篮且篮子直立。专家先成功放下篮、成功重抓罐；直接放置规划失败后执行备用放置动作，但最终罐与篮中心 L1 距离约0.495 m且罐仍接触桌面，原任务 `check_success()` 为假。结论：状态有部分可恢复性，但当前放置目标/动作序列不正确，真实策略失败态救回尚未通过。结果保存在 A800 `.../diagnostics/place_can_basket_real_policy_seed31000_recovery_step250_attempt7/`，不计入正式24/50。
- `move_can_pot` seed31000 的真实策略 rollout 在200步触发时已无法恢复：100步罐仍在桌上、闭爪但无接触；150步罐已跌出工作区。更新了**仅隔离诊断副本**的恢复逻辑：若罐仍接触桌面且高度接近目标桌面、锅/目标未明显偏移，就尝试专家重抓，再按原任务阶段2搬运、阶段3放置；越界或状态不明仍拒绝。attempt3 因启动参数引号错误、attempt4 因使用了任务不存在的 `object_start_height` 属性，均在仿真前退出，没有 rollout。attempt5 在100步真实策略状态触发，规划执行两次 stage-1 动作但实际未抓住罐，安全门控停止。**attempt6 在同 seed 第50步触发成功**：此时源臂正接触桌面罐但夹爪未闭合，专家 stage-1 重抓后执行原 stage-2 搬运与 stage-3 放置，三阶段规划均成功，`final_check_success=true`；不计入正式28/50。小体积 `eval_records.jsonl`、`rescue_result.json` 与状态轨迹已同步本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/move_can_pot_real_policy_seed31000_recovery_step50_attempt6/`。
- 新增隔离 smoke 专用 `--task-state-handback-stage-actions`：move_can_pot 在专家恢复指定阶段后，重新检查抓持/任务状态，安全时 reset 策略服务并把控制交回 π0.5。此功能只可和 `--smoke-only`、`--trace-task-state`、`--recover-at-step` 一起用于 `move_can_pot` 诊断。attempt7 在同 seed第50步，专家只执行 stage-1 重抓，门控复核通过后交还策略；策略再运行66步，最终 `final_check_success=true`，原无干预记录仍失败，单例混合控制闭环成功。`eval_records.jsonl`、`rescue_result.json`、阶段轨迹及 protocol 已同步到本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/move_can_pot_real_policy_seed31000_step50_handback1_attempt7/`。这验证自动 handback 路径，不等于人工键盘输入或正式训练数据收集已通过。
- 上述 attempt 都使用冻结 seed31000 进行故障定位，不纳入有效 HIL/训练数据，也不修改无干预评测的成功率。
- 2026-10-25 集成采集器 smoke 已通过：公开 π0.5 在 seed31000 第50步触发自动诊断 `i`，安全分类为桌面罐可重抓；专家只执行 stage-1 重抓，重分类确认可继续后交还策略。策略再运行66步，原任务 `check_success()` 成功；共记录68帧，mask 与 `episode.json` 一致，分段为 policy[1,4)、HIL[4,64)、policy[64,68)，60帧标为 HIL，关节合法。`session.complete=true`、`saved_valid_hil=1`；但因使用冻结 smoke seed，episode 带 `DIAGNOSTIC_ONLY.json`、`accepted_for_training=false`，绝不可训练。小型 JSON/元数据已同步本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/move_can_pot_hg_dagger_collector_seed31000_auto_i50_r1_attempt3/`；原始帧仅留 A800。独立服务已停止、18467 端口已释放。自动 `i/r` 路径已验证，正式非 smoke 运行中的人工键盘 `i/r/x` 尚未验证。
- 2026-10-25 正式候选队列只读进度：最近检查时 `put_object_cabinet` 已有41个唯一 seed 记录、11次原任务成功；`stack_bowls_three` 有39个唯一 seed、27次成功，第31039条正在运行（约1009/1200步）。两项仍在运行（GPU4/18461、GPU1/18463）；中途比例不作为最终排名，等待各自完整50 seed与 aggregate 后再更新。
- 2026-10-25 后续只读复核：上述正式候选队列已结束，`put_object_cabinet_h50` 与 `stack_bowls_three_h50` 的 `control/state.json` 均为 `complete`；各自 `aggregate_summary.json` 为 `complete=true`、正式 seed31000–31049 恰好50条且唯一、没有 infra 排除，分别 **14/50=28%** 与 **36/50=72%**。正式 runner/server 均已退出；汇总与 protocol 已同步本机 `outputs/robotwin_dagger_screen_20261024/results/{put_object_cabinet_h50,stack_bowls_three_h50}/`，这两项没有待完成测评。
- 2026-10-25 `move_can_pot` 自动 `x` 取消集成 smoke：公开 π0.5、冻结诊断 seed31000、step10 触发 `x`，10个策略动作后安全结束；`aborted=true`、`valid_hil=false`、关节合法、`raw_dir=null`、`saved_valid_hil=0`，没有 raw episode。GPU4 上自有服务退出、18465 端口释放、显存回到约1 MiB。日志与 JSON 在本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/stack_bowls_three_collector_gpu4_pinned_20261025/move_can_pot_x_only_seed31000/`。
- 2026-10-25 修正三项 DAgger collector 的 smoke CLI 约束：自动 `x` 可独立做 headless smoke；自动 `i` 可单独进入专家完成后缀；只有自动 `r` 才要求此前配置自动 `i`。`move_can_pot`/`handover_block` CLI 回归4项、`stack_bowls_three` CLI 回归4项均在 A800 qplanning/SAPIEN 环境通过；冻结 seed smoke 仍标记 `diagnostic_only=true`、`accepted_for_training=false`。这不代表真人 GUI 键盘或正式数据采集已通过。
- 2026-10-25 新增 `place_can_basket` 与 `rotate_qrcode` 的共享 HG-DAgger 控制器、任务适配器及两个入口脚本；均限定固定50动作chunk、手动 `i/r/x/q`、阶段/控制源标记，并且只有“原任务成功 + 任务物理质量门 + 合法关节”才保存有效 HIL。诊断自动按键仅允许单条 headless smoke，冻结 seed 的 raw 永远 `accepted_for_training=false`。本地、A800 SAPIEN 环境、Ubuntu 与 JG 各自运行同一组10项逻辑/CLI测试并全部通过；自动 smoke 触发器已改成一次性触发，避免拒绝接管后在每个动作边界重复按 `i`。
- 2026-10-25 `rotate_qrcode` 真正策略集成 smoke：公开 task-specific π0.5、`demo_clean` seed31000、H50/频率字段50、无预筛，在策略 step100触发自动 `i`。快照为右手夹持且接触二维码板，分类为 `place_held_qrcode`；任务原生 stage-3 `place_actor(..., constrain="align")` 规划成功，最终原 `check_success=true`、目标 XY 误差0.00328m、二维码板与桌面接触、双夹爪释放、关节合法。采集器 `valid_hil=true`、`saved_valid_hil=1`，但明确 `diagnostic_only=true`、`accepted_for_training=false`；不是正式收集，不是成功率估计，也没有测到策略 handback。JSON/元数据同步到本机 `outputs/robotwin_dagger_screen_20261024/diagnostics/hg_dagger_two_task_collectors_20261025/rotate_qrcode_auto_i100_v1/`；A800 原始帧只留远端。
- 2026-10-25 `place_can_basket` 新采集器 smoke 探针：公开 task-specific π0.5、冻结 seed31000、自动 step250 接管时，严格分类器返回 `can_placement_requires_separate_recovery` 并拒绝执行后缀。该状态与之前已知的“篮子/罐混合阶段”不相符或不满足当前可证恢复条件，故此次短诊断在 step269 被主动停止，没有完成 episode、没有写入 rollout/raw，也不计策略失败。之前已在该任务上多次测得：放篮、放篮释放、重抓罐可以规划，但把罐安全放入篮内仍规划失败或质量检查失败；结论仍是储备，不能开始正式 HIL 采集。
- 本轮在 A800 模拟器环境中发现 `websockets`、`msgpack`、`msgpack_numpy` 未安装到 qplanning Python；仅将可用纯 Python 模块复制进隔离诊断目录 `.../diagnostics/hg_dagger_two_task_collectors_20261025/deps/`，没有改共享虚拟环境。`pytorch3d` 提示缺失但这两条 headless 仿真仍成功进入任务执行。自有策略服务器使用 GPU4/18472，两个 smoke 结束后已停止，端口释放，GPU4显存回到约1MiB；无其他 GPU 进程被触碰。
- Ubuntu 与 JG 的 GPU 在检查时均空闲；Ubuntu 的 `/media/ruio/hdd` 仍为 `/dev/sda1`，约1.1TB可用，但预期任务目录下未找到这两个 task-specific checkpoint。由于 A800 已有权重、资产和已验证的服务器/仿真链路，没有把约12GB权重再传一遍，也没有在两台机器上重复做策略 rollout。两台机器只在 `/tmp/codex_hg_dagger_add2_20261025_1`（Ubuntu）和 `/tmp/codex_hg_dagger_add2_20261025`（JG）的隔离代码副本上独立跑了10项纯逻辑/CLI测试，无 GPU任务或正式数据写入。

### 2026-10-25 新增三项 checkpoint 与 DAgger 接管诊断完成

- 发布 checkpoint 固定H50结果最终复核：`aggregate_three_tasks.json` 为 `complete=true`；三项各50条、scene seed唯一、技术/基础设施错误0，且 `metadata.json` 均记录 `demo_clean`、`expert_check_mode=cached_scenes`、`prediction_horizon=50`、14维 absolute_joint。pick为16/50、put为38/50、stack为32/50。汇总及逐seed小 JSON 已同步到本机 `outputs/robotwin_dagger_screen_20261024/results/three_added_public_pi05_h50_20261025/`；A/B/C按任务分开的 `metadata.json` 和 `episodes.jsonl` 可复核逐 seed 结果。场景集合遵循各任务缓存 seed0 前50条，三个任务的具体scene seed列表不要求彼此相同；不能与先前无预筛评测直接合并。
- `put_bottles_dustbin` 的真实策略失败场景 seed100002 使用与正式评测完全相同的指令，在step100自动诊断接管。早先冒烟揭示两处脚本问题：自由但远离物体的闭合夹爪被过度保守拒绝；改进门控后，又发现投桶动作构造器错误地先调用 `task_env.move()`，把其布尔返回值再传给 `_run_stage`，导致 `TypeError: 'bool' object is not subscriptable`。修复为动作构造器仅返回 motion。最终 `session.complete=true`、`valid_hil=true`、任务原生 `check_success=true`、严格物理门 `strict_success=true`、`expert_failed=false`；逐瓶接管后全部规划阶段成功。该 frozen-seed smoke 的 `diagnostic_only=true`、`accepted_for_training=false`；原始帧仅留 Ubuntu，JSON记录已同步至 `outputs/robotwin_dagger_screen_20261024/diagnostics/three_added_tasks_dagger_smoke_20261025/put_bottles_dustbin/`。
- `stack_blocks_two` seed100000 在修复前遇到“顶块未抓住但夹爪闭合”时被误判；快照里左夹爪实际接触并持有已到位的底块，右臂才是待抓顶块的安全臂。分类器现在只在传感器明确证明同一夹爪正持有另一块已知方块、且其与当前顶块距离超过抓取阈值时放行；未知/近距离闭合状态仍拒绝。恢复先重定位底块、再抓放顶块；只有两块的目标位置和接触关系都稳定后，才释放仍有接触的夹爪并重验。最终六个专家动作规划均成功，原任务与严格接触/位置/双夹爪释放门均为成功，`session.complete=true`、`valid_hil=true`，但仍 `diagnostic_only=true`、`accepted_for_training=false`。原始帧仅留 Ubuntu，JSON已同步到 `outputs/robotwin_dagger_screen_20261024/diagnostics/three_added_tasks_dagger_smoke_20261025/stack_blocks_two/`。
- `pick_dual_bottles` 先前的真实策略集成诊断仍为一正一反：seed40006在step100专家后缀救回并通过严格物理门，seed40005桌面重抓/撤退规划失败而安全拒绝保存。三个新增采集器均只跑冻结seed的单条自动 `i` headless smoke；这验证任务状态分类、专家后缀、规划错误停止和成功门控，不代表真人按键验证或正式数据收集，也不应把这三条诊断样本加入训练。
- 三项相关恢复、CLI和适配器本地测试20项全通过；Ubuntu隔离副本恢复/CLI测试18项全通过；本地 `py_compile` 与 `git diff --check` 通过。修复后GPU0策略服务器已按自身PID身份停止，端口18481释放，显存回到约725 MiB；没有触碰其他服务或任务。建议当前筛选顺序为 stack_blocks_two（64%，一次失败态救回）、put_bottles_dustbin（76%，一次失败态救回）、pick_dual_bottles（32%，有成功与失败救回分支）。要比较收满30条有效HIL的效率，下一步必须用新的非 smoke seeds 做小批配对采集，并分别统计触发接管、专家救回成功、最终任务成功与实际保存有效轨迹；目前样本量不足以推断总体救回率。

## 2026-10-26 同观测 D(o,a) 专家支持筛选

### seed0 主批次：样本和方法

- 更正：seed0 主批次**已经包含训练 D 所需的配对动作 chunk**，不是缺失数据。每个 pair 行保存同一时刻的 `state`、三路图像、`expert_chunks`、`policy_chunks` 和 `stage_id`；策略动作由该行保存的 RGB/state/prompt 在同一观测上查询。每项任务 30 条专家成功 episode、30 个唯一场景、K=4 个策略候选；总量约 1.2k–11.8k 个 anchor/任务。表格里的 bootstrap 以 episode 为独立单位（每项 10,000 次重采样），不是把相关 anchor 当独立样本。
- D 为冻结 R3M 图像特征（1536维）+状态/action chunk 的 3×256 MLP，5 折按 episode 划分（18 train / 6 calibration / 6 test，轮换后每个场景恰好一次 OOF 测试），每折 10,000 步 balanced source BCE。标签只表示“专家动作 chunk / 策略动作 chunk”，不表示成功/失败。专家排序率越高，表示在这些专家访问过的同一观测上 D 越能把专家动作排在策略候选之上。
- 下表按 OOF 专家排序率降序排列。(D_B) 是同观测四个策略候选之间归一化分数差，ESS/K 反映采样权重集中，`Δjoint` 是加权相对均匀权重的关节动作 RMS 位移（mrad）；它们是不同诊断维度，不合成为一个“总分”。

| 排序 | 任务 | 专家排序率，均值 [95% episode-bootstrap CI] | (D_B) 中位数 | ESS/K | Δjoint (mrad) | anchors | 阶段标签验证覆盖率 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `lift_pot` | 1.0000 [1.0000, 1.0000] | 0.00679 | 0.826 | 0.496 | 1,743 | 100.0% |
| 2 | `place_empty_cup` | 0.9989 [0.9971, 1.0000] | 0.01506 | 0.595 | 0.654 | 2,684 | 99.48% |
| 3 | `beat_block_hammer` | 0.9959 [0.9905, 0.9994] | 0.00880 | 0.766 | 0.469 | 1,754 | 80.50% |
| 4 | `move_playingcard_away` | 0.9931 [0.9887, 0.9968] | 0.00711 | 0.816 | 0.620 | 1,869 | 98.93% |
| 5 | `place_phone_stand` | 0.9930 [0.9900, 0.9955] | 0.00789 | 0.793 | 0.559 | 1,997 | 100.0% |
| 6 | `click_bell` | 0.9822 [0.9734, 0.9899] | 0.01105 | 0.707 | 0.608 | 1,190 | 100.0% |
| 7 | `place_object_stand` | 0.9613 [0.9445, 0.9758] | 0.00995 | 0.729 | 0.687 | 2,170 | 100.0% |
| 8 | `click_alarmclock` | 0.9568 [0.9245, 0.9824] | 0.01296 | 0.665 | 0.927 | 1,342 | 100.0% |
| 9 | `press_stapler` | 0.9028 [0.8677, 0.9353] | 0.00730 | 0.812 | 0.654 | 1,896 | 98.84% |
| 10 | `open_microwave` | 0.8786 [0.8609, 0.8955] | 0.05628 | 0.580 | 0.824 | 11,791 | 100.0% |

- 阶段分组数依任务为 2–5。`beat_block_hammer` 的三个 stage replay 未通过（scene seeds 100006, 100008, 100034），因此阶段指标只覆盖其余27个 episode/每个对应阶段、342个 anchor 未验证；这些未知标签不计入阶段摘要，但仍保留在全局 D 指标。另有 `move_playingcard_away` 20、`press_stapler` 22、`place_empty_cup` 14 个阶段 anchor 未验证。阶段标注不足不能解释成 D 失败。
- 分阶段 OOF 专家排序率均值（stage key 是任务专家脚本 `move` 调用点源码行号；`n` 是有验证标签的 episode 数）：

  | 任务 | stage ID: OOF expert rank（n） |
  |---|---|
  | `beat_block_hammer` | `63: 1.000 (27)`; `65: 1.000 (27)`; `68: 1.000 (27)` |
  | `click_alarmclock` | `49: 0.975 (30)`; `63: 0.933 (30)`; `68: 0.901 (30)` |
  | `click_bell` | `46: 0.990 (30)`; `55: 0.980 (30)`; `61: 0.907 (30)` |
  | `lift_pot` | `32: 1.000 (30)`; `37: 1.000 (30)`; `42: 1.000 (30)` |
  | `move_playingcard_away` | `50: 0.998 (30)`; `52: 1.000 (30)`; `54: 0.964 (30)` |
  | `open_microwave` | `35: 0.999 (30)`; `40: 0.863 (30)` |
  | `place_empty_cup` | `62: 1.000 (30)`; `64: 1.000 (30)`; `72: 1.000 (30)`; `77: 0.996 (30)`; `85: 1.000 (30)` |
  | `place_object_stand` | `109: 0.982 (30)`; `111: 0.972 (30)`; `117: 0.928 (30)` |
  | `place_phone_stand` | `76: 0.997 (30)`; `82: 0.988 (30)` |
  | `press_stapler` | `35: 0.961 (30)`; `36: 0.794 (30)`; `39: 0.862 (30)` |

  这些是专家轨迹内部的子计划分组，不是策略失败检测器，也不是“在哪个失败阶段接管”的验证；只能表示 D 在不同专家子计划状态上的动作来源区分程度。
- 初步筛选含义：`lift_pot`、`place_empty_cup`、`beat_block_hammer` 的专家排序率最强；`open_microwave` 的 (D_B) 远高于其他任务，但专家排序率最低，提示“候选间分数有变化”不等同“能可靠区分专家动作”。ESS/K 均高于目标0.5，需连同 β≤100 上限、候选动作差异和实际加权位移解释，不能简单把 ESS 越高或 (D_B) 越大判成越好。
- **边界：**这批数据全来自专家成功轨迹访问的状态，不含策略失败访问状态、HIL 接管结果或成功救回标签。因此 D 能作为专家动作支持/可区分性的任务筛选信号，不能单独预测策略成功率、失败阶段或专家能否救回。后续 R/接管策略仍须用失败态与救回实验单独验证。
- 聚合及每任务 OOF 指标已同步到 `outputs/d_screening_seed0_20261026/`；其中 `aggregate_d_task_screening.json` 与 10 个任务的 `metrics.json` 均已逐文件比对 SHA256。

### 新独立配对批次与 JG 执行安排

- 为验证数据/排序对另一批专家状态是否稳定，另启动 seed1 新批次，不覆盖 seed0：Ubuntu 输出 `outputs/d_action_pairs_screen_20261026_seed1_n30/`，每项任务从 seed1 发布专家场景前40个中收集30条成功 episode；采集器记录专家未来 H10 与策略在精确相同 RGB/state/prompt 下查询的 H10。然后运行阶段标注、K=4策略候选补采和冻结 R3M 编码，再以同一五折/指标协议训练任务专用 D。
- 本记录更新时 `beat_block_hammer` 已完成 30 episode、30 个唯一场景、1,770 个 pair；抽查数组为 `state (N,14)`、图像 `(N,240,320,3)`、专家/策略 chunk 均 `(N,10,14)`，元数据确认策略查询使用所存的精确 RGB/state。`click_alarmclock` 正在采集，其余任务排队。采集器初始 PID 764879，状态位于 Ubuntu `control/d_pair_seed1_regeneration_20261026/state.json`；后处理 watcher 正在等采集完成。
- `JG-ts` 已成功连通：RTX 5090 D v2，检查时占用41 MiB显存、约136 GB可用磁盘；Ubuntu 到 JG 的 SSH agent 转发/直连通过只读测试。seed1 D 训练代码已单独部署到 `/home/ruihao/d_pair_screening_seed1_20261026/code/` 并通过导入/语法检查；另在 JG 用旧的 `place_object_stand` seed0 对 `--task-data-dir` 做了 **2步/折** 隔离 smoke，产生30条完整 OOF 行且输出 `complete=true`，只验证 JG 数据/代码路径，不计 D 指标、不进入 seed1 或 seed0 正式排名。新数据完成并校验后才传入独立目录开展正式五折训练。

## 用户最终选择的三任务失败场景批量救回验证（2026-10-25）

用户本轮选定 handover_block、move_can_pot、stack_blocks_two；已将旧测评的25/22/18个失败场景冻结为65条诊断清单。A800 GPU4/18474与GPU7/18475分别运行前两项，Ubuntu GPU0/18482运行叠两块。每场景先无接管复跑并保存完整动作，再比较实际失败终态救回与固定提前点（150/50/100步）的动作前缀重放救回。旧结果未存动作，不能称严格复刻旧轨迹；失败未复现必须单列。提前分支要求逐chunk状态对齐，所有成功还需物理/关节及稳定后检查。全部diagnostic_only=true、accepted_for_training=false，不生成正式HIL数据。

最新状态：用户收窄到单阶段问题后，A800 的 handover_block/move_can_pot 固定步数队列已停止，不能称65场景全部完成。2026-10-26复核 Ubuntu 的 stack_blocks_two 最终报告为18/18唯一场景完成、18条动作前缀匹配；13条自主失败复现，终态救回1/13，固定第100步提前救回9/13，另5条自主成功不计救回分母。此结果只支持部分中间态可恢复，不证明在线安全窗口检测或接近100%救回。完整协议、单阶段诊断、路径与分母规则见 [三任务失败重放诊断](../hg_dagger/THREE_TASK_FAILURE_RECOVERY_REPLAY_20261025.md)。本轮新增6项标准库逻辑测试、既有15项任务恢复单测和11项handover/move函数测试通过，py_compile及diff检查通过。

### 2026-10-26 15:24 CST：独立 seed1 数据实时核验

Ubuntu 原采集队列仍正常运行：前7项各30条成功专家episode已完成（beat_block_hammer、click_alarmclock、click_bell、move_playingcard_away、press_stapler、place_empty_cup、place_object_stand），第8项place_phone_stand为16/30；open_microwave与lift_pot尚在队列。后处理等待全部采集完成，未把原始pair宣称为已完成K4候选/编码数据。检查时发现后处理控制目录缺少其命令指定的collector副本，已从当前正式采集器复制并核对SHA256一致，未改动或重启正在运行的采集。

JG-ts此时可连接，5090 D v2占41MiB、GPU利用率0%，/home约135GB可用。JG训练代码及隔离2步smoke已准备，部分原始数据已由Ubuntu直接同步到staging_data；尚无seed1正式D训练进程。seed0十任务正式D结果已经完成，两者不能混淆。

### 2026-10-26：HIL 事件及成功／失败轨迹评分验证

先用已有真实标签的 `handover_to_tray` 旧 R3M 判别器做回顾性检查。数据是50条已保存HIL轨迹（首次人控保存帧为事件参考）和100条历史自主rollout（31成功、69失败）；评分对象是**策略建议动作**。复用旧冻结 checkpoint、R3M特征、K4候选与同一轨迹身份，文件SHA须与此前大指标结果一致。脚本 `RoboTwin/scripts/coverage_benchmark/validate_hil_and_outcomes_20261026.py`，机器结果 `outputs/hil_event_success_failure_validation_20261026/metrics.json`。

| 检查 | 结果 | 解释 |
|---|---:|---|
| HIL接管前5帧相对更早5帧的原始 D 分数变化 | −3.892 [−4.630, −3.126]，50轨迹bootstrap95% CI | 接管前确有下降，但这只是对齐后的时间趋势 |
| 无接管历史轨迹，在保存帧索引 `min(21, 轨迹长度)` 的同长度伪窗口变化 | −3.963 [−4.504, −3.399]，100轨迹 | 自主轨迹的普通进展也出现类似下降 |
| 以上下降量识别HIL事件／无事件的AUC | **0.481 [0.385, 0.578]** | 本构造下未优于随机，不能宣称可有效识别接管事件 |
| 100历史轨迹成功／失败：最初10帧均分 AUC | **0.439 [0.310, 0.567]** | 早期分数没有可靠分离 |
| 同上：最初15帧均分 AUC | **0.488 [0.366, 0.609]** | 早期分数没有可靠分离 |
| 同上：全轨均分 AUC | **0.741 [0.634, 0.842]** | 只能作描述：成功轨迹平均22.4个保存帧、失败轨迹均为60帧，单用“轨迹较短”在这批历史数据即可完美区分终局；旧仅观测对照全轨AUC为0.973 |

这里的无接管伪窗口按历史轨迹可用长度截取，不能严格匹配HIL真实接管时的任务阶段；HIL队列又只包含已保存接管轨迹，不能据此估算实际部署的事件检出率和误报率。结果提示“单纯见到分数下降”不足以判定即将接管。上述数字只适用于旧 `handover_to_tray` 判别器；新增十任务的D尚无同任务HIL事件轨迹，不能把这些AUC搬到新任务。新任务后续须保存策略访问状态、动作、阶段、专家接管请求/接受/拒绝与终局结果，在独立seed上按阶段匹配比较。

新增十任务 D 的后续验证口径固定如下，不能用来源分类的专家排序率代替：

1. 用未参与 D 训练的 scene seed 重新执行公开策略，逐次查询保存原始观测、策略 H10 候选、任务阶段、终局成功/失败；一整条 episode 是统计单位。分别报告前 10 次、前 15 次查询的预先固定前缀均分 AUC 与 episode bootstrap 95% CI。完整轨迹均分仅作回顾性描述，必须同时列出长度、阶段分布和仅观测基线，避免把“失败跑得更久”误称为预测能力。
2. 真实 HIL 事件要求同任务的策略 rollout 逐帧记录接管请求时间、请求前策略动作、是否实际接管、人工取消/拒绝及原任务结果。事件正例是**请求之前**固定窗口，负例从同阶段、相近进度且当时无人请求的策略窗口抽取；按 episode 分组拆分。报告 PR-AUC、固定误报率下的召回率、提前量、置信区间和覆盖的失败类型。只保存成功接管的 HIL episode 无法形成有效负例，也不能估计部署误报率。
3. 不在训练/调参所用 episode 上选择检测阈值；先在校准集固定阈值，再在独立测试集报告。若置信区间跨随机基线，或观测/进度基线已达到相同效果，就只记为“未证明额外识别能力”。任何诊断性 rollout 与 HIL smoke 均标 `diagnostic_only=true`、`accepted_for_training=false`。

Ubuntu 已另排队 `control/run_seed2_outcome_traces_20261026.sh`（当前重排 PID 828787；最初 PID 818603 已在第一轮后处理失败时按脚本退出；状态 `outputs/d_outcome_traces_seed2_20261026/control/state.json`），等待 seed1 配对后处理结束和 GPU0 空闲后运行。它保持公开策略原固定 H50 / 发布缓存场景评测及原任务成功判据，用**独立 seed2** 各50条；`trace_policy_queries.py` 在每个真实策略查询处无损保存三相机 RGB、14维状态与策略预测前 H10，`trace_python_shim.py` 只将原评测 worker 的客户端入口导向记录版，server 启停/健康检查不变。脚本已通过语法及单查询 RGB/动作形状 smoke；正式 rollout 尚未开始。后续 `score_outcome_traces.py` 读取已完成轨迹、seed1 五折 D 和同 SHA 冻结 R3M，输出各任务前5/10/15次查询及全轨 AUC、episode bootstrap CI，同时报告各前缀的可用成功/失败样本数及长度本身的 AUC。该结局诊断仍不能替代真实 HIL 请求／未请求事件数据。

2026-10-26 约17:00 CST 进展：seed1 十任务原始配对各30条已完成。`open_microwave` 为得到30条成功专家episode尝试68个发布缓存场景，38个专家场景不合格；`lift_pot` 为30/31。前者的专家规划拒绝没有算策略失败。后处理第一轮在刚关闭的采集端口18489遭遇 `EADDRINUSE`，原始数据无损；重启版移至18491并在候选器端口预检加入 `SO_REUSEADDR` 与有界等待，现正常逐任务运行。独立结局队列因第一轮后处理失败曾标blocked，已核对无旧进程并重排，现等待后处理。JG 上第一项 `beat_block_hammer` 完整特征已由 Ubuntu 直接传输并经 rsync checksum 比对一致；同配置 seed1 五折各1万步训练完成，30个OOF episode的专家排序率 **0.9938 [0.9882, 0.9983]**，与 seed0 的 **0.9959 [0.9905, 0.9994]** 接近。seed1 的 (D_B) 均值0.0161、ESS/K中位数按episode均值0.720、加权相对均匀动作的关节 RMS 位移0.521 mrad；这些仍是专家状态上的动作来源及采样效果指标，不能回答成功/失败或HIL事件检出率。小体积结果与SHA清单在 `outputs/d_screening_seed1_20261026/beat_block_hammer/`，其他九项尚在处理。

随后 `click_alarmclock` 也完成独立seed1五折训练并经SHA核验：专家排序率 **0.9502 [0.9146, 0.9793]**，seed0为 **0.9568**；(D_B) 均值0.0177，ESS/K为0.756，加权相对均匀动作的关节RMS位移0.672 mrad。小体积指标在 `outputs/d_screening_seed1_20261026/click_alarmclock/`。其余任务仍按 Ubuntu 后处理完成顺序直传 JG；两个已完成任务不得被最终总控重复训练。

`click_bell` 的独立 seed1 五折结果也已完成并经小体积文件 checksum 核验：30 个 OOF episode 的专家排序率 **0.9899 [0.9841, 0.9953]**，(D_B) 均值 0.0175、ESS/K 0.742、加权相对均匀动作的关节 RMS 位移 0.547 mrad；指标和 SHA 清单位于 `outputs/d_screening_seed1_20261026/click_bell/`。这仍只检验专家动作相对策略候选的排序，不能解释为 HIL 事件检出率或成功/失败分离度。`move_playingcard_away` 的同源配对数据已在 Ubuntu 后处理完成、直接传 JG 并经逐文件 checksum 比对，五折训练已启动；最终总控应跳过已完成的前三项。

`move_playingcard_away` 五折也已完成：seed1 的 30 个 OOF episode 专家排序率 **0.9863 [0.9797, 0.9921]**、(D_B) 均值 0.0155、ESS/K 0.799、加权动作关节 RMS 位移 0.635 mrad。指标与 SHA 清单在 `outputs/d_screening_seed1_20261026/move_playingcard_away/`。Ubuntu 已完成前八项后处理，其中 `press_stapler`、`place_empty_cup`、`place_object_stand`、`place_phone_stand` 的数据已直传 JG 并经逐文件 checksum 无差异核对；`press_stapler` 五折训练进行中。独立 seed2 成败轨迹队列仍须等待全部后处理完成，不得把上述专家排序率当作成功/失败判别结果。

### 2026-10-26 晚：十任务 D 复验及独立结局检验（已完成）

Ubuntu 后处理最终完成10任务各30条 seed1 专家轨迹：逐任务阶段标注、同观测4个策略候选、冻结 R3M 特征齐全。原后处理脚本末尾生成清单时把整个任务列表误作一个任务名，得到错误的 `postprocess_manifest.json`（已保留为 `postprocess_manifest.invalid_20261026.json`）；用 `RoboTwin/scripts/coverage_benchmark/build_seed1_postprocess_manifest.py` 重新核验每任务完成状态、30个唯一场景并对每任务181个关键文件计算 SHA256。修正后的清单与数据均由 Ubuntu 直传 JG、逐文件 checksum 无差异，未通过本机中转大数据。JG 五折每折10k训练/总控 `status=complete`，10项各30个独立 episode 的 OOF 指标及清单位于 `outputs/d_screening_seed1_20261026/`，汇总为 `aggregate_d_task_screening.json`。

| 任务 | seed1 专家排序率 | seed0 专家排序率 | seed1 \(D_B\) 均值 | ESS/K | 加权动作相对均匀权重的关节RMS位移 (mrad) |
|---|---:|---:|---:|---:|---:|
| `lift_pot` | 100.0% | 100.0% | 0.0104 | 0.863 | 0.442 |
| `place_empty_cup` | 99.8% | 99.9% | 0.0209 | 0.602 | 0.632 |
| `beat_block_hammer` | 99.4% | 99.6% | 0.0161 | 0.720 | 0.521 |
| `click_bell` | 99.0% | 98.2% | 0.0175 | 0.742 | 0.547 |
| `place_phone_stand` | 98.8% | 99.3% | 0.0169 | 0.742 | 0.614 |
| `move_playingcard_away` | 98.6% | 99.3% | 0.0155 | 0.799 | 0.635 |
| `place_object_stand` | 97.8% | 96.1% | 0.0184 | 0.770 | 0.667 |
| `click_alarmclock` | 95.0% | 95.7% | 0.0177 | 0.756 | 0.672 |
| `press_stapler` | 89.6% | 90.3% | 0.0183 | 0.773 | 0.660 |
| `open_microwave` | 85.9% | 87.9% | 0.0993 | 0.564 | 0.771 |

这是**专家状态上的动作来源排序**，不是危险识别率。详细 episode-bootstrap 95% CI、评分差/方差、逐阶段指标均在汇总 JSON；`open_microwave` 的阶段最低专家排序约0.839、`press_stapler` 约0.828，而其它任务阶段最低值至少约0.899。`open_microwave` 的 \(D_B\) 明显更高并非单独的“更好”证据，其专家排序反而最低；也不能把接近100%的来源分类能力直接解释为有用的接管触发器。

为检验结局关联，Ubuntu 另以发布 π0.5 checkpoint 对独立 seed2 发布场景每任务做50条 fixed-H50 自主 rollout，保存每次查询的精确RGB/状态/前10动作及终局（`diagnostic_only=true`，不进入训练）。10项各有50个唯一 scene_seed、完整 metadata、无基础设施错误；D 五折权重经清单 SHA 核对后，用相同 SHA 的 R3M 对真实策略建议动作评分。机器结果在 `outputs/d_outcome_auc_seed2_20261026/`；AUC 以“分高预测成功”为方向，95% CI 按 episode bootstrap。

| 任务 | seed2自主成功 | 首次查询D分数预测成功的AUC [95% CI] |
|---|---:|---:|
| `lift_pot` | 33/50 | 0.435 [0.264, 0.617] |
| `place_empty_cup` | 23/50 | 0.514 [0.345, 0.680] |
| `beat_block_hammer` | 28/50 | 0.562 [0.396, 0.726] |
| `click_bell` | 18/50 | 0.477 [0.300, 0.648] |
| `place_phone_stand` | 24/50 | 0.399 [0.244, 0.558] |
| `move_playingcard_away` | 37/50 | 0.514 [0.318, 0.709] |
| `place_object_stand` | 35/50 | 0.589 [0.415, 0.754] |
| `click_alarmclock` | 31/50 | 0.655 [0.491, 0.800] |
| `press_stapler` | 26/50 | 0.577 [0.415, 0.736] |
| `open_microwave` | 37/50 | 0.443 [0.266, 0.626] |

10项首次查询 AUC 的区间均包含0.5，不能证明 D 对成功/失败有可靠的早期分离能力。原拟报告的前5/10/15次查询必须保留“有多少 episode 足够长、两种结局各多少”口径：fixed-H50 下很多成功轨迹不到5次查询，部分任务前5次只剩0–4条成功，前10/15常只剩单一结局，AUC不可估计。为避免选择偏差补报前1/2/3次查询；全程平均也不能作为成功预测证据，因为此批数据每项失败轨迹都比成功轨迹长，单用查询次数就能完美区分两类（以“更长预测成功”为方向 AUC=0）。

因此目前适合把 D 当作**候选动作支持度/采样诊断**，不适合声称它已验证为 HIL 事件报警器或成功概率头。旧 `handover_to_tray` 有标签的回顾性 HIL 事件 AUC=0.481；新增10任务没有真实的接管请求正例及同阶段无请求负例，不能制造标签或外推“接管检出率”。后续若要检验触发能力，需冻结 D 与阈值，在独立真人/可审计事件记录中对齐安全窗口，并与同阶段负例比较；这不等于要求现在启动真人采集。

### 2026-10-30：turn_switch 原生 SFT 数据 D 支撑函数（50 train / 20 held-out）

本轮先完成 `turn_switch`，`dump_bin_bigbin` 暂缓。训练 cohort 是该任务自训 state=True、5k、原生 15Hz/H10 SFT 使用的同一批 50 个成功专家 demo（scene seed split=1，883 个锚点）；held-out cohort 是另外收集的 20 个成功专家 demo（split=2，354 个锚点），scene seed 与训练集完全不重叠。固定提示词为 “Turn the switch to its final position.”；每个 anchor 保持 RGB/state/prompt 完全相同，用该任务自己的 5k Pi0.5 checkpoint 查询 K=4 个 H10×14 的绝对关节动作 chunk。三个相机视图使用冻结 R3M-ResNet18 编码成1536维特征，源数据和 R3M/候选文件均经 SHA256 清单核验。

D 复用已有 `ChunkCoverageCritic`（1536维 R3M + 14维归一化 state + H10×14 归一化绝对关节 chunk，三层256 MLP），balanced source-BCE（专家为正类、同观测自训策略候选为负类），AdamW、1e-4、batch 256、每个模型10k步。train50 上做5折 episode-disjoint OOF（每折30训练/10校准/10测试），之后另训一个 full50 模型；最终温度/ESS 校准来自训练 cohort 的 OOF 分数，20条独立测试 demo 未参与拟合或校准。所有原始 demo 均没有经过核实的阶段标签，所以**不报告分阶段结果**，不从终局反推阶段。

| 指标 | train50 OOF | 独立 test20 |
|---|---:|---:|
| 同观测专家动作排序率（pairwise AUC） | **0.984 [0.970, 0.993]** | **0.953 [0.925, 0.975]** |
| (D_B) 均值 | 0.0588 [0.0536, 0.0642] | 0.0683 [0.0602, 0.0764] |
| (D_B) 观测中位数的 episode 均值 | 0.0262 [0.0231, 0.0294] | 0.0324 [0.0269, 0.0382] |
| ESS/K | 0.545 [0.506, 0.589] | 0.434 [0.387, 0.484] |
| 中位最大候选权重 | 0.617 | 0.738 |
| 加权相对均匀权重的关节 RMS 位移 | 0.896 mrad | 0.949 mrad |
| K4 候选完全重复对比例 | 0 | 0 |

因此，对 `turn_switch` 而言，D 在新的专家场景上仍能将同一观测下的专家动作块排在自训策略候选之前，且候选并非重复动作；这是支持度/采样诊断的正面结果。test20 的 ESS/K 低于训练 OOF，说明权重较集中，不能把该项本身解释成“更好”或成功率提升。该 test20 全是专家成功轨迹，所以这些结果**不是成功/失败轨迹 AUC、失败检测率、接管率或接管救回率**；目前也没有可核验的阶段标签。

完整 metrics JSON、训练日志和 final full50 D checkpoint 的权威副本在 JG `/home/ruihao/d_support_sft50_test20_turn_switch_20261030/results/turn_switch/`；本机 `outputs/d_support_sft50_test20_turn_switch_20261030/` 已有结果摘要、数据 checksum manifest 和 fold 进度清单。尝试同步其余文件时 JG SSH 路由中断，残留的 `last.pt.incomplete-transfer` 只是未完成传输文件，不是有效 checkpoint；待 JG 可连后再核对 SHA 并补传。候选生成中第一次仅因当前 OpenPI websocket 版本返回空可选元数据而提前终止，未写任何候选；对元数据校验作兼容修正后，70 episode 候选全部完成并通过动作形状校验。

### 按旧 `handover_to_tray` 整轨统计口径重算（seed2 查询点版）

上表的“首次查询 AUC”回答的是起始状态能否预测结局，**不是**旧 `handover_to_tray` 采用的整轨统计。现在用十任务同一批独立 seed2 自主测评结果重算：每条 episode 的所有**已保存查询点** D 分数先取均值，再让50条 episode 等权，计算成功减失败的原始均分差、在该任务50条episode均分中得到的经验百分位差、成功分高于失败分的 AUC，并对成功/失败episode分层 bootstrap 10,000次。与旧代码 `historical_summary` 的聚合与方向一致，输入并未重新选择轨迹。

| 任务 | 成功/50 | 成功−失败原始D均分差 | 成功−失败百分位差 | 整轨查询点 AUC [95% CI] | 成功/失败平均查询数 |
|---|---:|---:|---:|---:|---:|
| `beat_block_hammer` | 28 | −0.229 | −29.5 pp | 0.205 [0.086, 0.341] | 3.5 / 8.0 |
| `click_alarmclock` | 31 | −0.399 | −33.7 pp | 0.163 [0.058, 0.294] | 2.0 / 8.0 |
| `click_bell` | 18 | −0.080 | −8.3 pp | 0.417 [0.250, 0.582] | 2.4 / 8.0 |
| `lift_pot` | 33 | −0.208 | −30.9 pp | 0.191 [0.050, 0.357] | 2.8 / 8.0 |
| `move_playingcard_away` | 37 | −0.139 | −21.3 pp | 0.287 [0.139, 0.447] | 3.0 / 8.0 |
| `open_microwave` | 37 | −0.020 | −7.6 pp | 0.424 [0.268, 0.586] | 14.7 / 30.0 |
| `place_empty_cup` | 23 | −0.254 | −24.4 pp | 0.256 [0.126, 0.403] | 3.7 / 10.0 |
| `place_object_stand` | 35 | −0.420 | −45.4 pp | 0.046 [0.004, 0.110] | 3.0 / 8.0 |
| `place_phone_stand` | 24 | −0.276 | −32.2 pp | 0.178 [0.074, 0.306] | 3.0 / 8.0 |
| `press_stapler` | 26 | +0.053 | +1.1 pp | 0.511 [0.340, 0.679] | 2.8 / 8.0 |

这揭示了一个重要的**方向反转**：按旧式整轨口径，9/10项任务的成功轨迹平均 D 分数反而更低；不能把此前首次查询 AUC 0.399–0.655 概括为“整轨成功／失败分离度”。但它也**不能证明**“低D导致成功”：失败轨迹在所有任务中都持续到步数上限，成功轨迹较早结束；成功轨迹包含的查询点/阶段明显不同，单用“查询数越多预测成功”的 AUC 在十任务中全为0。D 是专家/策略动作来源支持度分数，不是成功概率；在不同阶段、不同分布的建议动作上，分数方向不必与终局同向。任务间独立训练的原始D差也不可横向排名。

这里与历史实验还存在数据粒度差异：旧 `handover_to_tray` 对**每个保存帧的K4策略建议**评分，新任务只保存固定H50执行时每次查询的**一条真实策略H10建议**，没有逐帧K4缓存或仅观测支持度对照。因此只能称“旧统计口径的查询点版”，不能声称严格复刻旧输入。要判定D是否在相同任务阶段、相同进度下区分风险，还需匹配阶段/时间的正负窗口；要复刻旧K4逐帧实验则须另采逐帧候选。可复算脚本为 `RoboTwin/scripts/coverage_benchmark/report_outcome_trajectory_parity.py`，机器结果为 `outputs/d_outcome_auc_seed2_20261026/historical_style_trajectory_summary.json`。

**计算口径审计（2026-10-27）**：旧 `handover_to_tray` 还分为两种不能互换的检验。原大指标里的“接管前下降”是50条已接管HIL轨迹的逐帧**经验百分位差**（R3M为−41.09 pp），没有未接管对照或检出AUC；后来事件诊断改用相同冻结D及K4候选的**原始logit窗口差**（接管前−3.892；自主历史轨迹在中位接管帧索引附近的伪窗口−3.963），以负下降量为“更像接管”方向得AUC 0.4806。原始差与百分位差不能直接比较，且伪窗口未按任务阶段匹配，所以0.4806只检验这个特定构造，不能作最终事件预警能力结论。另一个“整轨成功/失败”检验仍是原始D分数按帧、按episode等权，旧大指标与后来事件诊断在同一100条历史轨迹上AUC都精确为0.7405329593，原始均分差也只差约3×10⁻⁸；这里没有重算或方向错误。新十任务固定H50、查询点K1、五折校准D，不能与旧单D、逐帧K4的0.741直接横比。
