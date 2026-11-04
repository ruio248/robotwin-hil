# stack_blocks_three：人工门控 DAgger

更新：2026-11-04。用户所称 `stack three block` / `stack_block_three` 对应 RoboTwin 正式任务名 **`stack_blocks_three`**。

## 架构与策略配置

复用 `dagger/runner.py`、`takeover.py`、`recording.py`、`viewer.py` 和桌面选择；只增加 `dagger/adapters/stack_blocks_three.py` 任务 adapter。`dagger/run_stack_blocks_three.sh` 设置任务默认值及 Windows App/xrdp 桌面默认值，再调用共享启动器，不另写一套采集循环，不自动启动或终止策略服务。显式设置 `HIL_DESKTOP=local` 仍可改用 Ubuntu 物理显示器。

- 策略：本任务官方 50 条演示训练的 π0.5 `4999`，`state=True`、完整 H50、14 维绝对 joint 动作。
- Ubuntu checkpoint：`/hdd/official_nine_task_pi05_sft_5000_20261101/checkpoints/stack_blocks_three/4999`，只需 `params/assets`。
- 服务必须使用该 checkpoint 的 `pi05_stack_blocks_three_official50_h50_state_5000_20261101` 和对应 `norm_stats`；不能复用 tray 的 H10 模型。客户端先用既有 XPolicyLab 动作转换器将响应解码为绝对 joint，再拒绝非有限值或非 `[50,14]` chunk；形状检查本身不能证明服务加载了正确权重，仍须核实服务命令和加载日志。
- `Pi_05`、`demo_clean`，默认端口 **18510**，`frequency=50` 为接口字段，不代表 50 Hz，也不是把 H50 改成另一种 horizon。`save_freq=15` 沿用现有原始帧记录规则，不能把它直接解释为 15 Hz。
- 默认 `--episode-protocol official-seen`：每局先进行初始专家预筛，通过后调用 SFT 测评同一个 `build_instruction()`，从官方 `seen` 描述选取本局指令；不再整场固定同一句。红色在底、绿色居中、蓝色在顶，涉及左右臂的模板使用该局专家返回的信息展开。指令在本局所有策略查询及接管/交还期间保持不变。
- 暂不启用 `--one-stage` 或增强采样；CLI 会拒绝，防止把 tray 的阶段 6/H10 评分器误套到三方块任务。

这是沿用官方任务及抓取/放置 primitive 的 **HIL 恢复流程**，不是未修改的官方评测脚本。没有改官方 `RoboTwin/envs/stack_blocks_three.py` 或其 `check_success()`，不能据此声称 HIL 运行副本与历史 SFT 评测的 `_base_task.py` 逐字节相同。

## 初始专家预筛与逐局 seen 指令（2026-11-04）

当前 stack 默认通过共享 `dagger/episode_protocol.py` 执行以下顺序；其他 adapter 及历史 tray 的默认协议本次不变：

1. 从 DAgger 专用 seed 池选候选，先执行现有保留 seed 检查。本任务禁止 `0–99` 与 `100000–100056`，分别覆盖已知 SFT/D test20/专家筛查及冻结评测候选；这不是对未知历史数据集的全局查重证明。
2. 设置 `is_test=True, eval_mode=True, need_plan=True`，关闭窗口、训练帧与视频记录；调用原任务的完整 `play_once()`，不调用 DAgger 恢复 adapter。只有 `plan_success=True` **且** `check_success()=True` 才通过。正常专家失败记录为 `expert_rejected`，换下一个候选，不增加策略 rollout 数、不标成策略失败。
3. 关闭预筛场景，用同一 seed **重新初始化初始场景**，再调用最小记录版导出的 `build_instruction(args, expert_episode_info, "seen", 50)`。不能从专家已经堆好的终局继续策略，不能把预筛轨迹写成 HIL 训练数据。
4. 默认 `--instruction-pool-size 50` 对应正式 SFT 测评的 `test_num=50` 描述生成参数；它不是 DAgger 局数。即使这里只试采 5 局，也仍按 50 条描述候选的方式生成。随机选择发生在场景重建之后、adapter/viewer 初始化及策略 reset 之前。
5. 本局 prompt 保存到 `scene_screening.jsonl`、episode 元数据和 session 的逐局 `records[].instruction`，之后设置策略、进行 `prepare_case/reset/update_obs/get_action`。整场 session 的 `instruction=null` 表示逐局生成，不表示策略收到空指令。

异常验收比官方跳过候选更保守：预筛抛出技术异常、通过后重建场景失败，或官方生成函数退化为仅返回任务名时，写明阶段/seed/traceback 后停止本次采集，退出码 3；不继续隐藏错误、不算策略失败，也不自动重启。普通 `plan_success/check_success` 为 false 则继续筛选。初始预筛通过只说明专家从初始状态能完成，**不保证策略偏离之后的接管可恢复**。

候选日志先写 `expert_started`，再写 `expert_passed/expert_rejected/expert_error`；重建并生成有效指令后才记为 `policy_ready`。所以可区分“预筛通过但尚未进入策略”的情况。预筛无训练帧，`policy_steps=0`，不是额外训练演示。

`--episodes` 是最大策略 rollout 数，不是候选数；默认候选预算为其 10 倍，可用 `--max-scene-attempts` 或 wrapper 的 `MANUAL_MAX_SCENE_ATTEMPTS` 调整。连续模式从 `seed-start` 搜索；随机模式在 `seed-min..seed-max` 无重复抽取并排除保留区。预算/池耗尽写 session 摘要并退出码 4，不循环使用旧 seed。续跑会读取候选日志，连已拒绝/中断的 seed 也不重用；旧固定 prompt 的目录不能以新协议续跑，需全新输出目录。

跨不同新建目录仍不自动全局去重：反复用连续模式 `seed-start=40000` 会重复初始场景，正式采集应规划不重叠区间或审计合并。现有 SFT train50=`0–49`、D test20=`50–69`、正式 SFT 候选=`100000–100056` 与示例 `40000+` 分离。

`--instruction` 与新协议互斥，避免把固定文本伪称为官方 seen 随机选择。显式 `--episode-protocol legacy-fixed` 仅用于旧流程诊断；stack 在该模式不验收为合格训练 HIL。专家预筛/prompt 对齐不代表 HIL `_base_task.py`、Python RNG、GUI 和正式测评已完全等价；本次不修改这些底层差异。

### 本次部署与短诊断验收

2026-11-04 00:50 CST，操作者确认退出旧 runner 后，已核实 PID121083 消失，再同步这次改动并重建 Ubuntu 运行副本；策略服务 PID109311 / 18510 未重启，没有启动长期人工采集。

- 本地相关 CPU 回归 **148 passed**，覆盖预筛拒绝不消耗 rollout、异常停止、同 seed 重建顺序、官方指令生成函数、逐局 prompt、候选预算及续跑去重；Python 编译、shell 语法与 `git diff --check` 通过。这是列出的回归集合，不是整个仓库全部测试。
- 第一次短诊断 `smoke.uUWwdC` 在仿真加载前发现新 runtime 缺少 `assets/objects` 和 `assets/embodiments` 链接，未执行专家或策略动作；失败日志保留。按旧 runtime 的已验证目标恢复这两条链接后，在新目录重试。资产未修改、未复制大文件。后续重建 runtime 也应检查这些未纳入 Git 的机器资产链接，不能只凭源码指纹相同认为资产完整。
- 真实短诊断 `smoke.1QV8fT`：**seed40001 初始专家 `plan_success=true`、`check_success=true`，耗时 16.071 秒**；随后同 seed 重建，生成官方 seen 文本 `Stack green block on red block, then stack blue block on green block at the center.`，与该局 `expert_episode_info` 一起保存。
- 重建后的场景执行 **1 次完整 H50 查询、3 个策略动作**，正常退出。诊断设置 `step_limit=3/save_data=false/auto_save=false`，保存数为 0，`diagnostic_only=true/accepted_for_training=false`；短程截止的 failure 不能计入策略成功率。目录只有候选日志、session JSON 与锁文件，没有专家预筛训练帧或 raw 轨迹。
- Windows App/xrdp 选择为 `remote DISPLAY=:10 XAUTHORITY=/home/ruio/.Xauthority`，preview 创建成功；仍有 libGL/nouveau 告警，但本次未阻断运行。此项不是现场真人按键验收。
- 小体积结果位于本机及 Ubuntu 同名 `outputs/dagger_stack_protocol_validation_20261104/smoke.1QV8fT/session/`：`scene_screening.jsonl` 和 `session_20261104_005001.json`。原始运行日志只留 Ubuntu。
- 源码/运行副本 runner SHA256 一致：`8e08e37e294c2d1158a7859bf0c300739466d93c7d83f03ce66941cdc4fcfbdd`；`episode_protocol.py` 一致：`d63ad835836d15d4f0aa8733568a0bf881582ec71e8ec3e66632ff26f85c1c18`。本次前后 HIL `_base_task.py` SHA 均为 `3b4416f7e61833fe93442793bd9816aa9b6912ede30ccea16deffbb381f08f44`，底层仿真未改。
- 本次前源码备份在 `outputs/dagger_stack_adapter_validation_20261103/screen_before.qb0m5X`，运行副本备份在 `outputs/robotwin_hil_runtime/RoboTwin.backup-20261103T164712Z`；旧数据保留，未 commit/push。

尤其注意：seed40001 的初始专家成功并不否定此前该 seed 在策略偏离状态上的 `no_feasible_grasp_pose`。初始预筛与中途恢复面对不同状态；本次验证不能证明蓝块阶段救回成功，仍需真人小批量接管验收。

## 九阶段与恢复分支

| 阶段 | 操作 | 当前状态约束 |
| --- | --- | --- |
| 1 / 2 / 3 | 红块抓取 / 抬起 / 放置、松爪、撤离 | 红块尚未构成正确底座；已抓持时跳过重抓 |
| 4 / 5 / 6 | 绿块抓取 / 抬起 / 放到红块上 | 红块已受到桌面支撑；目标取实时红块上表面 |
| 7 / 8 / 9 | 蓝块抓取 / 抬起 / 放到绿块上 | 红绿两层已正确支撑；目标取实时绿块上表面 |

每次 `i` 都重新读取当前真实位置、物体实体接触、夹爪状态和支撑关系，不能靠旧阶段计数继续：

- 只完成红块：保留红块，从绿色阶段开始。
- 红绿已堆好：保留两层，从蓝色阶段开始；即使底座不在脚本中心，只要满足官方相对位置和实际支撑，也不搬动已完成的两层。
- 正确的下一块已被某臂抓住：沿用实际抓持臂，从抬起/放置继续，不按初始 x 位置改臂或重抓。
- 块已落在正确支撑上但还没松爪：先松爪并撤离，不再移动底座。
- 任务已经完成：不执行额外动作。

三个 Actor 在官方环境中都叫 `box`，所以物体接触按实体 ID/引用区分，不能按名称猜测。特权状态只供专家判定/诊断，不添加到策略的 RGB/state/prompt 输入。

保守拒绝并给出原因：掉出恢复工作区、仍在明显运动、夹爪部分闭合且抓持不明确、两块同时被抓、错误颜色堆叠、目标块仍托着尚未完成的塔、后续块已抓住但其支撑尚未就绪、无支撑物体、规划/关节检查失败、抓取或抬起后丢块。本版**不自动拆错塔、不捡回落桌外的块**；操作者可作废本局，不会偷偷重置场景或移动物体以制造成功。

## 接管、交还与验收

操作与 tray 一致：`i` 请求接管 → 终端显示阶段 → `Y/Enter` 确认或选合法阶段；取消不执行动作。策略在动作边界停止，丢弃当前 chunk 剩余动作。`r` 在专家子阶段边界交还并重新读取当前观测、查询策略；同局可以再次 `i`。`x` 作废当前局，`q` 结束会话。阶段号不能绕过实时抓持/支撑前置条件。

阶段确认框内的 `q/r` 表示取消本次接管并交还策略（与运行窗口中的 `q=退出会话` 区分）。确认前先检查该阶段的实时前置条件。选中不可执行的阶段时显示具体拒绝原因、保持当前场景并重新等待选择，不进入专家控制、不增加接管次数，也不结束本局；没有交互终端的诊断会取消被拒绝的选择，避免空输入死循环。

部分接触的特例：若且仅若一侧夹爪只接触当前应抓的红/绿/蓝块、该块仍有桌面或已完成堆叠的支撑、其余夹爪没有块接触，且假设解除接触后的实时状态仍可从对应抓取阶段恢复，界面会提示“可确认清除部分接触”。操作者必须**明确输入对应抓取阶段编号**（红 1、绿 4、蓝 7）；`Y/Enter` 不会授权该动作，其他阶段也不能绕过保护。执行时先松爪，再按世界坐标每次向上撤离 1 cm，总计最多 5 cm；每步检查规划/关节合法性、方块位置漂移不超过 5 mm、目标支撑及既有堆叠。接触解除后重新判定阶段，只有仍为原抓取阶段才继续专家重抓。若目标移动、支撑丢失、其他臂接触、规划失败、5 cm 后仍接触或阶段改变，就停在本局等待人工处理；不会无限循环，也不会把不完整的清除动作记为已验证恢复。此特例不覆盖已稳定抓持、接触多个块、无支撑或无法确认状态的情况。

空闲臂接触是另一种情况：选定抓取阶段后，若另一只手臂尚未回待机位且夹爪碰到方块，不能直接调用回位动作。适配器现在先记录空闲臂、所接触方块、夹爪及支撑状态；只有夹爪已张开、只碰一块、所碰方块仍受支撑且当前抓取臂没有碰块时，才按世界坐标每次上撤 1 cm、最多 5 cm。每步核对末端确实上移、全部方块漂移不超过 5 mm、接触对象不增加、已完成堆叠不失支撑。脱离接触后复查原阶段/抓取臂仍适用，再回待机位；回位后仍需复查，才继续抓取。任一检查不通过就停在本局，输出具体原因及接触诊断；已执行但未完成验证的子动作不得当作合格恢复记录。不能把它理解为无条件自动清除所有接触。

操作者明确看到目标块已被夹爪带起、即将放偏时，可选择对应**放置**阶段（红 3、绿 6、蓝 9）。若夹爪半闭合而接触检测只确认该块，旧版 `ambiguous_or_partial_grasp` 会无条件挡住该选择。现在仅在目标块离桌、只有一臂接触该块、此前的方块稳定受支撑时，界面提示“可确认部分抓持放置”；红块只需稳定放在桌上，不要求位于脚本中心，因为官方最终判据是相对堆叠。必须输入对应阶段编号；`Y/Enter` 不会自动放行。此例外使用当前真实物体与夹爪位姿执行官方放置 primitive，但**不能保证部分抓持会保持到放置结束**；放置、松爪、撤离及稳定支撑验证失败就暂停，不继续下一块，也不把动作记为合格专家恢复。诊断会打印两臂夹爪开合、逐块接触和桌面支撑。多块接触、目标仍在桌面或前序堆叠不成立时仍拒绝；没有全面关闭安全检查。

专家已经开始执行后若返回失败，交互采集会暂停，不再把失败 iterator 的 `done` 当作整局终局：`i` 重新选阶段、`r` 交还策略并查询新 chunk、`x` 明确作废、`q` 退出并保留未确认缓存。重选被取消则仍暂停，不自动交还。失败原因、重试和交还记录保留；失败恢复不能因此被当成成功专家标签。`--one-stage`、`--acceptance` 和自动介入诊断仍保留有限退出行为，不等待无人输入。

一个放置阶段包含 `place_and_release` 与 `retreat`。前者仅规划执行成功仍标为 `verified=false`；撤离并通过稳定性检查后才能补为已验证。若在两者之间交还，原始数据可以保留，但这段未完成的专家动作不会让整局自动成为合格 HIL。抓取/抬起的物理后置条件也在 yield/交还边界之前检查。

放置后的稳定性检查通过既有 `take_dense_action` 执行保持当前关节目标的短窗口，仍记录 `hil` 来源和帧，不使用无记录的裸 `scene.step()`。当前窗口为至少 0.24 秒、6 个采样间隔，已完成支撑链需满足位置漂移 ≤2 mm、姿态变化 ≤0.02 rad、接触支撑有效及有限速度门槛。这样区分接触求解器瞬时速度抖动与实际滑动。稳定证据在位置/姿态越界、失去支撑或策略接管后失效；这些是 adapter 的保守质量门槛，不替代官方终局判据，也不是经过统计验证的救回率标准。

原始数据保留 `control_mask`、`segments`、接管/交还事件、官方终局成功/失败、专家各子阶段与稳定窗口、规划状态、关节合法性。`segments` 的索引是保存帧索引，不是 H50 查询编号或物理步数。只在以下条件同时满足时标记 `accepted_for_training=true`：

1. 有人工接管、有非空 HIL 帧，最终官方任务成功且操作者标签为成功；
2. 关节合法、真实支撑/稳定质量门槛通过，专家事件均成功且验证完整；
3. 不是自动介入/自动交还、headless、强制步数或偏置诊断；
4. 操作者确认保存；如使用 `target-mode=expert`，还要求专家 iterator 报告完整成功。

保存失败 raw 只作诊断，标记 `diagnostic_only`，不增加目标有效 HIL 数，现有导出器跳过。后续训练只取通过审计的专家控制段，不能把策略前缀或跨控制权边界的动作窗口当专家标签。详情见 [数据整理与训练](HG_DAGGER_TRAINING.md)。

## 启动示例（Ubuntu）

先部署更新源码、按 [公共入口](../../dagger/README.md#启动方式) 重建 HIL 运行副本，确保 objects/embodiments 资产链接完整。运行副本过期不能靠仅更改源码热更新；也不要在已有 collector 未退出时重建。

策略服务应单独启动。此机器已验收的 stack 服务入口/配置如下（复用原 checkpoint 配置，CLI 只覆盖监听端口）：

```bash
cd /hdd/robotwin-hil-github
# 先确认 GPU0 和 18510 可用；已有正确服务时不要再启动。
ROBOTWIN_HIL_ROOT="$PWD" \
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
JAX_COMPILATION_CACHE_DIR="$PWD/.cache/jax" \
HF_HOME=/media/ruio/hdd/pace_pi05_eval_20261022/.hf-home \
OPENPI_DATA_HOME=/media/ruio/hdd/pace_pi05_eval_20261022/openpi_data \
bash robotwin_hil/enter_ubuntu.sh \
  /hdd/Openpi-PACE-eval-20261022/.venv/bin/python -u \
  /hdd/official_nine_task_pi05_sft_5000_20261101/eval_stack_blocks_three_20261101/stack_server_entry.py \
  --config_path /hdd/official_nine_task_pi05_sft_5000_20261101/eval_stack_blocks_three_20261101/server.yaml \
  --host 127.0.0.1 --port 18510
```

服务日志必须显示本任务 `4999`、`STATE=True H=50 ACTION=joint`，不能只看到端口打开就认为 checkpoint 正确。该服务入口依赖机器上已安装并验收的 OpenPI 环境，不是跨机器免配置安装器。

另开终端，先检查桌面，再启动少量人工试采：

```bash
cd /hdd/robotwin-hil-github
HIL_DESKTOP=remote HIL_DISPLAY_CHECK_ONLY=1 \
  bash dagger/run_stack_blocks_three.sh

HIL_DESKTOP=remote HIL_ES_MODE=off \
MANUAL_POLICY_PORT=18510 \
MANUAL_OUTPUT_DIR="$PWD/outputs/dagger/stack_blocks_three/manual_$(date +%Y%m%d_%H%M%S)" \
MANUAL_SEED_MODE=sequential MANUAL_SEED_START=40000 \
MANUAL_MAX_ROLLOUTS=5 MANUAL_TARGET_SAVED=1 \
MANUAL_MAX_SCENE_ATTEMPTS=50 \
bash dagger/run_stack_blocks_three.sh
```

重启后应先看到 `Episode protocol: official-seen` 和 `[EXPERT-SCREEN] candidate=... seed=...`；通过且重建成功后才显示 `accepted ... prompt=...` 与 `[ROLLOUT ...]`。初始预筛不显示窗口，会增加每个候选的专家执行耗时。正在运行的旧 runner 不会热更新；必须先由操作者处理旧局并退出，再重建 runtime、用全新输出启动。策略服务不需要因此重启。

上述示例在 Windows App/xrdp 中显示，三方块 wrapper 未设置 `HIL_DESKTOP` 时也默认 `remote`。使用 Ubuntu 物理显示器时显式设置 `HIL_DESKTOP=local`；不要手动硬编码 `DISPLAY` 或复制过期的 XAUTHORITY 路径。入口从当前桌面进程自动发现并验证连接；没有可访问的 xrdp 会话时会明确报错，不悄悄切到物理显示器。详见 [桌面选择](HG_DAGGER_EXECUTION.md#桌面选择ubuntu-本机与-windows-app2026-11-03)。默认保留已知 seed 区间 0–99、100000–100056；40000 仅是示例，正式收集前仍需按最新 SFT/D/评测 manifest 查重。硬编码护栏不是完整数据血缘证明。

### 2026-11-03 动作响应格式修复

官方 Pi_05 服务按协议可以返回 **50 个分臂动作字典**，每个字典有左右臂 6 维关节及各 1 维夹爪。直接对这个列表调用 `np.asarray` 得到的是对象数组 `(50,)`，不是已经展开的数值矩阵；初版 adapter 的入口检查因此在第一条动作执行前误拒绝。

修复在共享 runner 的验证入口复用 `xpolicylab_action_to_robotwin`，解码后交给 adapter 检查完整 H50×14。保留原始响应，并沿用逐动作执行时的转换、当前观测回退和接管边界；不额外查询、不重采样、不缩短 horizon、不改变关节顺序。回归覆盖数值矩阵、字典列表及 `actions` 包装响应，也检查 H10 和 chunk 末尾 NaN 在任何动作执行前被拒绝。

本次修复验收（2026-11-03 20:43 CST）：

- 下述 CPU 回归集合增至 **75 passed**，另通过 Python 编译、shell 语法及 diff 空白检查。
- Ubuntu 源码和重建运行副本的 `dagger/runner.py` SHA256 一致：`1562cb92d99f3271a4e8c5cf2d2723d47a54a7429edab978e1a4c0d1d0b09b44`。
- 复用已启动的本任务 4999 策略服务（18510），真实查询 **1 次完整 H50**，在仿真中执行 **3 个策略动作**，正常退出，无动作形状异常；并非把模型改成 H3。
- 窗口选择实际为 `desktop=remote DISPLAY=:10 XAUTHORITY=/home/ruio/.Xauthority`，preview 创建成功。xrdp 日志中有 libGL/nouveau 加载告警，但本次窗口初始化及三步运行没有因此失败；不能把连接检查当作真人可见性/按键验收。
- 诊断 seed=800203、`step_limit=3`、`save_data=false`、`auto_save=false`，`policy_decisions=1`、`policy_steps=3`、`diagnostic_only=true`、`accepted_for_training=false`、保存数为 0。短程截止的 failure 标签不计正式成功率。
- 结果：`outputs/dagger_stack_adapter_validation_20261103/policy_wire_remote.rpsCKu/session/session_20261103_204336.json`（Ubuntu 与本机同名小 JSON）；原始日志留 Ubuntu 的该诊断目录。
- 修复前源码备份 `outputs/dagger_stack_adapter_validation_20261103/action_wire_before.gcyTbf`，旧运行副本备份 `outputs/robotwin_hil_runtime/RoboTwin.backup-20261103T124256Z`；旧失败缓存未删除，现有策略服务未重启。未启动长期采集、未 commit/push。

## 验证范围与复现

2026-11-03 后续发现的接管选择问题：seed40001、策略第93步自动判为 `ambiguous_or_partial_grasp`，操作者选1；初版界面直接构造 iterator，iterator 在第一步拒绝，而 runner 遇到任何 `phase=done` 都结束本局。修复将无动作的前置校验放入确认框，并把交互专家恢复失败改为等待操作者决定。保留原抓持/支撑检查，**不把“手动输入1”当作已验证抓持，也不通过放松阈值假装能恢复**。

本次 CPU 回归新增覆盖：部分抓持时 `1→Y→r` 不运动且返回策略、错误阶段重选、确认后状态改变的二次校验、无终端拒绝不会死循环；以及恢复失败后重试/交还/作废/退出四条 runner 路径。2026-11-03 本地相关完整测试集合 **132 passed**（包括工作树中其他已存在 adapter 的测试），编译及 diff 检查通过。

操作者确认退出后，已核实旧 PID111329 消失，再同步源码并重建 Ubuntu 实际运行副本；没有热替换正在运行的 collector，没有重启策略服务 PID109311/18510。runner SHA256 为 `ff3062bf1160652530395c01ab3f84ae3f36b013df18722bd1db28c8b623f7a2`，源码与运行副本一致；修复前源码备份 `outputs/dagger_stack_adapter_validation_20261103/selection_before.rZ2pij`，运行副本备份 `outputs/robotwin_hil_runtime/RoboTwin.backup-20261103T130008Z`。

实机仿真诊断 `selection_smoke.3CLmNd/summary.json`（位于 `outputs/dagger_stack_adapter_validation_20261103/`）：seed800200 初始场景，程序化输入 `6→1`；阶段6明确拒绝、物体位置完全不变，改选1后实际执行 `red_grasp` 且 `ok=true/verified=true`。不加载策略、不保存训练帧，`diagnostic_only=true/accepted_for_training=false`。这验证了选择界面和真实专家 primitive 的连通，不是 seed40001 第93步部分抓持现场的复现，也不证明该状态能成功救回。Windows App 连接检查仍为 `remote=:10`；未 commit/push。

CPU 回归覆盖：实体接触、阶段判定、保留已完成前缀、抓持中续跑、拒绝错误改选/不安全状态、规划失败/丢块、交还再接管、H50 输入门槛、稳定窗口、未验证放置不得验收、raw/导出/续采计数，以及不破坏原 tray 公共流程。

```bash
python -m pytest dagger/tests robotwin_hil/tests/test_prepare_runtime.py \
  robotwin_hil/overlay/scripts/tests/test_hg_dagger_resume.py \
  robotwin_hil/overlay/scripts/tests/test_export_hg_dagger_dataset.py \
  robotwin_hil/overlay/scripts/tests/test_prepare_hg_dagger_dataset.py \
  robotwin_hil/overlay/scripts/tests/test_responsive_policy_client.py
```

真实仿真诊断脚本 `python -m dagger.tests.smoke_stack_blocks_three --help` 只从新建场景验证 adapter，不加载策略、不产出训练样本。`--interrupt-after red|green|blue_grasp` 在真实达到的物理状态中断 iterator，再从当前状态启动恢复；不 teleport 或重放旧计划。诊断结果留在 Ubuntu `outputs/dagger_stack_adapter_validation_20261103/`，初版失败记录保留，不能只统计修复后通过记录冒充成功率。

2026-11-03 验收记录：

- 上述 CPU 回归集合 **66 passed**；Python 编译检查、shell 语法检查、`git diff --check` 通过。另尝试纳入历史 `test_hg_dagger_collector_cli.py` 时，其旧导入链在收集阶段缺少 `envs.CONFIGS_PATH`；未为此改动旧采集器，66 项通过不代表整库全部测试通过。
- 最终 adapter 的无策略整套恢复：seed **800200、800201、800202** 三项均通过，结果为 `final_initial/summary.json`。
- 同一 seed **800200** 分别在放好红块、放好红绿两层、已抓住蓝块处中断，再新建 iterator：三个分支均通过，结果为 `live_arm_resume_red/summary.json`、`live_arm_resume_green/summary.json`、`live_arm_resume_blue_grasp/summary.json`。这不是另外三条独立随机场景。
- 六项诊断均满足官方成功、物理质量门槛及 `recovery_ok`；全部 `diagnostic_only=true`、`accepted_for_training=false`，未生成训练数据。较早版本的接触识别、瞬时速度误拒绝、交还后空闲臂未归位失败记录原样保留。
- Ubuntu `/hdd/robotwin-hil-github` 源码及生成运行副本已同步；adapter SHA256 均为 `21dc8fd236d06ad333ea1fb6f65d96e7d1186d9d1dd316487581eddacd9fb75f`。重建前运行副本保存在 `outputs/robotwin_hil_runtime/RoboTwin.backup-20261103T122105Z`，未删除旧 raw 或其他结果。
- 初次 adapter 验收时，本机 `local=:1` 与 xrdp `remote=:10` **连接检查**通过；这不是现场真人按键验收。当时尚未启动端口 18510 策略服务，后续真实服务连通与格式修复验收见上文。小体积诊断 JSON 已同步到本机同名 `outputs/dagger_stack_adapter_validation_20261103/`，未传模型或大轨迹，未 commit/push。

**尚不能宣称完成的验证**：真人通过本机/Windows App 窗口操作完整 stack episode、该 π0.5 策略造成的多种偏离状态接管、正式 DAgger 训练增益及统计救回率。开始正式收集前需完成现场小批量验收。本次实现不启动长期采集、不重训策略/D、不改已有 SFT 测评结果。
