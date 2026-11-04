# HG-DAgger：执行与数据采集

任务入口：`handover_to_tray` 沿用本文既有普通/末段协议；新增 `stack_blocks_three` 的普通交互采集见 [三方块任务说明](STACK_BLOCKS_THREE_DAGGER.md)。后者使用同一个顶层 runner 和桌面选择，不继承 handover 的 H10 末段前缀或增强采样设置。

`rotate_qrcode`、`place_empty_cup`、`handover_block` 的普通人工接管见下节，沿用同一 runner；本文后面的 **tray 末段配对/H10 协议不适用于这三项**。`rotate_qrcode` 现默认使用初始专家预筛和逐局官方 `seen` 指令；另外两项仍保持各自现有默认值。

## 二维码、空杯、方块交接三任务 DAgger（2026-11-03）

三项共享 `dagger/runner.py`、`takeover.py`、`recording.py`、`viewer.py`；`dagger/adapters/live_object.py` 只封装物体接触、专家事件和稳定性验证，不另起策略循环。官方任务源文件与 `check_success()` 未修改；专家从策略访问到的当前场景规划，不调用 `play_once`、不 reset、不 teleport。

空杯交互启动器 `dagger/run_place_empty_cup.sh` 默认只渲染**一个正面 spectator 视角**：相机居中在工作台 +y 侧，面向 y=-0.65 的两条 Aloha 机械臂；内部预览 320×240，默认请求窗口 1200×900。这样每次预览只取一张图，降低三路取图和 960×540 合成开销。这个 spectator 视角不是策略的 `head_camera`；右键拖动/滚轮可以调整预览位置，不会改变策略相机位姿、观测、动作、仿真或 SFT 测评。`i/r/x/q` 等接管按键仍有效。三路布局保留为可选模式：显式设置 `HIL_VIEWER_IMAGE_SOURCE=head_wrist HIL_VIEWER_RENDER_RESOLUTION=960x540 HIL_VIEWER_WINDOW_RESOLUTION=1600x900` 时，左侧 2/3 为主摄像头，右侧 1/3 上下为左右腕摄像头；只看主摄像头则设 `HIL_VIEWER_IMAGE_SOURCE=head_camera`。默认值在下次启动 runner 时生效，其他任务默认不变。

Ubuntu/Windows App 预览验收：此前单主摄像头 seed 43024 真实启动日志确认 `source=head_camera rendered=320x240`。三路布局已通过独立 SAPIEN 窗口测试；空杯 seed 43030 的真实短诊断确认 `source=head_wrist`，三路原始帧均为 320×240，内部合成画布 960×540。两局都在首个策略步后由测试钩子自动作废，`save_data=false`，不计采集。后来用户反馈三路实时预览明显偏慢，seed 44000 在旧 runner 中只推进到 195/500 步；经用户确认后该 runner 被中断，只有未确认的原始 `.cache/episode0`，没有正式保存的 episode，缓存已保留。

正面视角修正（2026-11-04）：早先 seed 43031 只验证了 `source=spectator`，未检查相机实际姿态，不能据此声称正面预设生效。根因是 `_base_task` 调用 `setup_scene()` 时没有传入任务参数，相机仍使用默认斜角。现由 `dagger.viewer.render_initial_frame` 在窗口创建后、首帧绘制前直接调用 viewer 的相机设置接口，`robot-front` 实际位置为 `(0, 0.35, 1.25)`、RPY 为 `(0, -0.50, π/2)`；日志会打印 `applied spectator preset=robot-front` 及实际设置值。已在同一空杯场景 seed 43032 保存并检查真实渲染图，确认是正面居中、轻微向下看工作区的单视角。诊断不保存训练帧，不计成功率；完整交互速度仍需人工复核。窗口管理器可能限制外窗大小，不改变内部预览尺寸。

| 正式任务名 | 阶段 | 实时续跑分支 | 独立默认端口 |
| --- | --- | --- | --- |
| `rotate_qrcode` | 1 抓取；2 抬起；3 旋转并放回；4 释放/撤离/验证 | 桌面支持时重抓；已抓持则沿实际抓持臂继续；已正确放好则验证支撑 | 18511 |
| `place_empty_cup` | 1 夹爪预形+抓杯；2 抬杯；3 放上杯垫；4 释放/撤离/验证 | 同上；保留官方左臂 contact point 2、右臂 0、预形夹爪 0.6 | 18512 |
| `handover_block` | 1 左抓；2 左抬；3 交接位姿；4 右接；5 左松开并撤离；6 放上蓝垫/验证 | 左持→继续交接；双持→验证右持后左松；右持→直接放置；已落蓝垫→验证支撑 | 18513 |

2026-11-04 的恢复动作审计：二维码与空杯从阶段 1 接管时不再无条件张开/停放闲置臂，也不把抓取臂送回初始位姿；从当前机械臂位姿调用任务原生 `grasp_actor`。空杯保留官方 `close_gripper(pos=0.6)` 预形，二维码直接抓取；只有抓取臂空爪却关闭时才先张开。阶段 2/3 沿用官方抬起/放置 primitive，阶段 4 是中途恢复额外需要的释放、撤离与稳定性验证，不是官方 HDF5 的原生阶段标签。交接块在左抓和右接之前也不再无条件回位；左松开并抬离后，最终阶段按官方 `play_once()` 在**同一个** `env.move` 中并行执行左臂回位与右臂放蓝垫。右臂接球前或左臂直接从右持状态续跑时，只在空爪确实关闭的情况下张开。这里的回位仅为交接块官方末段动作，绝不重置场景或物体。三项均仍由实时抓持/支撑状态选择可执行后缀，不能把任意阶段号当作无条件跳转。

方块任务名是 `handover_block`，不是 `hand_over_block`；参考任务名是 `handover_to_tray`。方块 prompt 固定使用 PR#510 的第一条 seen 指令：

```text
Pass the red block from the left arm to the right arm and put it on the blue pad.
```

二维码已在 2026-11-04 改为默认 `official-seen`：从 DAgger 独立 seed 池中取候选，完整脚本专家先通过 `plan_success` 和 `check_success`，随后重建同 seed 初始场景，调用 SFT 测评共用的 `build_instruction(..., "seen", 50)` 生成本局 prompt，并在本局冻结。正常专家拒绝写入 `scene_screening.jsonl` 后换 seed，不占用策略 rollout；预筛技术异常单独记录并停止。该模式不能同时传固定 `--instruction`。二维码旧固定句仅在显式 `--episode-protocol legacy-fixed` 的诊断中可用；不能把旧句说成与 SFT 逐局随机文本一致。空杯仍默认 `Place the empty cup on top of the coaster`，方块交接仍使用上面的固定 PR#510 句子；两项不自动继承二维码的新默认。特权物体姿态/阶段/支撑信息仅用于专家判断与诊断，不加入策略观测。

共同策略契约为 `Pi_05`、`demo_clean`、`state=True` 的相应任务 checkpoint、完整绝对 joint **H50×14**。`frequency=50` 是接口字段，`save_freq=15` 是现有保存规则，不能据此推断物理 Hz。共享 runner 使用执行时同一个转换器解码数值数组或官方分臂动作字典，再做 H50/有限值检查；端口与形状检查不证明服务权重正确，启动前仍须核对该任务的服务命令、加载配置、checkpoint 和 norm_stats。新任务拒绝 `--one-stage` 与增强采样，不能误用 tray 的 H10 评分器。

### 人工操作与安全边界

`i` 请求接管，动作边界生效并丢弃剩余 chunk；终端展示实时阶段，`Y/Enter` 确认，阶段号只能选择满足当前物理前提的阶段。三项单物体适配器与三方块任务都会在确认阶段时拒绝“部分抓持”等不可恢复状态或不符的手选编号，并在真正执行前再次检查；手选编号不能强制绕过安全条件。`r` 在专家子阶段边界交还，重置策略并重新取观测；同局再次 `i` 会重算状态。`x` 作废本局，`q` 退出。阶段选择与 iterator 构造均不移动机器人。

抓持要求夹爪闭合且物体与该爪真实接触，不能单凭距离近就认为抓住。物体接触使用实体引用/ID，避免多个 actor 同名误判。单臂任务拒绝双臂同时抓持、部分闭爪不明确、无支撑/明显运动、越界或非有限状态。空杯倾倒时不执行未验证的扶正；方块倒卧或掉在右侧但无人抓持时不强行让左臂跨区重抓。操作者可放弃，不自动复位场景。

左爪松开前必须验证右爪接触并闭合；抓取和抬起后也立即复查抓持，失败事件在交还边界前写入。规划构造异常、规划失败、非法关节、掉块都会使本局不满足训练验收。二维码保留官方朝向判据，并额外要求目标 XY 误差 ≤8 cm、桌面接触；空杯额外要求杯垫接触和杯口朝上；交接块要求蓝垫接触、官方位置条件和双爪释放。这些是 **HIL 数据质量门槛**，不更改官方 SFT 测评成功率定义。

放置事件先记录 `verified=false`，经过释放/撤离及至少 0.24 秒、6 个采样间隔的保持窗口后才确认。保持由正常 `take_dense_action` 执行并记录 HIL 来源，不使用裸 `scene.step()` 偷跑；要求目标支撑、≤2 mm 漂移、≤0.02 rad 旋转、有限速度门槛和官方终局成功。若在放置与验证之间交还，旧的未验证专家事件不会被后来另一轮接管悄悄补成通过。上述阈值是保守实现门槛，尚未通过真实多 seed 仿真标定。

每局保留官方成功/失败、`physical_quality_gate`、`recovery_ok`、`live_recovery_events`、`control_mask`、`segments`、人工接管/交还和保存决定。只有人工介入、有 HIL 帧、官方成功、物理质量及全部专家事件验证通过、关节合法、标签成功且确认保存，且非自动/headless/强制步限诊断，才计入 `accepted_for_training=true`。失败 raw 可诊断保存，但不会计入有效 HIL 数；后续只导出合格专家控制段，不跨控制权边界构造训练标签。

### Ubuntu 启动示例

2026-11-04 已将二维码 adapter 与共享依赖同步到 Ubuntu 并重建运行副本，完成下述短诊断；原始数据未动。后续再次部署时，先结束旧 collector 并处理保存决定，再重建 runtime；资产链接不属于源码复制结果，需核实 `assets/objects` 和 `assets/embodiments`。本次二维码运行副本的 objects 指向与既有官方 39/50 测评相同的 `/hdd/pace_pi05_eval_20261022/assets/objects`，其中包含 `070_paymentsign`；仅有 stack 所需对象的旧目录不够运行此任务。

```bash
cd /hdd/robotwin-hil-github
python -m robotwin_hil.prepare_runtime --rebuild
```

策略服务独立启动，每次只运行一个 collector。使用该任务自己的 4999 params/assets、训练配置及 norm_stats；服务与渲染同 GPU 时沿用已验证的 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.55`。当前 GPU0 的 stack 服务占用约 14 GiB，启动二维码服务前先确认没有 stack runner 在使用它，并完成服务切换；两模型同驻已因显存不足失败。二维码服务在单独终端的已验证启动命令为：

```bash
cd /hdd/robotwin-hil-github
ROBOTWIN_HIL_ROOT="$PWD" CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
JAX_COMPILATION_CACHE_DIR="$PWD/.cache/jax" \
HF_HOME=/media/ruio/hdd/pace_pi05_eval_20261022/.hf-home \
OPENPI_DATA_HOME=/media/ruio/hdd/pace_pi05_eval_20261022/openpi_data \
bash robotwin_hil/enter_ubuntu.sh \
  /hdd/Openpi-PACE-eval-20261022/.venv/bin/python -u \
  /hdd/official_nine_task_pi05_sft_5000_20261101/eval_rotate_qrcode_min_trace_20261103/control/server_setup/stack_server_entry.py \
  --config_path /hdd/official_nine_task_pi05_sft_5000_20261101/eval_rotate_qrcode_min_trace_20261103/control/server_setup/server.yaml \
  --host 127.0.0.1 --port 18511
```

加载日志须显示 `rotate_qrcode`、对应 `4999`、`STATE=True H=50 ACTION=joint`，且 18511 正在监听。以下是**连接已就绪策略服务**的试采示例，不会自动拉模型或启动模型服务：

```bash
cd /hdd/robotwin-hil-github
# 先在 Windows App 中连接并登录同一 Ubuntu 用户。
HIL_DESKTOP=remote HIL_DISPLAY_CHECK_ONLY=1 bash dagger/run_rotate_qrcode.sh

# 任选一个任务；把 task 改为 place_empty_cup / handover_block 即用对应默认端口。
task=rotate_qrcode
HIL_DESKTOP=remote HIL_ES_MODE=off \
MANUAL_SEED_MODE=sequential MANUAL_SEED_START=42000 \
MANUAL_MAX_ROLLOUTS=5 MANUAL_TARGET_SAVED=1 \
MANUAL_OUTPUT_DIR="$PWD/outputs/dagger/$task/manual_$(date +%Y%m%d_%H%M%S)" \
bash "dagger/run_${task}.sh"
```

任务 wrapper 默认 Windows App/xrdp；`HIL_DESKTOP=local` 可改到物理显示器。无需写死 `DISPLAY/XAUTHORITY`。若服务使用别的已核实端口，用 `MANUAL_POLICY_PORT` 覆盖；这些默认端口只是配置，不代表已监听。二维码专用服务需加载 `rotate_qrcode_official50_4999`、`state=True`、H50 与自身的 `norm_stats`。此主机只有一张 24 GiB GPU0：实测 stack 服务与二维码服务同时加载时，二维码以 25% 的 JAX 预留仍因额外 1.2 GiB 分配失败；试采前应确认 GPU 资源并在没有活跃 runner 时切换服务。保留区间 0–999、100000–109999 是防误用护栏；42000 是示例，采集前仍需与实际 SFT/D/冻结测评 seed 清单核对，不能只信区间；不同输出目录之间没有全局去重。

### 二维码真实仿真短诊断（2026-11-04）

Ubuntu 运行副本先前缺少二维码 adapter 注册，已补齐并验证源码与运行副本一致。恢复诊断还发现直接创建场景时可能没有 `episode_metadata`，共享单物体 adapter 初始化现会创建该字段；相关 CPU 测试 **66 passed**。

使用已完成正式 SFT 测评的二维码 4999 checkpoint、GPU0/18511、Windows App 的 `remote DISPLAY=:10` 做了两次互不计训练的诊断：

- `rollout.vurMJo`：seed41000 初始专家预筛通过，按该局专家信息选出 `Take the small paymentsign for mobile payments, lift it, orient it so the QR faces you, then set it down`；预览窗口创建，策略一次完整 H50 查询后执行 3 步。`step_limit=3/save_data=false/auto_save=false`，没有 HIL 接管、没有保存训练数据；不能拿其短程终局算成功率。
- `takeover.dSyUuR`：同 seed 再做独立诊断，策略执行 1 步后以 `--auto-intervene-step 1` 自动请求接管，adapter 判为阶段 1，并实际完成抓取、抬起、旋转放置和撤离验证。官方 `check_success=true`、`physical_quality_gate=true`、`recovery_ok=true`。这局 `diagnostic_only=true/accepted_for_training=false`、保存数为 0；**没有验证真人按 `i/r` 的窗口操作，也不是救回率估计**。
- 较早的无预筛独立恢复诊断 `recovery.8IIEe1` 在 seed800300 报 `left_grasp_not_verified`：抓取后左爪闭合但未与物体接触，按质量门槛失败。此 seed 与已预筛通过的 41000 不同；不应从一个诊断推断所有偏离状态可恢复。

三份小体积诊断 JSON 留在 Ubuntu `/hdd/robotwin-hil-github/outputs/dagger_rotate_validation_20261104/` 并同步到本机同名 `outputs/`，成功接管的 session JSON 在本机和 Ubuntu 的 SHA256 均为 `c2121948b173a59bf3d3a66f26b38dffa825fcf2f5f4c0c5e8437137b2140c6f`。首轮 18511 与 stack 18510 同时加载时出现 `RESOURCE_EXHAUSTED`，这属于服务启动资源问题，不是专家规划失败。诊断后已停止本次二维码服务，恢复原 stack 4999 服务（PID134120、18510 已监听）；因此再次运行上述二维码启动器前，仍需先切换 GPU0 服务。没有下载模型、没有启动长期人工采集；这次只验证了短诊断，不改变已报告的 SFT 39/50 结果。

### 验证范围

CPU 测试覆盖三项整套阶段、左右臂、实时重新接管、交接右爪丢失、阶段误选、规划异常/失败、抓空/掉块、支撑漂移、非法关节、H50 响应、记录/交还/续采计数与 shell 参数。测试会强制使用源码 helper，避免混用开发机上已有的生成 runtime：

```bash
python -m pytest dagger/tests -q
```

2026-11-04 本机回归：`PYTHONPATH=. pytest dagger/tests -q` **146 passed**，覆盖三任务动作序列、物理前提、阶段误选提前拒绝、主相机显示绑定、堆叠任务兼容和 runner 基本契约。先前 2026-11-03 的阶段性结果为 109/124 passed；那是较早版本，不是当前最终计数。

另提供显式真实仿真诊断入口；它不加载策略、不生成训练数据，所有输出标记 `diagnostic_only=true, accepted_for_training=false`，不会由普通启动器自动调用。下面命令尚未在 Ubuntu 执行验收：

```bash
python -m dagger.tests.smoke_live_object_adapters \
  --task place_empty_cup \
  --runtime-root /hdd/robotwin-hil-github/outputs/robotwin_hil_runtime/RoboTwin \
  --output-dir /hdd/robotwin-hil-github/outputs/dagger_cup_smoke_new \
  --seeds 800300 800301
# 新输出目录下可加 --interrupt-after lift 验证真实抬起状态后的续跑；
# handover_block 用 source_lift / receiver_grasp / source_release。
```

2026-11-04 Ubuntu 单局额外诊断：`place_empty_cup` seed 43013，策略执行 48 步后自动触发接管；实时阶段判为 1，专家从当前场景完成抓杯、抬杯、放杯及稳定性验证，官方 `check_success=True`、物理质量门槛通过。该诊断关闭 viewer、`save_data=false`，报告在 `/hdd/robotwin-hil-github/outputs/dagger_place_empty_cup_validation_20261104/stage_aligned_takeover_seed43013/session_20261104_020405.json`；**不是**真人试采，也**不计**有效 HIL 数据。

**尚待验收**：二维码与方块交接的改版真实多 seed 续跑、堆叠任务的多 seed 偏离状态、Windows App 真人 i/r/x/q 完整试采。单局诊断和 CPU 测试都不能作为 DAgger 救回率、HIL 检测率或训练增益。

## 启动前

先确认生成运行目录已更新、图形桌面可见、任务和服务端加载同一策略 checkpoint/配置/`norm_stats`，独立策略服务端口身份明确。`robotwin_hil/local_serving/start_local_policy_server.sh` 是服务入口；采集器只连接既有服务，不应为重开采集而停止其他端口。`HIL_DRY_RUN=1` 可检查 `robotwin_hil/local_serving/run_one_stage_dagger.sh` 组装的命令；无 viewer 的 headless 冒烟不能验证真实按键。

### 桌面选择：Ubuntu 本机与 Windows App（2026-11-03）

Ubuntu 本机显示器与 Windows App/xrdp 是**两个独立图形会话**，不是同一屏幕的镜像。普通交互采集启动器 `robotwin_hil/local_serving/run_hg_dagger_manual.sh` 使用以下环境变量选择窗口位置：

- `HIL_DESKTOP=local`：显示在 Ubuntu 本机连接的物理显示器上；即使命令从 SSH 发出，也选择本机桌面。
- `HIL_DESKTOP=remote`：显示在 Windows App 连接的 xrdp 桌面中。

只切换这一项，其他采集参数保持不变；不需要重配 Python、CUDA、策略 checkpoint 或策略服务。启动器自动探测所选会话的 `DISPLAY` 和 `XAUTHORITY` 并验证连接，**不要手动长期写死 `DISPLAY=:10.0` 或授权文件路径**。本机曾检测到 `:1`、远程检测到 `:10`，但它们不是固定协议值，重新登录后可能变化。`current` 只沿用当前终端可访问的显示配置；默认 `auto` 优先沿用可访问的当前显示，否则查找本机桌面。需要明确窗口位置时应显式使用 `local` 或 `remote`。

通过 Windows App 操作时，先连接 Ubuntu 并登录运行采集器的同一用户（当前机器为 `ruio`），保持该远程桌面会话打开。然后在 SSH 终端或远程桌面终端中检查：

```bash
cd /hdd/robotwin-hil-github

# 只检查 Windows App/xrdp 的桌面连接，不启动仿真或采集。
HIL_DESKTOP=remote HIL_DISPLAY_CHECK_ONLY=1 \
  bash robotwin_hil/local_serving/run_hg_dagger_manual.sh
```

成功时输出类似下面的内容，实际显示号以探测结果为准：

```text
[HIL-VIEWER] desktop=remote DISPLAY=:10 ...
[HIL-VIEWER] desktop connection ready: DISPLAY=:10
```

检查 Ubuntu 本机显示器时，将上述 `remote` 改为 `local`。正式启动时，去掉 `HIL_DISPLAY_CHECK_ONLY=1`，并在**原有完整采集命令**中将 `HIL_DESKTOP=local` 替换为 `HIL_DESKTOP=remote`；保留其 checkpoint、任务、seed、保存和采样设置。完整普通采集示例见 [顶层 DAgger 启动说明](../../dagger/README.md#ubuntu-本机显示器与远程桌面)。不要把去掉检查变量后的检查命令当作已配置好实验参数的完整采集命令。

切换桌面前，先处理当前 episode 的保存决定并退出旧 collector，确认进程已经结束，再启动新 collector。**修改环境变量不会迁移已经打开的窗口，关闭远程连接也不等于采集进程退出。** 策略服务可以继续运行，不需要重复加载；不要同时启动两个采集器连接同一策略服务。新会话使用独立输出目录；需要续采旧目录时按下文的 `--resume` 规则处理，不能直接覆盖或重新使用旧 seed 计作新增数据。

### 窗口无响应与输入法兼容

保存/标签确认、阶段输入和普通策略 RPC 等待期间，主线程继续处理窗口事件；不能因为终端正在等 `y/n` 就停止事件轮询，否则桌面会报告 `runner.py` 无响应。事件处理不推进物理步。该主机的旧 SDL 在继承 fcitx 输入法设置时另有可复现的初始化段错误，预览进程已禁用其自身的输入法路径，桌面/终端中文输入不变。更新代码后需要退出旧 collector 再启动；已经运行的进程不会自动加载修复。

上述事件处理和预览输入法兼容修复对 `local`、`remote` 两种显示选择均生效；选择远程桌面不需要额外切换一套 Python 环境。场景加载或同步专家规划仍可能短暂占用主线程，不能把这些修复理解成所有阶段都保证无阻塞。

### 末段对照启动

末段对照的仓库根目录启动示意（输出目录与报告文件须替换为本次新路径）：

```bash
ONE_STAGE_ARM=baseline MANUAL_TARGET_SAVED=30 \
  MANUAL_OUTPUT_DIR=/data/one_stage_baseline \
  bash robotwin_hil/local_serving/run_one_stage_dagger.sh

ONE_STAGE_ARM=enhanced MANUAL_TARGET_SAVED=30 \
  HIL_ONE_STAGE_PAIRED_REPORT=/data/one_stage_baseline/session_actual.json \
  MANUAL_OUTPUT_DIR=/data/one_stage_enhanced \
  bash robotwin_hil/local_serving/run_one_stage_dagger.sh
```

`ONE_STAGE_ARM=baseline` 是 K=1、增强采样关闭；`enhanced` 为 K=32、D 对完整 H10 chunk 评分和加权，从阶段 6 首次决策自动开启。增强组读取**完成的** baseline session，并跑其合格末段 seed 清单；即使先收满目标有效 HIL，也不得为了整齐数字中途截断配对清单。Gaussian 是另一个独立 arm，需显式设定 σ，不应混入上述两组。

## 数据采集 seed 与配对规则

单阶段脚本强制顺序取 scene seed，默认 `MANUAL_SEED_START=100000`，最多尝试 `MANUAL_MAX_ROLLOUTS=200` 次，目标 `MANUAL_TARGET_SAVED=30` 条**有效 HIL**。baseline 第 `j` 次尝试使用 `100000+j`（若覆盖了起点则以实际起点为准）；前五阶段专家前缀失败、按 `x` 放弃、接管后失败或自主成功都可能消耗一次尝试。因此“收满 30 条”不等于只用了 30 个 seed，也不能从目标数推断最后一个 seed。逐条真实 seed、状态与保存决定以 `episodes.jsonl` 和最终 `session_*.json` 为准；`--resume` 会保留已用 seed 并检查起点/配置，不能把续跑当作新样本从起点再记一次。

增强/Gaussian 配对组不是从自己的 `100000` 重新生成一批成功样本：`HIL_ONE_STAGE_PAIRED_REPORT` 指向 `aborted=false` 的 baseline 最终报告，脚本按原顺序重放其中 `rollout_status=completed` 的末段合格 seed；baseline 的 `prefix_failed` 或 `aborted` 不进入此清单。相同 seed 只保证相同场景随机化条件，不保证策略随机动作和介入时刻后的状态逐帧相同；比较时还须剔除配对组未到合格交接点或按 `x` 取消的记录，并分别报告共同合格分母、接管请求和有效 HIL 保存数。不要把两组各自的“前 30 条有效 HIL”当成天然配对。

采集 seed 应与本任务 SFT 演示、D 训练/测试、开发及冻结测评 seed **按实际清单**查重。`hg_dagger_handover.py` 当前硬编码保留区间 1–579、30000–30022、31000–31103，顺序/随机模式均会拒绝这些 seed；这是防误用护栏，**不是完整数据血缘证明**，例如 seed 0 不在此护栏内。跨任务或换数据版本时，应检查各自的 `seed.txt`/manifest，不能仅信这三个区间。普通非单阶段随机采集还需保存 `seed_mode`、`seed_min/max` 与 `rng_seed`；未显式给 `rng_seed` 时会自动产生并记入 session，以便重现抽取顺序。

## 现场操作与记录

普通交互采集启动器默认目标现在为 **20 条确认保存的有效 HIL**（`MANUAL_TARGET_SAVED=20`）；`MANUAL_MAX_ROLLOUTS` 是最多策略局数，可能需要大于 20，因为自主成功、失败、作废或未通过质量门槛的局不计目标。终端在开始及每局结束后显示 `[PROGRESS]`：20 格进度条、`有效已保存/目标`、`已跑/最多局数`，以及两种按局接管率。`takeover requested` 统计至少按过一次 `i` 的有效 rollout；`triggered` 统计至少真正进入一次专家接管的有效 rollout。两者分母都是有策略步或接管请求的局数，**不是每步概率**；零有效局显示 `n/a`。结束时 `session_*.json` 的 `takeover_measurement` 保存相同两种比例及计数，逐局 `records` 留有接管次数与事件。

续采必须在原 collector 完全退出后，指向**同一个** `MANUAL_OUTPUT_DIR`，加 `HIL_RESUME=1`，并保持任务、checkpoint、协议、`MANUAL_SEED_START` 等既有配置一致。可以提高 `MANUAL_MAX_ROLLOUTS` 和目标数，例如从试采 `5 局/1 条` 扩到 `100 局/20 条`；已有 session 报告必须完整，原始 episode 与索引须一致，且锁不能被另一个进程持有。`--resume` 会继承已用 seed、已保存有效 HIL 和既有记录，保留未确认的缓存但**不会把缓存自动算作已保存**。不要更换输出目录或省略 `HIL_RESUME=1` 却声称是在续采，也不要在正在运行的 collector 上热替换源码。

```bash
cd /hdd/robotwin-hil-github
HIL_DESKTOP=remote HIL_ES_MODE=off HIL_RESUME=1 \
MANUAL_SEED_MODE=sequential MANUAL_SEED_START=47000 \
MANUAL_MAX_ROLLOUTS=100 MANUAL_TARGET_SAVED=20 \
MANUAL_OUTPUT_DIR="$PWD/outputs/dagger/stack_blocks_three/manual_20261104_47000" \
bash dagger/run_stack_blocks_three.sh
```

这只是与上次 `seed=47000` 试采目录配套的示例；运行前需核对实际旧目录、session 报告及种子起点。旧版采集器的窗口不会自动出现新进度条，新代码需在其退出后重建运行副本才会生效。

操作者在 GUI 中按 `i` 请求接管、`x` 放弃当前 episode、`q` 结束；单阶段不按 `r`。`hg_dagger_handover.py` 在每条记录保存 `stage6_handoff`、`manual_takeover_requests`、`expert_result`、`hil_frames`、`joints_legal`、`valid_hil`、`autonomous_success`、`save_decision`，会话报告保存总量与 `aborted`。有效 HIL 需发生介入、有 HIL 帧、专家恢复成功、最终任务成功且规划关节合法；X 与接管后失败不能算有效数据。

原始帧在 `raw/episode_*/frames/*.pkl`，元数据在 `episode.json`，总表在 `episodes.jsonl` / `session_*.json`。`robotwin_hil/overlay/scripts/export_hg_dagger_dataset.py` 可按 `full` 或 `hil` 导出；带 `diagnostic_only` 或 `accepted_for_training=false` 的记录会跳过。继续中断会话只能用相同配置 `--resume` 并检查已有报告，不能在旧目录上直接重启。采集结束先核查 `aborted=false`、计数与原始数据完整，再启动离线整理/训练。
