# DAgger 主流程与任务 adapter

`dagger/` 是仓库顶层的交互式采集模块。主流程沿用 `handover_to_tray`：连接已启动的策略服务、创建环境与窗口、策略执行、人工接管/交还、保存逐帧控制来源。策略服务继续由现有部署脚本启动；采集器不负责启动或终止其他策略服务。

```text
dagger/
  runner.py                 # 唯一的新主入口、episode 循环、策略调用、会话计数
  takeover.py               # i/r/x/q 输入和阶段确认
  recording.py              # raw 帧、control_mask、segments、接管事件与会话 JSON
  viewer.py                 # 窗口刷新、帧率与性能诊断
  runtime.py                # 定位生成的 HIL 运行目录和仿真资产工作目录
  adapters/
    base.py                 # 新主流程的 TaskAdapter 接口
    handover_to_tray.py      # 六阶段判定、从当前状态恢复、专家前缀
    stack_blocks_three.py   # 红绿蓝九阶段、实时抓持/支撑验证、保留已完成前缀
    live_object.py          # 单物体恢复的接触、事件、稳定性和验收公共逻辑
    rotate_qrcode.py        # 四阶段二维码抓取、旋转、放回
    place_empty_cup.py      # 四阶段空杯抓取、杯垫放置
    handover_block.py       # 六阶段左→右交接红块、蓝垫放置
    legacy.py               # 旧多任务采集器的 callback 接口
    <task>.py               # 已有其他任务的独立 adapter
  tests/                    # 不依赖 GPU/仿真的回归测试
```

这些文件在目录上平级，执行由 `runner` 串行协调。窗口绘制和按键轮询在主线程。接管在策略动作边界生效，当前 chunk 的剩余动作作废；专家 iterator 每次推进一个恢复阶段，允许在阶段边界按 `r` 交还；交还后重置策略并用当前观测重新查询。

`desktop.py` 负责启动时选择图形会话。Ubuntu 本机显示器与 Windows App/xrdp 是不同桌面，窗口不会自动镜像。

## 当前接入范围

新主入口 `python -m dagger.runner --task handover_to_tray` 已接入 `handover_to_tray`。其任务物体、原始专家演示和成功判据仍在 `robotwin_hil/overlay/envs/handover_to_tray.py`；阶段/恢复方法迁至 adapter 的 `HandoverRecoveryMixin`，环境继承它以保持旧调用兼容。

`stack_blocks_three` 现也注册在同一 runner；使用独立九阶段恢复 adapter 和 `dagger/run_stack_blocks_three.sh` 启动器，不复制 handover 的主循环。其 H50 checkpoint、冻结指令、接管边界、保守拒绝规则和验证范围见 [三方块 DAgger](../docs/hg_dagger/STACK_BLOCKS_THREE_DAGGER.md)。适配器仿真测试不等于真实策略偏离状态下的救回率，仍需操作者完成现场介入验收。

`rotate_qrcode`、`place_empty_cup`、`handover_block` 也已接入同一 runner，分别使用 `dagger/run_<task>.sh`。三项都使用官方任务的抓放 primitive、H50×14 绝对关节策略、严格训练验收，并默认在 Windows App/xrdp 显示。阶段表、启动示例与验证范围见 [三任务接管流程](../docs/hg_dagger/HG_DAGGER_EXECUTION.md#二维码空杯方块交接三任务dagger2026-11-03)。`rotate_qrcode` 已于 2026-11-04 在真实仿真中通过一局自动接管短诊断，使用官方 seen 逐局指令和初始专家预筛；真人 `i/r` 操作、多个偏离状态及救回率仍待验收。空杯与方块交接目前仍是 CPU 合同测试范围，不能称已验证的真实恢复方案。

已有的 `place_can_basket`、`pick_dual_bottles`、`put_bottles_dustbin`、`stack_blocks_two` 仍使用各自旧采集入口和 `snapshot/classify/recovery/quality` 协议，尚未在新 runner 注册。二维码的旧 `get_adapter()` 工厂和 `handover_block`、`move_can_pot` 的旧专用采集器保留兼容；新三任务启动器不调用这些旧恢复循环。

## 启动方式

在目标 Ubuntu 的既有 RoboTwin 环境中，从仓库根目录组装运行副本：

```bash
python -m robotwin_hil.prepare_runtime
# 已有副本过期时使用 --rebuild，构建器会保留旧副本的备份。
source robotwin_hil/activate.sh
cd "$ROBOTWIN_ROOT"
python -m dagger.runner --help
```

构建器将 `dagger/` 一并复制到运行副本，并纳入源码指纹。可在仓库根目录运行 `python -m dagger.runner --runtime-root /absolute/generated/RoboTwin ...`，入口会在加载仿真代码之前切换到该工作目录。官方 `RoboTwin/` submodule 不参与编辑。

相对的运行目录、输出目录、baseline 报告和采样模型/标定路径，均相对于**启动命令时的目录**解析，不会在切换资产工作目录后重新解释。辅助模块也从所选运行副本加载；若同一 Python 进程已加载另一副本的同名模块，入口会明确拒绝混用，请用新进程启动。

下面是**历史 handover_to_tray 配置**的交互采集示例。先用现有策略部署入口加载与参数一致的权重，确认端口，再在有图形桌面的终端执行：

```bash
python -m dagger.runner \
  --task handover_to_tray \
  --policy-name Pi_05_RobotTwin \
  --host 127.0.0.1 --port 18300 \
  --ckpt-name v2_promptfix_9999 \
  --task-config handover_to_tray_v2_promptfix \
  --frequency 30 --save-freq 15 --render-freq 5 \
  --seed-mode sequential --seed-start 40000 \
  --episodes 5 --target-saved 1 --target-mode hil \
  --output-dir /absolute/new/dagger_collection
```

参数继承原协议；这不是新官方 H50 checkpoint 的默认运行配置。`frequency` 是接口字段，不能用它推断 horizon 或采样物理频率。模型、任务版本、动作尺度和 norm_stats 应按所用 checkpoint 核对；示例 seed 也需与本次 SFT/D/冻结评测实际清单查重。历史 `legacy-fixed` 模式下，`--instruction` 可显式冻结整场指令，缺省沿用本任务 adapter 的原 prompt。

2026-11-04 起，**stack_blocks_three 与 rotate_qrcode 默认 `official-seen`**：完整初始专家预筛通过后重建同 seed 场景，再复用 SFT 的官方 `build_instruction(..., "seen", 50)` 生成本局指令并冻结。正常专家拒绝不消耗 `--episodes`，候选预算默认 `10×episodes`，由 `--max-scene-attempts` 限制；技术异常单独记录并停止。`scene_screening.jsonl` 保存所有候选及原因，续跑避免重用；不能将旧固定指令目录混作新协议继续。新协议不能同时传 `--instruction`。tray、空杯和方块交接的默认协议未迁移。详见 [stack 采集协议](../docs/hg_dagger/STACK_BLOCKS_THREE_DAGGER.md#初始专家预筛与逐局-seen-指令2026-11-04)与[二维码短诊断](../docs/hg_dagger/HG_DAGGER_EXECUTION.md#二维码真实仿真短诊断2026-11-04)。

现有 `robotwin_hil/local_serving/run_hg_dagger_manual.sh` 已改用新入口；旧 `scripts/hg_dagger_handover.py` 继续作为兼容入口。恢复/配对模式的原参数仍可用：`--resume`、`--one-stage`、`--paired-baseline-report` 和已有增强采样选项。新 adapter 必须显式声明支持单阶段前缀；不能把 handover 的末段配对协议套到其他任务。

普通启动器默认目标是 20 条确认保存的有效 HIL（`MANUAL_TARGET_SAVED=20`）；每局结束后显示进度条和按局计算的接管请求率/实际触发率，最终计数写入 `session_*.json`。续采已完成的会话用原 `MANUAL_OUTPUT_DIR` 加 `HIL_RESUME=1`，保持任务、策略、seed 起点和协议一致；可增大 `MANUAL_MAX_ROLLOUTS` 与目标条数。必须先结束旧进程，不能在同目录同时运行两个 collector，也不能把未确认保存的缓存计入 20 条。具体命令见 [执行说明](../docs/hg_dagger/HG_DAGGER_EXECUTION.md#现场操作与记录)。

### Ubuntu 本机显示器与远程桌面

从 SSH 启动、让窗口出现在 Ubuntu 本机显示器上时，使用 `HIL_DESKTOP=local`。启动器读取本机 Xorg/Xwayland 的显示号与授权文件，并验证连接；不需要硬编码 `:1` 或会随登录变化的授权文件名。

```bash
cd /hdd/robotwin-hil-github
HIL_DESKTOP=local HIL_DISPLAY_CHECK_ONLY=1 \
  bash robotwin_hil/local_serving/run_hg_dagger_manual.sh

HIL_DESKTOP=local \
  MANUAL_OUTPUT_DIR="$PWD/outputs/hg_dagger_local_$(date +%Y%m%d_%H%M%S)" \
  MANUAL_SEED_MODE=sequential MANUAL_SEED_START=800000 \
  MANUAL_MAX_ROLLOUTS=5 MANUAL_TARGET_SAVED=1 \
  bash robotwin_hil/local_serving/run_hg_dagger_manual.sh
```

需要在 Windows App/xrdp 中操作则选择 `HIL_DESKTOP=remote`；`current` 只使用当前终端的 `DISPLAY/XAUTHORITY`；默认 `auto` 优先沿用可访问的当前显示，否则查找本机显示。指定 `local` 找不到可用本机会话时会报错，不会悄悄转到远程桌面。策略服务独立运行，重开 collector 不需要重复加载服务。

2026-11-03 在此 Ubuntu 主机复现了 Pygame 2.1.3 / SDL 2.0.22 继承 `XMODIFIERS=@im=fcitx` 等桌面输入法设置时，`pygame.display.init()` 发生原生段错误。预览窗口仅接受控制快捷键，现为该进程设置 `SDL_IM_MODULE=none`、`XMODIFIERS=@im=none`，通过本机 Xwayland 显示；不会修改桌面或终端的中文输入法配置。

等待 `y/n` 保存确认、阶段编号输入和普通策略 RPC 时，也必须在主线程处理窗口事件。预览的 `pump_events()` 不推进仿真、不查询策略、不改记录帧。收到接管请求时仍等待已有 RPC 排空并丢弃返回 chunk；禁止并发发出下一次策略请求。选择保存前按 `q` 或关闭窗口会退出并保留未验收的 raw 缓存，不会自动标成成功或保存为合格 HIL。场景加载或同步专家规划仍可能占用主线程，不能把本次修复解释成所有阶段都保证无阻塞。

## 人工介入与数据

- `i`：停止消费剩余策略动作，显示 adapter 推断的阶段；`Y/Enter` 确认、阶段编号改选、`q` 取消本次接管。构造恢复 iterator 时不执行动作。
- `r`：在专家阶段边界交还策略；单阶段对照模式仍由专家接管至终局，不使用 `r`。
- `x`：作废本局；`q`：结束采集会话。
- 终局记录原任务成功条件，再由操作者决定是否保存 raw。`control_mask` 标记 `policy/hil`，`segments` 的 `start_step/end_step` 沿用历史**保存帧索引**，不是策略动作序号。

handover 保留原保存语义：普通模式可保留操作者认可的诊断/失败原始轨迹，严格 `valid_hil` 判据用于单阶段模式。三方块普通模式另外启用严格训练门槛：人工介入、有 HIL 帧、官方成功、物理支撑/稳定性检查、关节合法和专家阶段验证全部通过，且不是自动触发/headless/偏置诊断，才可标记 `accepted_for_training=true`；失败 raw 可以保留但不会计入目标有效 HIL 数。**保存 raw 不等于验收为训练数据**。后续按 [HG-DAgger 数据整理与训练](../docs/hg_dagger/HG_DAGGER_TRAINING.md) 审计来源和结果，只导出有效专家控制段；不能把整条混合轨迹当专家，也不能跨控制权边界构造动作标签。已有 `prepare_hg_dagger_dataset.py`、`export_hg_dagger_dataset.py` 的数据格式与调用方式保留。

## 接入下一个任务

在 `adapters/<task>.py` 实现 `TaskAdapter`：任务名、默认配置、冻结 prompt、阶段表；读取当前物理状态；构造从当前状态执行的专家后缀；给出任务质量诊断。主流程使用原环境 `check_success()`。恢复时不能重置场景、将物体移回初始位姿，或把特权阶段状态加入策略观测。

先验证 adapter 的阶段与恢复动作，再加入 `adapters/__init__.py` 的注册表。至少覆盖：接管取消、不安全状态、阶段改选、规划失败、交还、同局再次接管、策略/HIL 数据边界。每个任务需单独做真实桌面窗口和仿真接管验收；CPU 测试只证明调度/记录等代码行为。

## 本次验证

CPU 回归覆盖旧/新入口兼容、阶段确认/取消、动作 chunk 丢弃、交还后重新查询、同局再次接管、raw 记录与拒绝覆盖、x/q、旧恢复边界、运行副本打包及源码指纹、策略 RPC 的主线程轮询与排空、离线数据控制边界。

```bash
python -m pytest dagger/tests robotwin_hil/tests/test_prepare_runtime.py
```

真实 Ubuntu viewer、物理恢复与策略 checkpoint 联调仍需在选定任务上执行；本次重构没有启动新的 GPU 采集或训练。

2026-11-03 桌面修复验收：本机 CPU 回归共 28 项通过；Ubuntu 启动器选择到本机 Xwayland `:1`，修复后的真实 SDL 窗口在继承 fcitx 设置的情况下初始化成功，并在保存提示等待 6 秒后接收到合成测试按键。该窗口测试未执行物理步、未保存 episode，不是完整人工接管验收。正在运行的旧 collector 须由操作者先处理保存决定并退出，再重建运行副本；源码同步不会热更新旧进程。
