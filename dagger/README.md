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
    legacy.py               # 旧多任务采集器的 callback 接口
    <task>.py               # 已有其他任务的独立 adapter
  tests/                    # 不依赖 GPU/仿真的回归测试
```

这些文件在目录上平级，执行由 `runner` 串行协调。窗口绘制和按键轮询在主线程。接管在策略动作边界生效，当前 chunk 的剩余动作作废；专家 iterator 每次推进一个恢复阶段，允许在阶段边界按 `r` 交还；交还后重置策略并用当前观测重新查询。

## 当前接入范围

新主入口 `python -m dagger.runner --task handover_to_tray` 已接入 `handover_to_tray`。其任务物体、原始专家演示和成功判据仍在 `robotwin_hil/overlay/envs/handover_to_tray.py`；阶段/恢复方法迁至 adapter 的 `HandoverRecoveryMixin`，环境继承它以保持旧调用兼容。

已有的 `place_can_basket`、`rotate_qrcode`、`pick_dual_bottles`、`put_bottles_dustbin`、`stack_blocks_two` adapter 已按任务拆到本目录，旧 `hg_dagger_task_adapters.py` 只做兼容导出。这五项仍使用各自旧采集入口和原来的 `snapshot/classify/recovery/quality` 协议，尚未在新 runner 注册。`handover_block`、`move_can_pot` 的旧专用采集器也仍保留。`stack_blocks_three` 尚未有新 runner adapter；不能仅因 SFT 测评成功就把它标成可接管。

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

参数继承原协议；这不是新官方 H50 checkpoint 的默认运行配置。`frequency` 是接口字段，不能用它推断 horizon 或采样物理频率。模型、任务版本、动作尺度和 norm_stats 应按所用 checkpoint 核对；示例 seed 也需与本次 SFT/D/冻结评测实际清单查重。`--instruction` 可显式冻结整场指令，缺省沿用本任务 adapter 的原 prompt。

现有 `robotwin_hil/local_serving/run_hg_dagger_manual.sh` 已改用新入口；旧 `scripts/hg_dagger_handover.py` 继续作为兼容入口。恢复/配对模式的原参数仍可用：`--resume`、`--one-stage`、`--paired-baseline-report` 和已有增强采样选项。新 adapter 必须显式声明支持单阶段前缀；不能把 handover 的末段配对协议套到其他任务。

## 人工介入与数据

- `i`：停止消费剩余策略动作，显示 adapter 推断的阶段；`Y/Enter` 确认、阶段编号改选、`q` 取消本次接管。构造恢复 iterator 时不执行动作。
- `r`：在专家阶段边界交还策略；单阶段对照模式仍由专家接管至终局，不使用 `r`。
- `x`：作废本局；`q`：结束采集会话。
- 终局记录原任务成功条件，再由操作者决定是否保存 raw。`control_mask` 标记 `policy/hil`，`segments` 的 `start_step/end_step` 沿用历史**保存帧索引**，不是策略动作序号。

本次重构保留原保存语义：普通模式可保留操作者认可的诊断/失败原始轨迹，严格 `valid_hil` 判据仍用于单阶段模式。**保存 raw 不等于验收为训练数据**。后续按 [HG-DAgger 数据整理与训练](../docs/hg_dagger/HG_DAGGER_TRAINING.md) 审计来源和结果，只导出有效专家控制段；不能把整条混合轨迹当专家，也不能跨控制权边界构造动作标签。已有 `prepare_hg_dagger_dataset.py`、`export_hg_dagger_dataset.py` 的数据格式与调用方式保留。

## 接入下一个任务

在 `adapters/<task>.py` 实现 `TaskAdapter`：任务名、默认配置、冻结 prompt、阶段表；读取当前物理状态；构造从当前状态执行的专家后缀；给出任务质量诊断。主流程使用原环境 `check_success()`。恢复时不能重置场景、将物体移回初始位姿，或把特权阶段状态加入策略观测。

先验证 adapter 的阶段与恢复动作，再加入 `adapters/__init__.py` 的注册表。至少覆盖：接管取消、不安全状态、阶段改选、规划失败、交还、同局再次接管、策略/HIL 数据边界。每个任务需单独做真实桌面窗口和仿真接管验收；CPU 测试只证明调度/记录等代码行为。

## 本次验证

CPU 回归覆盖旧/新入口兼容、阶段确认/取消、动作 chunk 丢弃、交还后重新查询、同局再次接管、raw 记录与拒绝覆盖、x/q、旧恢复边界、运行副本打包及源码指纹、策略 RPC 的主线程轮询与排空、离线数据控制边界。

```bash
python -m pytest dagger/tests robotwin_hil/tests/test_prepare_runtime.py
```

真实 Ubuntu viewer、物理恢复与策略 checkpoint 联调仍需在选定任务上执行；本次重构没有启动新的 GPU 采集或训练。
