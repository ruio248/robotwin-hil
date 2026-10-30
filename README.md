# robotwin-hil

RoboTwin 上的 HIL / HG-DAgger、增强采样、专家支撑函数 D 与策略评测研究代码。
本仓库只跟踪源码、配置和小体积文档；checkpoint、演示数据、视频、日志及评测输出保留在各机器的数据盘，不进入 Git。

## 项目结构与上游版本

| 路径 | 用途 |
| --- | --- |
| [`RoboTwin/`](RoboTwin/) | **原样保留**的官方 RoboTwin Git submodule，固定在 [`6dde57155eaf`](https://github.com/RoboTwin-Platform/RoboTwin/commit/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755)；其中官方 `XPolicyLab` submodule 固定在 `fa431ecd893ee706883e64fe5fe1464ec8cd928d`。不要直接在此目录修改实验代码。 |
| [`robotwin_hil/`](robotwin_hil/) | 本项目的自定义任务、HIL/DAgger、采样、D、策略适配和评测代码。`overlay/` 按 RoboTwin 的相对路径存放历史扩展，`compat/XPolicyLab/` 保留当时使用的策略适配源码，`local_serving/` 和 `eval/` 为额外入口。 |
| [`experiments/`](experiments/) | 每次实验专属配置、运行和统计脚本；实验协议不得与其他实验静默混用。 |
| [`docs/`](docs/) | 方法、测评协议、实验结果和迁移说明。 |
| [`README.md`](README.md) | 版本、结构、入口与基本复现步骤。 |

上游源码与自定义实验的边界是刻意的：**普通 RoboTwin 使用 `RoboTwin/`；HIL 实验使用生成的兼容运行目录**。历史脚本依赖 `envs.*`、`scripts/*`、`XPolicyLab/*` 等原路径，因此 `robotwin_hil.prepare_runtime` 会将固定上游版本与本项目的 overlay 组装到被 Git 忽略的 `outputs/robotwin_hil_runtime/RoboTwin/`。它不会改写 `RoboTwin/`，也不会把生成目录提交到 Git。这是为维持既有实验行为的兼容层，不应误称为已完全解耦的插件接口。迁移细节见 [`docs/project/UPSTREAM_ISOLATION.md`](docs/project/UPSTREAM_ISOLATION.md)。

## 获取源码与构建 HIL 运行目录

```bash
git clone --recurse-submodules https://github.com/ruio248/robotwin-hil.git
cd robotwin-hil
git submodule update --init --recursive
python3 -m robotwin_hil.prepare_runtime
source robotwin_hil/activate.sh
cd "$ROBOTWIN_ROOT"
```

`ROBOTWIN_SOURCE_ROOT` 指向官方 submodule，`ROBOTWIN_ROOT` 指向生成的 HIL 运行目录。构建器校验官方提交与扩展源码指纹；运行目录过期时需显式传 `--rebuild`，旧目录会保存在带时间戳的备份路径，不会自动删除。Python/Conda、Sapien、CUDA、OpenPI 及机器人数据依旧需要按目标机器环境单独安装；这个命令只组装源码，不下载模型或数据。

Ubuntu 的 `/hdd` 环境可在仓库根目录运行 `bash robotwin_hil/enter_ubuntu.sh`。该脚本沿用原来的用户命名空间挂载与缓存配置，并在进入前检查/构建兼容运行目录。已有硬编码旧路径的机器专属脚本，迁移后需明确指向 `$ROBOTWIN_ROOT`；不要把官方 `RoboTwin/` 当作 HIL 运行目录。

## 主要扩展入口

在 `cd "$ROBOTWIN_ROOT"` 后，历史入口仍按原相对路径使用：

```bash
python scripts/hg_dagger_handover.py --help
python scripts/eval_policy_xpolicylab.py --help
python -m unittest discover -s scripts/tests -p 'test_*.py'
```

`robotwin_hil/overlay/envs/handover_to_tray.py` 与对应 `env_cfg/`、指令文件定义自定义任务；`overlay/scripts/` 包含 HIL/DAgger、增强采样、D 训练和评测入口；`compat/XPolicyLab/policy/Pi_05_RobotTwin/` 是 Pi0.5 适配器。具体实验参数、checkpoint 和 seed 必须使用相应实验文档，不要从上述通用示例推断。参见 [`docs/tasks/HANDOVER_TO_TRAY_FULL_PIPELINE.md`](docs/tasks/HANDOVER_TO_TRAY_FULL_PIPELINE.md)、[`docs/hg_dagger/ONE_STAGE_DAGGER_PROTOCOL_20261013.md`](docs/hg_dagger/ONE_STAGE_DAGGER_PROTOCOL_20261013.md) 与 [`docs/tasks/ROBOTWIN_DAGGER_TASK_SCREENING_20261024.md`](docs/tasks/ROBOTWIN_DAGGER_TASK_SCREENING_20261024.md)。

文档已按 SFT、专家支撑函数 D、HG-DAgger、采样方法、任务、历史 Coverage 实验和项目协议分类；入口见 [`docs/README.md`](docs/README.md)。其中 11 篇流程说明只给出操作与指标口径，带日期的历史记录另行归档，不把某次结果写成通用结论。

## 数据与复现边界

本仓库公开，但不包含权重、归一化统计、演示数据和评测轨迹。复现时需要提供各实验记录指定的 checkpoint、`norm_stats`、数据集、场景/seed、动作 chunk、频率和仿真版本。`docs/` 中的不同成功率不一定使用相同筛选场景或仿真修复，不能直接合并为统一排行榜。添加新文件前请检查 `.gitignore` 与暂存区，避免提交凭据和机器数据。

上游来源：[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)、[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)、[OpenPI](https://github.com/Physical-Intelligence/openpi)。上游版权与许可文件继续随固定源码及兼容快照保留。
