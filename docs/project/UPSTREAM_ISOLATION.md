# RoboTwin 上游隔离与 HIL 扩展迁移

## 版本边界

- 官方 RoboTwin 现固定为 `4c5c753d66c54c3f6ce855ddf426bcaf3e327250`（PR #510 合并提交）；原固定版本为 `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`。两版之间仅 `handover_block` 和 `rotate_qrcode` 的指令 JSON 发生变化。它以 `RoboTwin/` Git submodule 形式存在，不能承载本项目的修改。
- 该版本内部的官方 XPolicyLab Git submodule 固定为 `fa431ecd893ee706883e64fe5fe1464ec8cd928d`。
- 迁移前的本地 `RoboTwin/` 相对此官方提交有 246 个新增文件、9 个修改文件，以及一份展开并含 Pi0.5 定制代码的 XPolicyLab。新增与修改的 RoboTwin 相对路径已迁至 `robotwin_hil/overlay/`；历史 XPolicyLab 版本保留在 `robotwin_hil/compat/XPolicyLab/`。

## 为什么目前仍有 overlay

既有 HIL/DAgger、D 和策略脚本不是独立插件：部分文件修改了 RoboTwin 基类、测评入口和 XPolicyLab websocket 客户端；大量脚本用 `Path(__file__).parents[...]` 或 `envs.*` 的方式假定自己运行在 RoboTwin 源码树内。直接把它们移动到一个普通 Python 包会改变导入路径、资源定位与运行语义。为保持既有实验可复现，迁移采用显式、可审计的**兼容组装**：

1. 从固定的官方 `RoboTwin/` 拷贝源码到 Git 忽略的 `outputs/robotwin_hil_runtime/RoboTwin/`。
2. 将 `robotwin_hil/compat/XPolicyLab/` 放入该运行目录，维持旧策略服务与 OpenPI 适配的实际代码版本。
3. 按相同相对路径复制 `robotwin_hil/overlay/` 文件；这些文件仅覆盖**生成目录**，绝不修改官方 submodule。
4. 在运行目录写入上游提交、扩展 SHA256 和组装时间，重建前校验指纹；过期目录仅在显式 `--rebuild` 时备份并替换。

普通 RoboTwin 运行、官方对照测评应直接使用 `RoboTwin/`；本项目历史 HIL 与修改后的仿真使用 `$ROBOTWIN_ROOT` 指向的组装目录。两者绝不能不加标记地混合统计。`dump_bin_bigbin` 等对官方文件的覆盖也在 overlay 中，属于自定义仿真版本，不能称为原版场景。

## 使用与验证

```bash
git submodule update --init --recursive
git -C RoboTwin rev-parse HEAD
git -C RoboTwin status --short
python3 -m robotwin_hil.prepare_runtime
source robotwin_hil/activate.sh
test -f "$ROBOTWIN_ROOT/scripts/hg_dagger_handover.py"
test -f "$ROBOTWIN_ROOT/envs/handover_to_tray.py"
```

第一个 Git 命令应输出上面的固定提交，第二个应为空。若有新的扩展源码提交，重新运行构建器会提示显式 `--rebuild`，并保留旧运行目录备份。**不要**把 checkpoint、`norm_stats`、数据集或实验输出复制进 Git 跟踪的 `RoboTwin/` 或 `robotwin_hil/compat/`。现有机器上的绝对路径脚本可能仍指向历史目录；迁移到新版仓库后须先核对其输入与输出路径，再运行或做实验对比。

## 后续真正解耦的方向

兼容组装解决的是“上游不被修改、历史实验不失效”，并未声称所有源码已插件化。后续可逐项将 `overlay/envs` 改为注册式自定义任务，将基类改动改为 wrapper/hook，将 `overlay/scripts` 的路径假设改为显式配置，并将 XPolicyLab 兼容快照缩减为最小 adapter。每次此类替换都须用固定 seed 的动作和结果回归测试，不能仅凭 import 成功判定等价。
