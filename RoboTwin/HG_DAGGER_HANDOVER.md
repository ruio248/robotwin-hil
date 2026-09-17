# handover_to_tray 人工监管 HG-DAgger

## 控制契约

- 模型控制时，监督者在 5090 的 SAPIEN 窗口中观察 rollout。
- 按 `i` 后立即丢弃尚未执行的模型 action chunk，本 episode 永久切换为脚本专家。
- 按 `q` 停止当前 collection session。
- 专家不会 reset 场景；它从当前物体、夹爪接触和机器人 qpos 推断恢复分支并重新规划。
- 只有专家恢复成功、规划成功且所有规划关节在 `[-pi, pi]` 内时，恢复段才写入训练数据。
- 模型自主动作绝不作为专家标签写入。
- HG-DAgger runner 只把 SAPIEN 材质/纹理描述符的预留容量恢复到 SAPIEN
  官方默认 `128/512`（已确认本任务场景能完整创建），不改变 shader、相机图像或物理；
  这样可避免沿用 RoboTwin 面向大型资产任务的 `50000/50000` 冗余预留。
- 启动脚本默认要求至少 `7000 MiB` 空闲显存；不足时会立即退出并提示等待，
  可通过 `HG_DAGGER_MIN_FREE_MIB` 覆盖该门槛。

固定 prompt：

```text
Pass the red bar from the left arm to the right arm and place it in the blue tray.
```

## 1. 人工验收

确认 A800 v2 checkpoint 服务和 5090 到 A800 的 `18303` 隧道正在运行，然后在 5090 执行。先在 clean 仓库根目录执行 `source ./activate_robotwin_hil.sh`。启动脚本默认使用当前未被正式评测占用的 `18303`；可通过 `HG_DAGGER_PORT` 覆盖：

```bash
cd "$ROBOTWIN_ROOT"
bash scripts/run_hg_dagger_acceptance.sh
```

SAPIEN 窗口打开以后，先让模型执行若干步，在仍可恢复时按 `i`。脚本只在以下条件全部满足时以状态码 0 退出并显示 `PASS`：

1. 确实检测到人工 `i`；
2. 已停止模型 action chunk 并进入专家接口；
3. 专家从当前状态规划成功；
4. 最终 `check_success()` 为真；
5. 专家规划关节没有超过 `+/-pi`。

机器可读报告：

```text
$ROBOTWIN_HIL_ROOT/outputs/hg_dagger_acceptance/acceptance_report.json
```

验收模式不写训练数据。

## 2. 第一轮数据收集

默认运行 20 个 rollout，从 collection seed 40000 开始：

```bash
cd "$ROBOTWIN_ROOT"
bash scripts/run_hg_dagger_collection.sh 20 40000
```

可选第三个参数指定输出目录：

```bash
bash scripts/run_hg_dagger_collection.sh \
  20 \
  40000 \
  "$ROBOTWIN_ROOT/data/handover_to_tray_hg_dagger_r1/handover_to_tray/aloha_agilex"
```

不要使用冻结的 31000--31103 test 范围收集训练数据。

每次成功 intervention 会生成：

```text
data/episode_XXXXXXX.hdf5
instruction/episode_XXXXXXX.json
scene_info.json
interventions.jsonl
```

失败的专家接管不会写入 HDF5，只记录在：

```text
rejected_interventions.jsonl
```

## 3. 数据验收

```bash
cd "$ROBOTWIN_ROOT"
bash scripts/validate_hg_dagger_collection.sh
```

如果预期应有 20 个成功恢复 episode：

```bash
bash scripts/validate_hg_dagger_collection.sh \
  "$ROBOTWIN_ROOT/data/handover_to_tray_hg_dagger_r1/handover_to_tray/aloha_agilex" \
  20
```

验收脚本检查 prompt、14 维双臂 joint state/action、`action[t] == state[t+1]`、三路相机长度、stage 范围、seed 一致性、专家成功状态、关节范围和 HG-DAgger 接受标志。

## 4. 数据转换与训练

在收集数据验收通过以后，再把 recovery-only native HDF5 转成单独的 LeRobot 数据集。不要直接覆盖 v2 原始数据集。训练时由配置按比例混合：

```text
原始 v2 专家数据：50%--70%
HG-DAgger recovery 数据：30%--50%
```

继续使用 v2 prompt、joint action schema 和 v2 norm stats。第一轮建议从 checkpoint 9999 继续训练 2500--5000 step，再在 development gate 上关闭专家做纯自主评测。冻结 test seeds 只用于最终报告。
