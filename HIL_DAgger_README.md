# HIL/DAgger 数据收集与训练流程

本文档描述 RoboTwin `handover_to_tray` 长序任务上，从模型部署、HIL 数据
收集与保存、到保存后的训练和最终测试的完整流程。

## 1. 概览

整体闭环是：

```text
部署 Pi0.5 策略服务
      ↓
交互式 HG-DAgger 收集（策略 ↔ 专家手动接管/交还）
      ↓
保存完整轨迹，并给每一帧打 policy/hil 控制来源
      ↓
从保存的数据中抽取 hil 段，转成 LeRobot 数据集
      ↓
与原 v2 专家数据混合后重训 Pi0.5
      ↓
在固定 dev/test seeds 上评估，并按需做扰动救援测试
```

## 2. 环境与模型部署

### 2.1 仓库与环境

- 仓库根目录：`/hdd/robotwin-hil`
- RoboTwin 代码：`/hdd/robotwin-hil/RoboTwin`
- 进入 HDD 环境（会自动建立 `/media/ruio/hdd` 绑定挂载并选择 conda 环境）：

```bash
cd /hdd/robotwin-hil
bash ./enter_robotwin_hil.sh
```

### 2.2 Checkpoint

当前使用 `v2_promptfix_9999` 的 bf16 推理副本，本地路径：

```text
/hdd/robotwin-hil/RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/checkpoints/
  pi05_robotwin_handover_to_tray_v2_promptfix/
    robotwin_handover_to_tray_v2_promptfix_bf16_inference/9999/{params,assets}
```

该路径由配置文件指定：

```text
RoboTwin/XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml
```

### 2.3 启动本地策略服务

```bash
bash /hdd/robotwin-hil/local_serving/start_local_policy_server.sh
```

服务默认在 GPU 0 上监听 `127.0.0.1:18300`，使用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`。
确认服务已就绪：

```bash
ss -ltn | grep ':18300 '
```

## 3. HIL 数据收集

### 3.1 交互式收集命令

在远程桌面的终端里执行：

```bash
cd /hdd/robotwin-hil
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
  DISPLAY=:1 XAUTHORITY=/home/ruio/.Xauthority \
  bash ./enter_robotwin_hil.sh python -u scripts/hg_dagger_handover.py \
    --host 127.0.0.1 --port 18300 \
    --policy-name Pi_05_RobotTwin \
    --ckpt-name v2_promptfix_9999 \
    --task-config handover_to_tray_v2_promptfix \
    --seed-start 40000 \
    --episodes 50 \
    --target-saved 10 \
    --render-freq 5 --frequency 30 \
    --save-data true \
    --output-dir /media/ruio/hdd/robotwin-hil/outputs/hg_dagger_collection
```

> 采集阶段不生成 MP4/HDF5，只落原始帧；视频和 LeRobot 转换在离线阶段用
> `export_hg_dagger_dataset.py` 完成（见 4.1 节）。

`--target-saved`（默认 10）是**有效 HIL 条数**的目标：程序会一直跑 rollout，
直到累计保存了 10 条有效 episode 才停；`--episodes` 是 rollout 次数上限。
session 报告里会记录 `total_rollouts`、每个 rollout 的 `rollout_seconds`、
`total_seconds` 和 `seconds_per_saved_episode`。

两个和"有效"相关的开关：

- `--target-mode hil|expert`：`hil`（默认）= 保存且含 HIL 帧就算一条；`expert`
  = 还要求专家完整恢复 `success=True`。

  HG-DAgger 的典型用法是**短暂接管、修正到安全点后按 `r` 交还策略**，这种
  episode 的 expert 恢复并没有跑到 `done`，但 HIL 段本身就是专家动作，属于有效
  样本，因此默认用 `hil` 口径；`expert` 只在你专门想收"专家完整救回"样本时使用。
- `--seed-mode sequential|random`：`random`（默认）会在
  `[--seed-min, --seed-max]`（默认 40000–99999）里随机抽 seed，并自动跳过
  demo（1–579）、dev（30000–30022）、frozen test（31000–31103）区间，用
  随机采样提高场景多样性；`sequential` 则从 `--seed-start` 顺序取。

### 3.2 操作键

- rollout 中：`i` = 接管，`r` = 交还策略，`q` = 退出。
- 按下 `i` 后，终端会弹出任务共 6 个阶段以及专家**自动判定**的接管分支和
  当前夹爪/杆位状态。监督者先确认判断是否正确：

  - `Y` / 回车 = 接受自动判定；
  - `1`–`6` = 强制从某个阶段进入恢复（例如杆已悬空在托盘上方时选 `4`，
    让右臂原地抓取后直接放托盘，而不是错误地回到左臂重抓）；
  - `q` = 取消本次接管，继续由策略执行。

  六个阶段与恢复入口对应关系：

  | 阶段 | 名称 | 说明 |
  |---|---|---|
  | 1 | source_grasp | 左臂抓取红杆（从头重抓的入口） |
  | 2 | source_lift | 左臂抬起（并入阶段 3 的 `resume_handover`） |
  | 3 | handover_pose | 移向交接位姿（`resume_handover` 入口） |
  | 4 | receiver_grasp | 右臂抓取（含"悬空直接放"入口） |
  | 5 | source_release_and_retreat | 左臂松开并后撤（`receiver_place` 入口） |
  | 6 | guided_tray_placement | 放入蓝托盘（`receiver_place` / `release_at_tray` 入口） |

  每次接管会把 `chosen_stage_id` 写进 episode 的 intervention 记录，便于事后
  检查人工选择的阶段是否覆盖了专家自动误判的场景。
- episode 结束后由程序自动判定成功/失败（基于 `check_success()`），日志会打印
  `[TRAJECTORY] auto label=success|failure`；
- 监督者只需要决定这条轨迹是否有效：`y` = 保存，`n` = 丢弃。

如需人工打标签，加 `--label-mode manual`（此时才出现 `s`/`f` 提示）。

一个 episode 内可以多次 `i`/`r`，实现 HG-DAgger 的专家 gating。

### 3.3 窗口分辨率与速度

SAPIEN viewer 的**固有渲染目标分辨率**（framebuffer）是速度的主要瓶颈，
和窗口拖多大无关。代码现在固定为 `1280x720`，并在创建 viewer 后显式调用
`window.resize()` 强制该尺寸；可用 `HIL_VIEWER_RESOLUTION` 覆盖：

```bash
HIL_VIEWER_RESOLUTION=960x540 bash ./enter_robotwin_hil.sh python ...
```

同一批 test seed、`--save-videos none` 下的实测步速：

| viewer 分辨率 | 步速 |
|---|---|
| 4K（3700x2032） | ~5.5 步/秒 |
| 960x540 | ~8.0 步/秒 |
| 无窗口（`--render-freq 0`） | ~9.3 步/秒 |

默认 `1280x720` 的预期速度介于 960x540 和 4K 之间。需要人看时用默认或
960x540；纯自动化测试直接用 `--render-freq 0`。

## 4. 数据保存（raw-first）

采集阶段只落**原始帧 + 元数据**，不做 HDF5 / MP4 / LeRobot 转换，避免编码拖慢
控制循环、也避免在切分规则还没定稿时提前固化格式。

```text
<output-dir>/
  raw/episode_NNNNNNN/frames/*.pkl   # 逐帧原始观测，按 --save-freq 采样
  raw/episode_NNNNNNN/episode.json   # control_mask / segments / label / seed
  episodes.jsonl
  session_YYYYMMDD_HHMMSS.json
```

`episode.json` 中保存：

- `control_mask`：逐帧 `policy` / `hil` 标签；
- `segments`：`{source, start_step, end_step}` 控制段；
- `supervisor_label`：`success` / `failure`；
- `save_freq`、`seed`、`episode_metadata`、`info`。

监督者按 `n` 丢弃的 episode 会直接删除 cache，不落盘。

### 4.1 离线导出（HDF5 / MP4）

```bash
python -u scripts/export_hg_dagger_dataset.py \
  --raw-root /media/ruio/hdd/robotwin-hil/outputs/hg_dagger_collection/raw \
  --output-dir /media/ruio/hdd/robotwin-hil/outputs/hg_dagger_export \
  --mode full \
  --save-video true
```

- `--mode full`：完整 policy+HIL 轨迹；
- `--mode hil`：只保留 `control_mask == "hil"` 的帧（重新编号），用于训练；
- 输出 `data/*.hdf5`、`video/*.mp4`（可选）、`instruction/*.json`、`manifest_*.json`。

## 5. 保存后的训练

### 5.1 HIL 帧抽取与 LeRobot 转换（待实现）

计划脚本：

```text
RoboTwin/scripts/prepare_hg_dagger_dataset.py
```

它需要：

1. 先用 `export_hg_dagger_dataset.py --mode hil` 导出 HIL-only HDF5。
2. 把 HIL-only HDF5 转成 recovery-only LeRobot 数据集。
3. 与原始 `ruio248/robotwin_handover_to_tray_v2_promptfix` 数据混合。

建议混合比例：原始 v2 专家数据 50–70%，HG-DAgger hil 数据 30–50%。

### 5.2 重训 Pi0.5

使用已有 `pi05_robotwin_handover_to_tray_v2_promptfix` 训练配置，从 `9999`
checkpoint 继续训练：

```bash
cd <training-host-openpi>
.venv/bin/python scripts/train.py \
  pi05_robotwin_handover_to_tray_v2_promptfix \
  --exp-name robotwin_handover_to_tray_hg_dagger_r1 \
  --data.repo-id <hil-leRobot-repo> \
  --batch-size 128 \
  --num-workers 8 \
  --fsdp-devices 4 \
  --num-train-steps 2500 \
  --save-interval 1000
```

> 注意：这一节的数据过滤/转换脚本尚未完成，当前仓库只完成了数据收集端。

## 6. 最终测试

### 6.1 无人验收 smoke

```bash
bash /hdd/robotwin-hil/local_serving/run_local_takeover_validation.sh 40003 87
```

会在第 87 步自动触发专家接管，验证丢弃 chunk → 专家恢复 → 成功的链路。

### 6.2 扰动救援测试

```bash
cd /hdd/robotwin-hil
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
  DISPLAY=:1 XAUTHORITY=/home/ruio/.Xauthority \
  bash ./enter_robotwin_hil.sh python -u scripts/perturbation_rescue.py \
    --host 127.0.0.1 --port 18300 \
    --policy-name Pi_05_RobotTwin \
    --ckpt-name v2_promptfix_9999 \
    --task-config handover_to_tray_v2_promptfix \
    --seed-start 40000 --episodes 100 \
    --lead-in-steps 30 --bias-duration 10 \
    --perturb-mode action_bias --bias-magnitude 0.08 \
    --output-dir /media/ruio/hdd/robotwin-hil/outputs/perturbation_rescue_100
```

报告会汇总 `rescue_rate`、每个 seed 的 `expert_success`、`branch` 和
`executed_stage_ids`。

### 6.3 真实 rollout 上的接管救援测试（live intervention）

在真实策略 rollout 上直接触发接管（不是回放数据），用于回答“test seed
上的真实失败状态，专家能不能救回来”。

触发条件（任一满足即接管）：

- `bar_dropped`：bar 之前被某只手握住，随后两只手都不再握住它；
- `out_of_workspace`：bar 离开可恢复工作区；
- `fixed_step`：兜底触发步数（`--intervene-step`，默认 600）。

```bash
cd /hdd/robotwin-hil
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
  DISPLAY=:1 XAUTHORITY=/home/ruio/.Xauthority \
  bash ./enter_robotwin_hil.sh python -u scripts/live_intervention_eval.py \
    --host 127.0.0.1 --port 18300 \
    --policy-name Pi_05_RobotTwin \
    --ckpt-name v2_promptfix_9999 \
    --task-config handover_to_tray_v2_promptfix \
    --seed-start 31000 --episodes 100 \
    --render-freq 10 --frequency 30 \
    --intervene-step 600 \
    --output-dir /media/ruio/hdd/robotwin-hil/outputs/live_intervention_100
```

输出：

- `live_intervention_records.jsonl`：每个 seed 的触发原因、接管前步数、
  接管时的特权状态、专家 branch/stage、`expert_success`、最终成功与指标；
- `summary_*.json`：整体 `rescue_rate`，以及按触发原因和按专家 branch 分组的
  rescue 统计；
- `video/`、`data/`：完整 policy+HIL 轨迹（`--save-videos none` 可关闭）。

想要无人值守且更快时，可以加 `--save-videos none` 并把 `--render-freq` 调大。

### 6.4 SFT 策略评测记录（policy_eval_record.py）

纯策略评测（不接管），逐 seed 记录结果，并把失败样本的 rollout 视频和 HDF5
保存下来。

当前 100 个 test seed 的评测命令：

```bash
cd /hdd/robotwin-hil
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
  DISPLAY=:1 XAUTHORITY=/home/ruio/.Xauthority \
  bash ./enter_robotwin_hil.sh python -u scripts/policy_eval_record.py \
    --host 127.0.0.1 --port 18300 \
    --policy-name Pi_05_RobotTwin \
    --ckpt-name v2_promptfix_9999 \
    --task-config handover_to_tray_v2_promptfix \
    --seed-start 31000 --episodes 100 \
    --render-freq 10 --frequency 30 --save-freq 15 \
    --save-videos all \
    --output-dir /media/ruio/hdd/robotwin-hil/outputs/sft_policy_eval_100
```

输出：

- `eval_records.jsonl`：每个 seed 的 `success`、`policy_steps`、
  `final_check_success`、`final_success_metrics`、`bar_pose`、`current_stage_id`；
- `summary_*.json`：成功率汇总；
- `data/episode_*.hdf5`、`video/episode_*.mp4`：完整 rollout（`--save-videos
  failure` 只存失败，`none` 只存 JSON）。

`--step-limit N` 可临时截断步数，适合快速 smoke。

注意：`31000-31103` 是冻结的 test 区间，只用于最终评测，不要用于训练或调参
（`hg_dagger_handover.py` 里有对应的保护）。

## 7. 常用路径速查

- 仓库根目录：`/hdd/robotwin-hil`
- 策略服务端口：`127.0.0.1:18300`
- 本地服务启动：`local_serving/start_local_policy_server.sh`
- 验收验证：`local_serving/run_local_takeover_validation.sh`
- HG-DAgger 入口：`RoboTwin/scripts/hg_dagger_handover.py`
- 扰动救援：`RoboTwin/scripts/perturbation_rescue.py`
- 策略评测记录：`RoboTwin/scripts/policy_eval_record.py`
- 真实接管救援测试：`RoboTwin/scripts/live_intervention_eval.py`
- 本地 checkpoint 配置：`RoboTwin/XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml`
