# Tail coverage value：数据、训练与离线诊断

三个入口位于 `RoboTwin/scripts/`：`tail_data.py`、`train_tail.py`、
`eval_tail_value.py`。它们不会执行机器人动作、更新 Pi0.5、修改已有 HIL
采集器或启动全量长时间训练。

## 实验含义与数据划分

学习的是状态—动作支持度 `C(o,a)`，输出为实数，不是成功概率。
现有专家 action 来自下一保存帧关节目标，不是逐步真实命令；因此本版是
**近似转移原型**，不能称为严格动力学 TD 实验或已验证恢复能力。

| 数据 | 划分 | 用途 |
| --- | --- | --- |
| 450 条 prompt-fixed LeRobot SFT | 按完整 episode、seed 42 固定为 405 train / 45 val | 训练 / 内部监控；45 条仍被基础策略见过 |
| 原生 episode 450–499 | 原 manifest 的 `validation_episodes` | 50 条独立示范诊断，不参与拟合统计量、模型选择或调参 |
| 已保存 HIL raw | `hil` | 只看记录观测上的策略建议评分与接管区间 |

**不使用** `outputs/sft_policy_eval_100`。HIL 接管不等于低覆盖，接管后成功
不等于自主成功；HIL 不包含 `a_demo` 或 TD 样本。其曲线是“在记录观测上
重新询问当前策略”的分数，不是当时执行动作的分数。保存帧间隔含义不同，
横轴只使用帧索引。

示范只构造相邻帧 `(o[t], action[t], o[t+1])`，并检查
`action[t] ≈ state[t+1]`。末帧没有转移，不用重复末帧补齐、不把文件截断
当作任务终止。当前缓存没有终止奖励；常数支持信号为 1，折扣按保存转移计，
不是按物理秒计。有限轨迹截断及新状态外推都是需另行验证的局限。

## 候选、网络与损失

每个观测独立调用冻结策略 K 次，默认 K=4。每次返回一个 10 步 chunk，
只取首个 14D 动作，顺序为 `[left_arm(6), left_gripper, right_arm(6), right_gripper]`。
**不是**从同一个 chunk 里拿不同时刻的动作冒充独立候选。
服务已输出绝对关节/夹爪目标，不再次做 OpenPI delta 或归一化变换。
缓存 manifest 会明确记录 `action_space=absolute_joint_qpos` 和动作顺序；
`use_delta_joint_actions=True` 只表示 SFT 模型内部训练时对机械臂关节维度做
delta 变换，策略适配器输出后已经恢复为绝对 joint/qpos，因此候选距离直接
计算 `a_policy_abs - a_demo_abs`。
所有候选都保留在缓存中；不做“确定不在专家集”的预筛选。

三路 RGB 为 head、left wrist、right wrist；保持宽高比缩放、居中零填充到
224×224，再按 ImageNet mean/std 归一化。共享、冻结且处于 eval 模式的
ResNet18 输出 3×512 维特征。MLP 输入 `1536 + 14 state + 14 action`，三层
256/LN/SiLU，实数输出。状态、动作 mean/std 只用 train 有效转移估计，
std 下限为 0.05；metadata 不进入网络。

训练有两个**独立**开关：

- `--candidate-reduction mean`（默认）：`R = mean_k C(o, a_pi[k])`。
- `--candidate-reduction farthest`：`R = C(o, a_pi[k*])`，其中
  `k* = argmax_k sqrt(mean_j ((a_pi[k,j]-a_demo[j])/action_scale[j])**2)`。
  距离使用训练统计量，不使用 critic 分数。它改变了采样分布，是困难候选变体，
  不是原式策略期望的无偏估计。K 越大，最远选择也会变化，比较时固定 K 和缓存。
- `--mode original`：`loss = TD + alpha * mean(R-C_demo)`。
- `--mode stabilized`（默认）：增加
  `output_reg/2 * mean(C_demo**2 + mean_k C_pi[k]**2)`，始终约束全部候选。

两种候选模式的 target 都是
`y = 1 + gamma * mean_k C_target(o_next, a_pi_next[k])`，停止梯度。
不能用专家的后继观测当作某个不同候选动作的已观测执行结果；候选只参与
评分，TD 当前动作始终是记录的专家目标。

默认 AdamW：lr=1e-4、weight_decay=1e-4、batch=256、gamma=.99、alpha=.01、
output_reg=1e-4、gradient clip=1、target EMA=.005、10,000 steps。
这些是起始值，不是调优结论。每 500 steps 验证和保存；评分超出绝对值 1e4
或出现非有限值时中止并记录 `failure.json`，不静默裁剪分数。

## Ubuntu：独立 checkout 与依赖

已有 `/hdd/robotwin-hil` 工作区可能有未提交改动。使用新 worktree，
不要 checkout/reset 那个工作区。以下目录不存在时运行一次：

```bash
git -C /hdd/robotwin-hil fetch origin
mkdir -p /hdd/robotwin-hil/outputs/tail
git -C /hdd/robotwin-hil worktree add --detach \
  /hdd/robotwin-hil/outputs/tail/checkout origin/codex/tail-coverage-value

export TAIL_ROOT=/hdd/robotwin-hil/outputs/tail
export TAIL_PYTHON=/hdd/miniconda3/envs/robotwin_hil/bin/python
export ROBOTWIN_ROOT=/hdd/robotwin-hil/RoboTwin
export TAIL_POLICY_CONFIG=$ROBOTWIN_ROOT/XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml
export TAIL_ENCODER=/home/ruio/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
cd "$TAIL_ROOT/checkout/RoboTwin"
```

普通离线脚本使用现有 `robotwin_hil` conda Python；策略服务继续使用独立 OpenPI
`.venv`。不要为此混装或升级两个环境的 torch/jax。
已用依赖：Python >=3.10、NumPy、torch、匹配的 torchvision、PyYAML、pyarrow、
h5py、matplotlib、FFmpeg，以及仓库 XPolicyLab WebSocket 依赖。无需 SAPIEN、
PyAV、LeRobot Python 包或 pytest 来运行这些脚本。

```bash
"$TAIL_PYTHON" -c 'import torch, torchvision, numpy, yaml, pyarrow, h5py, matplotlib; print(torch.__version__)'
ffmpeg -version
"$TAIL_PYTHON" -m unittest discover -s scripts/tests -p 'test_tail_value.py' -v
```

## 1. 同步输入（数据不进 Git）

以下复用 Ubuntu 已配置的 JG SSH key 和局域网地址。若机器不在同一局域网，
只替换可达的 JG 地址；不复制或上传私钥。

```bash
mkdir -p "$TAIL_ROOT/input/sft" "$TAIL_ROOT/input/native"
rsync -a --info=progress2 -e 'ssh -i /home/ruio/.ssh/id_ed25519_JG' \
  ruihao@192.168.101.11:/home/ruihao/robotwin_hil/lerobot_datasets/ruio248/robotwin_handover_to_tray_v2_promptfix/ \
  "$TAIL_ROOT/input/sft/"

rsync -a --info=progress2 -e 'ssh -i /home/ruio/.ssh/id_ed25519_JG' \
  --include='/split_manifest_v1.json' --include='/data/' \
  --include='/data/episode_00004[5-9][0-9].hdf5' --exclude='*' \
  ruihao@192.168.101.11:/home/ruihao/robotwin_hil/RoboTwin/data/handover_to_tray_v1/handover_to_tray/aloha_agilex/ \
  "$TAIL_ROOT/input/native/"
```

同步没有 `--delete`，不会清理已有文件。完整 SFT 约 3 GB；native 只复制
50 条留出记录和划分。HIL 直接读取 Ubuntu 现有 raw 文件，不更改它们。

## 2. 独占策略服务（另一个终端）

确认 18301 空闲并有足够显存后启动；原 18300 服务不受影响。
此服务只给一个 `tail_data.py` 客户端使用，不能同时运行采集/推理客户端：
`update_obs` 与 `get_action` 是分离的、有状态的 RPC。

```bash
/hdd/robotwin-hil/enter_robotwin_hil.sh env \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONUNBUFFERED=1 \
  /hdd/robotwin-hil/RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/openpi/.venv/bin/python -u \
  XPolicyLab/setup_policy_server.py \
  --config_path XPolicyLab/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml \
  --host 127.0.0.1 --port 18301
```

`--policy-config` 是调用者声明的 checkpoint 身份；当前服务协议不提供权重
远程证明，脚本不能核验服务进程是否真的加载了另一个 checkpoint。
缓存记录该配置（不把 host/port 当作模型身份）。WebSocket 不传噪声 seed，
所以客户端 `--seed` 只控制数据划分；训练比较必须复用已保存候选。

## 3. 生成三类缓存（依次运行）

先用文末的小规模冒烟参数确认接口。完整 133,794 帧、K=4 需要约 535,176 次
策略推理，数据缓存可能明显比小型 critic 训练更耗时；下列全量命令不会被脚本
自动执行。后继候选直接复用下一帧缓存，不额外再采样一次。

```bash
"$TAIL_PYTHON" scripts/tail_data.py \
  --source-format lerobot --dataset-root "$TAIL_ROOT/input/sft" \
  --policy-url ws://127.0.0.1:18301 --policy-config "$TAIL_POLICY_CONFIG" \
  --encoder-weights "$TAIL_ENCODER" --output-dir "$TAIL_ROOT/cache/sft" --device cuda

"$TAIL_PYTHON" scripts/tail_data.py \
  --source-format native --dataset-root "$TAIL_ROOT/input/native" \
  --policy-url ws://127.0.0.1:18301 --policy-config "$TAIL_POLICY_CONFIG" \
  --encoder-weights "$TAIL_ENCODER" --output-dir "$TAIL_ROOT/cache/heldout" --device cuda

"$TAIL_PYTHON" scripts/tail_data.py \
  --source-format hil --dataset-root /hdd/robotwin-hil/outputs \
  --policy-url ws://127.0.0.1:18301 --policy-config "$TAIL_POLICY_CONFIG" \
  --encoder-weights "$TAIL_ENCODER" --output-dir "$TAIL_ROOT/cache/hil" --device cuda
```

`hil` 可接受单个 episode、raw 目录、某个 collection 根目录或 outputs 根目录；
只匹配 `hg_dagger_collection_r*/raw/episode_*`，不扫描冻结策略评估集。
HIL pickle 以及之后加载的 `.pt` 只允许使用可信本地文件。

所有套件必须使用相同 encoder 权重参数、策略身份与 K，避免混合表示。
native 输入使用修正后的固定任务提示词，原 HDF5 不改动。

缓存目录结构：

```text
manifest.json             # 来源指纹、固定划分、表示/策略配置、完整性与限制
episodes/<id>.npz         # features[T,1536], state[T,14], a_pi[T,K,14], frame_index
                          # demo: a_demo[T,14] 与 valid_transition[T]
                          # HIL: control_source[T]；valid_transition 全 false
```

同一命令追加 `--resume` 只跳过校验通过的完整 episode；中断 episode 会重新
采样，不能承诺保持中断前的随机候选。输入/配置指纹变化或已完成 shard 损坏
会拒绝恢复，不静默覆盖。不要让两个进程写入同一个缓存目录。
完成后用该专用服务终端的 Ctrl-C 关闭 18301，释放显存；不要关闭 18300。

## 4. 在同一缓存上训练对照

```bash
"$TAIL_PYTHON" scripts/train_tail.py --cache-dir "$TAIL_ROOT/cache/sft" \
  --output-dir "$TAIL_ROOT/runs/stable_mean" --mode stabilized --candidate-reduction mean

"$TAIL_PYTHON" scripts/train_tail.py --cache-dir "$TAIL_ROOT/cache/sft" \
  --output-dir "$TAIL_ROOT/runs/stable_farthest" --mode stabilized --candidate-reduction farthest

"$TAIL_PYTHON" scripts/train_tail.py --cache-dir "$TAIL_ROOT/cache/sft" \
  --output-dir "$TAIL_ROOT/runs/original_mean" --mode original --candidate-reduction mean
```

默认 seed 相同，缓存相同。首先比较 stable_mean 与 stable_farthest，隔离
候选选择的影响；再与 original_mean 比较输出正则影响。

恢复时重复原超参数并增加总步数，例如：

```bash
"$TAIL_PYTHON" scripts/train_tail.py --cache-dir "$TAIL_ROOT/cache/sft" \
  --output-dir "$TAIL_ROOT/runs/stable_mean" --mode stabilized --candidate-reduction mean \
  --resume "$TAIL_ROOT/runs/stable_mean/last.pt" --steps 15000
```

如果需要“后台启动一轮、达到指定 checkpoint 后离线评测、再挂起训练”，使用
仓库内的 `scripts/run_tail_train_eval.sh`。默认总步数为 10,000，在第 5,000
步 checkpoint 上评测，然后对仍在运行的训练进程发送 `SIGSTOP`；状态文件会写入
run 目录，继续训练使用其中的 `kill -CONT <pid>`。它不启动策略服务，也不执行
机器人动作：

```bash
nohup bash scripts/run_tail_train_eval.sh \
  --python "$TAIL_PYTHON" \
  --sft-cache "$TAIL_ROOT/cache/sft" \
  --heldout-cache "$TAIL_ROOT/cache/heldout" \
  --hil-cache "$TAIL_ROOT/cache/hil" \
  --run-dir "$TAIL_ROOT/runs/stable_mean" \
  --report-dir "$TAIL_ROOT/reports/stable_mean_step5000" \
  > "$TAIL_ROOT/runs/stable_mean_orchestrator.log" 2>&1 &
echo $! > "$TAIL_ROOT/runs/stable_mean_orchestrator.pid"
```

脚本拒绝覆盖非空的 run/report 目录；`run_dir.train.log` 保存训练输出，
`run_dir/orchestration_state.txt` 保存暂停状态、训练 PID、checkpoint 和继续命令。

Checkpoint 保存 online/target、optimizer、训练统计量、随机状态、缓存指纹。
不自动从独立示范/HIL 选择最佳 checkpoint。`last.pt` 是最近一次有效保存。

## 5. 离线评估（不需要策略服务）

```bash
"$TAIL_PYTHON" scripts/eval_tail_value.py \
  --cache-dir "$TAIL_ROOT/cache/sft" "$TAIL_ROOT/cache/heldout" "$TAIL_ROOT/cache/hil" \
  --checkpoint "$TAIL_ROOT/runs/stable_mean/last.pt" \
               "$TAIL_ROOT/runs/stable_farthest/last.pt" \
               "$TAIL_ROOT/runs/original_mean/last.pt" \
  --suite all --output-dir "$TAIL_ROOT/reports/comparison_001"
```

只看内部验证时使用 `--suite val` 和 SFT 缓存；其指纹必须与 checkpoint 的
训练缓存一致。`--suite all` 要求三类套件齐全，不静默跳过缺失数据。
每 checkpoint/suite 默认绘制最多 10 条轨迹，**全部**帧仍参与指标与 Parquet。

- `metrics.json`：每 checkpoint/suite 的 TD MSE、评分分位数、平均 gap、
  最远距离及其分数、距离/分数相关性、候选多样性。
- `scores.parquet`：逐帧分数、全部候选分数/距离、最远索引和可空 TD 字段。
  HIL 的专家动作分数、距离与 TD 均为空。
- PNG：轨迹曲线、HIL 接管区间、评分直方图、动作距离散点图。

保守项本身推动“专家分高于策略”，因此 gap 大或 loss 下降不是恢复有效性的
证据。输出正则抑制发散也不能解决 OOD 状态泛化；这些仍需后续仿真分支实验。

## 小规模真实冒烟

使用独立输出目录，SFT 加 `--episode-limit 2`、native/HIL 各加
`--episode-limit 1`，HIL 指定单个 collection。小样本内部划分是 1 train / 1 val，
不能冒充正式 405/45 实验。三个训练命令各加 `--steps 100 --eval-every 100`，
再运行 all-suite 评估。该过程仅验证接口、对齐和数值运行，不宣称算法有效。
