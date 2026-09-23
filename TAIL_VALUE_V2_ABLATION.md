# Tail coverage value v2：策略 bootstrap 与 alpha 消融

分支：`tail-coverage-value-v2`，基于 `codex/tail-coverage-value` 的 `1241d3f`。
历史报告仍描述旧 expert-bootstrap 实验；v2 的结果输出到新的目录。

## 目标函数

对记录的专家转移 `(o_t, a_t^E, o_{t+1})`：

\[
y_t=1+\gamma\frac1K\sum_{k=1}^K
C_{\bar\phi}(o_{t+1},\tilde a_{t+1}^{(k)}),
\]

\[
\mathcal L=\mathbb E[(C_\phi(o_t,a_t^E)-\operatorname{sg}(y_t))^2]
+\alpha\mathbb E[\tfrac1K\sum_k C_\phi(o_t,\tilde a_t^{(k)})-C_\phi(o_t,a_t^E)].
\]

`a_pi[t]` 来自同一个专家观测下的 K 次独立策略调用，每次只取 chunk 首动作。
target 读取 `a_pi[t+1]`，对各候选的 **target-network 评分** 取均值；不对动作
先求均值。target 停止梯度，EMA 为 0.005。没有输出平方正则，没有最远候选筛选。
AdamW 仍保留原先的权重衰减 1e-4。

checkpoint 配置记录 `bootstrap=policy_next_action_mean`。恢复训练时严格检查
完整配置及缓存指纹，拒绝用旧 expert-bootstrap checkpoint 续训。
离线评估读取 checkpoint 自身的 bootstrap 类型，旧权重仍按旧 TD 定义报告。

## 数据与四组配置

复用已经完成的 `sft_450` 缓存，源位置为
`new_server_my_2:/root/data/my/robotwin-hil/outputs/tail/cache/sft_450`。
缓存指纹：`e93de3b4b29eccd38a3e4bdd004c3488b87018305b5d6ae3971c1e0cbfe43a05`。

- 405 train / 45 val，按完整 episode 固定划分；119,956 / 13,388 个有效转移。
- 每个观测 K=4 个 14D 绝对关节动作候选，复用下一帧候选，无须重新调用 Pi0.5。
- 三视角冻结 ResNet18 特征 1536D + 14D 状态 + 14D 动作；三层 256/LN/SiLU MLP。
- 四组均为 seed=42、10,000 steps、batch=256、AdamW、lr=1e-4、weight_decay=1e-4、
  gamma=0.99、EMA=0.005、grad clip=1.0、score limit=1e4。
- 只改变 alpha：0.1、0.5、0.01、1.0；每组从相同随机初始化开始。
- 每 50 steps 记录训练指标，每 500 steps 完整验证并保存 checkpoint。
- 训练结束自动评估 45 条 val，输出逐帧 Parquet、PNG 和 `metrics.json`。

标准化只拟合训练数据。所有组记录相同初始化的 SHA-256，消融汇总会核验
各组初始化一致、除 alpha 外配置完全一致。暂为单 seed 消融，不能据此给出跨 seed 置信区间。
当前 val 是 critic 内部验证数据，基础策略见过这些示范。现有 v1 重复数据不能算独立
heldout，HIL 没有覆盖真值，本轮只评估 val。示范转移仍是下一保存帧关节目标近似。

## 后台训练

使用已配置好的 PyTorch CUDA Python，不需要加载 Pi0.5 checkpoint 或启动策略服务。
脚本根据 `nvidia-smi` 的物理 GPU 索引解析 UUID，使用 UUID 绑定每个子进程；
启动前拒绝占用中的 GPU。`--output-dir` 必须是新目录。

```bash
mkdir -p /path/to/outputs/tail_v2
nohup /path/to/python -u RoboTwin/scripts/run_tail_alpha_ablation.py \
  --cache-dir /path/to/cache/sft_450 \
  --output-dir /path/to/outputs/tail_v2/alpha_ablation_seed42 \
  --gpus 0 5 6 7 --alphas 0.1 0.5 0.01 1.0 \
  --steps 10000 --eval-every 500 --log-every 50 \
  --code-revision "$(git rev-parse HEAD)" \
  > /path/to/outputs/tail_v2/alpha_ablation_seed42.log 2>&1 < /dev/null &
```

对应关系按参数顺序：GPU 0→0.1，GPU 5→0.5，GPU 6→0.01，GPU 7→1.0。
目录结构：

```text
status.json                    # 编排 PID、每组 GPU UUID/PID/阶段/命令/退出码
logs/alpha_*.train.log          # 完整训练日志
logs/alpha_*.eval.log           # 离线评估日志
runs/alpha_*/config.json
runs/alpha_*/metrics.jsonl
runs/alpha_*/step_*.pt
runs/alpha_*/last.pt
reports/alpha_*/metrics.json
reports/alpha_*/scores.parquet
reports/alpha_*/.../*.png
comparison.json                # 完成后汇总四组、核验配置及初始化
```

`metrics.jsonl` 记录 TD、保守项、alpha 加权保守项、总 loss、梯度范数、专家/策略/
后继候选/TD target/专家减策略评分的均值、标准差与分位数。分数超过绝对值 1e4
或出现非有限值时训练失败并保留上一有效 checkpoint，编排器报告失败，其余组继续。
不会把低 loss、较大 gap 或输出分数解释成机器人成功率。

后台过程会训练至总步数并自动评估，不发送 SIGSTOP。需要恢复某个失败组时，
单独调用 `train_tail.py --resume` 并重复其超参数；不要重用整个消融输出目录。

## 核验

```bash
CUDA_VISIBLE_DEVICES='' /path/to/python -m unittest discover \
  -s RoboTwin/scripts/tests -p 'test_tail_value.py' -v
```

测试涵盖后继候选正确索引/episode 边界、均值在评分之后、target 无梯度、
四种 alpha 的 loss 与梯度方向、EMA、断点恢复、数值保护、训练/评估 TD 一致、
旧 checkpoint 评估语义，以及 HIL 无 TD/控制来源不进入网络。
真实 GPU 冒烟可以对上述后台命令指定新输出目录，改成 `--steps 100 --eval-every 100
--max-plots 1`；冒烟运行的 checkpoint 不用于正式消融初始化。

### 本次核验记录（2026-09-24）

在 `new_power_3` 上使用独立环境
`/data-training/robotwin-hil-lyt/outputs/tail_v2/venv`；Python 3.11、
PyTorch 2.13.0+cu130、CUDA 13.0，显卡为 A800-SXM4-80GB。
该环境复用现有只读 Torch 安装，新增的评估依赖安装在独立 venv 内。

- 27 项 unittest 全部通过（包括完整恢复一致性测试）。
- 四张卡各训练 100 steps 后自动评估 45 条 val，全都正常退出。
- 四份报告各包含 13,433 个观测，其中 13,388 个有效 TD 转移；生成了四份
  `scores.parquet` 和共 12 张 PNG。
- 四组初始模型哈希一致，配置除 alpha 外完全一致，源缓存指纹与旧版正式缓存一致。
- 对照源文件 SHA-256 确认冒烟所运行的训练/评估/编排代码与提交源文件一致。

| alpha | 100-step 冒烟 val TD MSE |
| --- | ---: |
| 0.1 | 0.00125840249 |
| 0.5 | 0.00126324984 |
| 0.01 | 0.00125744982 |
| 1.0 | 0.00127063839 |

这些数值只证明短程训练和评估接口可运行，不作为 alpha 选择结论。
冒烟目录为 `/data-training/robotwin-hil-lyt/outputs/tail_v2/smoke_alpha_seed42`。
正式消融使用新目录 `/data-training/robotwin-hil-lyt/outputs/tail_v2/alpha_ablation_seed42`，
从同一个随机初始化重新开始 10,000 steps；正式运行的具体 commit、PID 与状态以
该目录 `status.json` 为准。
