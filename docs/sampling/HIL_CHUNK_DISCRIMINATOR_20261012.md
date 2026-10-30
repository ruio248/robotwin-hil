# HIL 完整 chunk 判别器接入

## 算法与不变项

新入口：`local_serving/run_hg_dagger_discriminator.sh`。旧 `run_hg_dagger_smooth.sh` 默认继续使用旧 TD 首动作评分器。

- 策略：baseline `v2_promptfix_9999`；任务提示词、策略 norm_stats、绝对 joint 顺序、`take_action(qpos)` 不变。
- 评分器：原 ImageNet ResNet18 冻结视觉特征 + `disc_s42.pt` 的完整 chunk BCE logit；不是 R3M/DINOv2 编码器消融模型。不要混用编码器资产。
- 默认 K=32、H=10、policy microbatch=8（在当前 4090 服务的 JAX memory fraction=0.30 下实测可用）。每个 batch=8 是一次 `policy.infer`，生成 8 个独立 chunk；K=32 分四次顺序 RPC 生成全部候选。判别器每次决策三视角只编码一次，所有候选复用。策略网络本身当前仍按 batch 编码图像，尚未加入跨候选的视觉前缀缓存复用。
- 每个完整 `[10,14]` chunk 一个分数 B，不取首动作，也不模拟执行各个候选。
- B 高表示专家支持强，增强采样继续偏向低 B：

  `w_k = softmax_k(-beta * (B_k - min(B)) / sigma_cal)`

  `A_exec[h,j] = sum_k w_k * A_k[h,j]`

  每个候选的同一权重覆盖全部 10 步和 14 维（包括两个夹爪）。不是选择一个候选，也不是把一个 chunk 的 10 个时刻互相平均。
- 最终仍执行 10 个顺序目标；接管、取消、成功和 episode 上限可以提前中断。
- 关闭 e 窗口时仍单次普通推理，无额外候选或视觉评分。e 再按取消，i/r/x/q 的行为不变。

## 校准与防止混用

读取 `outputs/coverage_benchmark_v1_20261009/fit/calibration.json` 的 `disc_s42` 项，默认 beta=100、sigma_cal=1.5994237859316014。必须通过同目录已完成 manifest 的 calibration SHA 和 checkpoint SHA 校验；checkpoint 还必须声明 H=10、absolute_joint_chunk、balanced_source_bce。评分用原始 logit，不加 sigmoid。

`HIL_ES_BETA` 留空时使用冻结校准；显式设置才覆盖，并记录 effective_beta。校准退化时均匀权重，NaN/形状错误则报错，不能伪装成正常结果。

该校准的 K=4 中位 ESS/K=0.8848395，没有达到目标 0.5。这里是固定温度重加权，不是硬 KL 约束。日志中的 candidate_weight_kl_to_uniform 仅是候选索引上的分类分布 KL，不是最终平均动作分布相对原策略的 KL 保证。

## Ubuntu 启动

默认 microbatch=8 与旧服务已支持的候选接口相容。batch 大于 8 时，采集脚本会查询服务能力；服务不支持或 JAX memory fraction=0.30 报告安全上限 8 时，会在 rollout 前明确报错。旧的策略服务进程不会自动加载文件更新；使用 batch 大于 8 需要用新版 adapter 重启策略服务。采集脚本本身不会启动、停止或重启策略服务。

```bash
cd /hdd/robotwin-hil-enhanced-sampling
bash local_serving/run_hg_dagger_discriminator.sh
```

仅检查展开参数，不连接服务、不打开窗口：

```bash
HIL_DRY_RUN=1 bash local_serving/run_hg_dagger_discriminator.sh
```

`HIL_ES_POLICY_MICROBATCH` 默认 8；K=32 因此发四次顺序 batch=8 请求。可显式设为 4 以使用八次请求，或设为更大值；在当前 JAX 0.30 配置下不建议超过 8，启动检查会拒绝超出服务安全上限的值。不同 batch 配置使用新采集目录，避免断点续跑混用。

WebSocket 请求等待期间通过 `responsive_wait` 在主线程约每 33 ms 刷新窗口并轮询按键；物理仿真不前进、观测不变。收到 i/e/x/q 后立即记录意图，继续刷新并等待当前请求返回，然后丢弃其输出并处理原有控制事件。请求不会并发重叠，也不会把尚未完成的推理与 reset 或新的观测混用。它改善窗口与输入响应，但不能消除计算本身的时间；接管开始仍须等待当前请求排空。非 WebSocket 的旧 TCP 路径没有该刷新回调。

首次使用新 batch 尺寸时 JAX 可能编译。性能比较应分别报告冷启动和预热后的耗时；仅功能测试通过不能证明实时速度。

默认仍是 320×180 渲染、1280×720 窗口、非全屏。创建独立 `outputs/hg_dagger_discriminator_<日期时间>`，不要复用旧 TD 采集目录。恢复本方法的同次采集，显式设置其原输出目录及 `HIL_RESUME=1`。

`--frequency 30` 保留。本链路只将其写入 additional_info，Pi0.5 encode_obs 不使用它。真实执行由 TOPP 路径和物理子步决定；`save_freq=15` 是保存步距，不能解释为真实 15Hz。这次没有修改执行频率、物理子步或保存步距。

## 日志解释

`selection_score_basis=current_observation_full_chunk`，`score_action_index=null`。

- `scores`、`weights`、`score_range`、`score_std`：同观测候选评分和权重。
- `effective_sample_size`、`ess_fraction`、`max_weight`：权重诊断。
- `blended_chunk`：实际将顺序执行的 float32 完整 chunk。
- `weighted_candidate_score`：`sum(w_k*B_k)`。
- `blended_chunk_score`：独立计算 `B(o, A_exec)`，复用同一观测特征。非线性网络下这两个分数通常不同。
- `joint_rms_shift_from_uniform`：相对候选均匀平均，整个 10 步、12 个机械臂关节的 RMS 位移，单位 rad；两个夹爪另报。
- `sampling_seconds`、`scoring_seconds`、`output_scoring_seconds`、`selection_seconds`：策略采样、候选评分、输出加权/评分和完整选择耗时。
- `candidate_generation=batched`、`policy_inference_requests=4`：默认 K=32 分成四个 batch=8 请求；计划请求数如有自定义会按真实值记录，并标记 `microbatched`。

完整 chunk 的分数仅在该 chunk 首个动作实际执行时，作为一个决策级 coverage 记入 executed 日志；它不是每一步重新观测得到的 B，也不是执行后的成功概率。动作被提前接管时，原分数依然是原计划完整 chunk 的预测。

## 验证与边界

- 回归覆盖旧首动作 TD、off、vanilla、e 窗口、候选批推理、断点恢复记录格式。
- 新测试覆盖后续动作进入评分、同一权重覆盖完整 chunk、退化均匀、排列不变、负 stride 数组、校准模型错配拒绝，以及测评循环确实执行平均 chunk 并记录其分数。
- `scripts/verify_online_chunk_discriminator.py` 可以读取本项目可信的已保存 RGB/状态/32 候选缓存，比较在线适配器与离线 NeuralScorer 的分数、冻结权重和平均输出，并断言每次决策只编码一次；不进行任何策略 RPC 或仿真执行。
- 本次真实缓存核对使用 CPU 的 6 个观测，32 候选在线/离线最大分数误差为 0，K=4 的加权结果一致。

K=32 修正补充了服务端候选 batch、客户端 microbatch RPC 与完整加权输出、主线程刷新、接管后排空并丢弃输出、RPC 超时传播的回归测试。`scripts/benchmark_hil_candidate_batch.py` 在独立进程加载 baseline，用已保存的同一观测对比 8×4 与 1×32，并检查真实模型调用批大小、候选形状和不同候选数。另通过固定噪声 WebSocket 直接测了现存诊断服务，覆盖 JAX memory fraction=0.30。基准脚本不执行仿真动作，也不连接 18300。

## PiperX 真机批量部署与本实验的计时口径

已检查 JG 上 `/home/ruihao/PiperX-ToolKit/piperx_toolkit/deploy/openpi_remote.py` 和
`scripts/deploy_openpi_bimanual_stall_debug_batch.py`。该脚本名里的 “batch” 是一批有效任务
评测：每个控制循环调用一次 `policy.predict(...)`。`OpenPIRemotePolicy.predict` 在动作队列为空时
发起一次 `client.infer(...)`，取回一条动作序列；随后从 `action_queue` 依次返回已取回的动作。
`--chunk-size 60` 和 `--exec-chunk-size 30` 限制每次响应可用的动作数，不是从同一观测采样
60 个或 30 个独立候选 chunk。它主要减少策略请求频率，和本实验同一观测下采样 K 个独立候选
不是同一工作量。

JG 的部署说明实际从 `/opt/openpi` 启动 `scripts/serve_policy.py`；
`/home/ruihao/openpi/src` 目前为空，`/home/ruihao/openpi-rtc-sync` 是另一份完整源码。
`/opt/openpi/src/openpi/serving/websocket_policy_server.py` 对每条 WebSocket 请求直接调用一次
`policy.infer(obs)`。`/opt/openpi/src/openpi/policies/policy.py` 给单个观测加 batch 维 `1`，
调用 `sample_actions` 后取结果的第 `0` 行，因此这条部署路径向 PiperX 返回一条 `[H,d]`
动作序列。`/opt/openpi/src/openpi/models/pi0.py` 的 `sample_actions` 会在该次推理中
缓存视觉/语言前缀的 KV 供后续去噪步复用；它没有跨请求缓存该观测前缀，也没有自动把
动作时域 `H` 当作独立候选数 `K`。`openpi-rtc-sync` 有可选的 RTC 指导实现，但给出的
PiperX 命令没有提交 RTC 字段，其客户端也只发普通观测。JG 当前没有正在运行的
8000 策略服务，因此历史真机运行具体使用哪个 checkpoint/后端不能仅凭当前源码确认。

HIL 的适配器则显式把同一观测复制成 batch=8；一次请求由模型生成 8 条独立
`[10,14]` chunk，K=32 需要连续四次请求。当前模型每次请求仍重新计算视觉/语言前缀；
若要继续降低 K=32 的采样等待时间，需要单独实现并验证同观测前缀复用，不能把
PiperX 的 30 步动作队列直接移植成 30 个候选。

本次 WebSocket 测试使用同一缓存观测、同一 checkpoint、固定独立噪声，总计生成 32 个不同的
`[10,14]` chunk，并读取服务端实际时延和输出形状。memory fraction=0.30 下，8 次 batch=4 的
预热耗时中位数为 1.536 s；4 次 batch=8 为 1.372 s，降低约 10.7%。batch=8 预热编译耗时
13.36 s；稳定运行后每个 RPC 约 0.342 s。batch=16 在推理时 OOM（申请约 508 MB 失败），
batch=32 OOM（申请约 1.02 GB 失败）。所以当前 HIL 默认选 batch=8；先前用 memory fraction=0.50
测得的 batch=32 约 1.234 s 不适用于 HIL 的 0.30 配置。

这些数据解释了候选采样为什么仍然明显慢于 PiperX 真机执行：HIL 每次决策都必须从同一观测
算出 32 条独立动作轨迹，当前安全设置要做四次 batch=8 推理；PiperX 队列只做一次普通策略推理，
再连续消费响应中的动作序列。它们的批次含义和每次完成的工作量都不同。之前在线 K=32 约
1.56 s 的记录与当前 batch=4 WebSocket 基准接近；新 batch=8 后预计降到约 1.37 s，但尚未在
18300 的 HIL 回路里重测。

这组计时没有在实体机器人控制循环中重测，也不是整条 rollout 的速度结论。HIL 端仍会等待四次
候选请求全部完成才开始执行加权 chunk；响应式等待让画面每约 33 ms 刷新并及时登记按键意图。
收到按键后仍需等当前 RPC 返回，再丢弃结果并切换控制。因而窗口不再被阻塞，候选计算时间仍在。

这些属于代码、评分链路和推理性能验证，不是新的在线成功率或接管率结果。
