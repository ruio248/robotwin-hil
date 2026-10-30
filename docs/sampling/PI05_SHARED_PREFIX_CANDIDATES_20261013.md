# Pi0.5 同观测 K=32 共享前缀推理验证（2026-10-13）

## 实现与启用边界

- `openpi/src/openpi/models/pi0.py::sample_actions_shared_prefix` 对单个观测运行一次三视角视觉/语言编码及 PaliGemma KV 预填充，再把 KV 广播给 K 份独立噪声并行运行完整 10 步 flow-matching 去噪。普通 `sample_actions` 与新方法共用同一去噪积分器。
- `openpi/src/openpi/policies/policy.py::infer_candidates` 只做一次输入变换，并将输出状态广播给 K 条轨迹，以保留同一套反归一化和绝对关节动作变换。
- RoboTwin 适配器只有在策略服务进程启动前设置 `OPENPI_SHARED_PREFIX_CANDIDATES=1` 时才走新路径；未设置时仍用原来的复制观测 batch 路径。当前线上 18300 进程未重启，仍为旧路径。
- K=32 所需的 HIL 客户端配置是 `HIL_ES_POLICY_MICROBATCH=32`。只改客户端而没有切换策略服务时，服务的能力检查会拒绝请求，不会悄悄退回旧实现。

## 同机实测

主机：Ubuntu RTX 4090；同一 baseline-9999 checkpoint、同一已保存 HIL 观测、同一份固定的 32 组 `[10,32]` 高斯噪声；JAX 显存预分配比例均为 `0.30`。两种模式各用独立短进程，完成一次预热后取两次稳态运行中位数；返回完整 `[32,10,14]` 物理绝对关节 chunk。

| 模式 | 稳态生成 32 条耗时 | JAX allocator 峰值 | 完成情况 |
|---|---:|---:|---|
| 原来 4 次 batch=8 | 1.3709 s | 7,559,316,992 B | 完成 |
| 共享前缀、一次 K=32 | 0.1067 s | 7,312,685,824 B | 完成，32 条互异 |

此观测上的速度比约为 12.85×。固定噪声下，新旧动作的绝对差：全部元素均值 0.000206、最大值 0.001953；12 关节完整 chunk 的逐候选 RMS 差中位数 0.334 mrad，而这 32 条新候选两两 RMS 差中位数 11.13 mrad。两实现**数值接近但非逐位相同**，不能表述为完全等价；后续上线还须观察实际 HIL 时序与控制效果。

独立 18305 WebSocket 诊断服务完成了一次真实 `update_obs` → `get_action_candidates(32)` 调用，返回 32 条互异的 `[10,14]` chunk，稳态该次 RPC 用时 0.124 s。诊断后已关闭 18305、恢复原 18304 服务；18300 HIL 服务始终未停止或重启。

原始数值及固定噪声动作保存在 `outputs/pi05_shared_prefix_diag_20261013/`。该测试只覆盖一个已保存观测和一次 WebSocket 调用，不是整段 HIL 采集的延迟/成功率测评。0.30 的 allocator 峰值距限额约 255 MB，余量不大；不同形状、配置或并行负载仍需独立验证。
