# 四任务 HG-DAgger 正式测评负结果（2026-11-08）

结论（给后续读者的一句话）：**当前多任务 HG-DAgger 流程没有走通**——SFT50+HIL15 → 300 step 再训后的 DAgger299，在已测任务上相对各自 SFT4999 参照全面掉点，不能写成 DAgger 增益或“流程已闭环成功”。

更新时间：2026-11-08（Asia/Shanghai）  
实验名：`dagger_four_h50_5050_300_20261107`  
协议：官方 min-trace / `demo_clean` / `expert_check` / `seen` / seed0→100000 / H50 / frequency 字段 50 / 无人工；1 smoke 不计 50。

## 主表

| 任务 | 批次 | DAgger299 formal | 旧 SFT4999 参照（交接） | 粗差 |
| --- | --- | --- | --- | --- |
| `handover_block` | A800 `retry1`（`REPORTED`，交接快照） | **0/50（0%）** | ~24/50（48%） | −24 |
| `rotate_qrcode` | Ubuntu `ubuntu1`（`VERIFIED`） | **16/50（32%）** | ~39/50（78%） | −23 |
| `place_empty_cup` | Ubuntu `ubuntu1`（`VERIFIED`） | **4/50（8%）** | ~32/50（64%） | −28 |
| `stack_blocks_three` | 未测 | — | ~21/50（42%） | ? |

`stack_blocks_three`：Ubuntu 本地缺少 `.../dagger_5050_h50_state_300/299`，且本机无法登录 A800 拉取；本轮未补测。

## 已核实的 Ubuntu 产物

根目录：

```text
/media/ruio/hdd/robotwin-hil-github/outputs/dagger_four_h50_5050_300_20261107/evaluations/
```

| 批次 | formal | 专家拒绝 | runner |
| --- | --- | --- | --- |
| `place_empty_cup_ubuntu1` | 4/50，rate=0.08 | 12 | exit 成功，`phase=complete` |
| `rotate_qrcode_ubuntu1` | 16/50，rate=0.32 | 5 | exit 0，`phase=complete` |

轻量证据副本（不含 trace/NPZ/视频）：

```text
docs/resample_handoff_20261108/evidence_eval_20261108/
  place_empty_cup_ubuntu1_formal_summary.json
  place_empty_cup_ubuntu1_summary.json
  rotate_qrcode_ubuntu1_formal_summary.json
  rotate_qrcode_ubuntu1_summary.json
```

## 如何理解“没有走通”

1. **工程闭环**：采集 → 转换（含本轮 `diagnostic_all` 例外）→ A800 训练到 299 → 部分正式测评，链路能跑完。  
2. **科研闭环**：相对冻结 SFT 基线提高自主成功率——**未达成**。已测三任务全部显著低于交接中的 SFT 参照。  
3. **归因边界**：这是“当前实现 + 数据资格 + 训练配方 + 评测批次”的净负向结果；交接仍列出混杂因素（diagnostic_all、schema/gate、norm 重算、跨机协议等）。在未做同机同 scene/prompt 的严格对照前，**不要**简化成“HG-DAgger 算法天生只有副作用”，也**不要**写成已验证有效。

## 明确不要再声称的内容

- 不要写四任务 DAgger 已成功泛化或已超过 SFT。  
- 不要把 tray 历史 / H10 / 接管率与本表混比。  
- 不要把 A800 `retry1` 与 Ubuntu `ubuntu1` 局数合并成同一成功率。  
- 在未冻结数据资格与评测协议前，不要打开四任务 `--es-mode enhanced`。

## 建议的下一步（PROPOSED）

1. 拉取并测评 `stack_blocks_three` 299，补齐四任务表。  
2. 同机同协议重核 SFT4999 vs DAgger299（scene/prompt/`_base_task`）。  
3. 只用严格 accepted HIL 重导出/重训一版，隔离 `diagnostic_all` 例外的影响。
