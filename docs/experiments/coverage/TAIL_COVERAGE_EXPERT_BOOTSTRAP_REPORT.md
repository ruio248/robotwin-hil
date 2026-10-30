# Tail Coverage Expert-Bootstrap Training Report

**Run date:** 2026-09-21 (Asia/Shanghai)  
**Code version:** `785115f` (`Use expert next-action coverage objective`)  
**Run status:** completed normally, 10,000/10,000 steps  
**Machine:** `new_server_my_2`, NVIDIA A800-SXM4-80GB, CUDA-enabled PyTorch

## 1. What was trained

This run uses the expert-trajectory consistency objective implemented in
`codex/tail-coverage-value`:

\[
y_t^E = 1 + \gamma C_{\bar\phi}(s_{t+1},a_{t+1}^E),
\]

\[
\mathcal L_C =
\mathbb E[(C_\phi(s_t,a_t^E)-y_t^E)^2]
+ \alpha\mathbb E\left[
\frac{1}{K}\sum_k C_\phi(s_t,\tilde a_t^{(k)})
- C_\phi(s_t,a_t^E)\right].
\]

The TD target uses the next saved expert action. The policy candidates are
used only at the current state in the candidate-separation term. The run has
no output-score regularizer, and no `farthest` candidate is selected for the
training objective.

## 2. Data and configuration

The cache is the completed LeRobot SFT cache generated from the
`handover_to_tray` task:

- 450 demonstration episodes in total;
- 405 episodes in the critic training split;
- 45 episodes in the critic validation split;
- 119,956 valid adjacent-frame training transitions;
- 13,388 valid adjacent-frame validation transitions;
- 4 policy candidates per observation;
- 14-dimensional absolute joint action: 6 left-arm joints, left gripper,
  6 right-arm joints, right gripper;
- RGB representation: three camera views, frozen ImageNet ResNet18,
  1,536 visual features plus the 14-dimensional state and action input.

Training hyperparameters:

| Parameter | Value |
|---|---:|
| MLP width | 256 |
| Batch size | 256 |
| Steps | 10,000 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| \(\gamma\) | 0.99 |
| \(\alpha\) | 0.01 |
| Target EMA \(\tau\) | 0.005 |
| Gradient clipping | 1.0 |
| Score guard | absolute value < 1e4 |
| Seed | 42 |
| Output regularizer | none |

The cache manifest was `complete=true`, with the expected 405/45 episode
split. The cache fingerprint recorded in the training configuration is:

```text
e93de3b4b29eccd38a3e4bdd004c3488b87018305b5d6ae3971c1e0cbfe43a05
```

## 3. Final metrics

The last training record at step 10,000 was:

| Metric | Value |
|---|---:|
| Train total loss | 2.3056540e-05 |
| Train TD loss | 2.3259643e-05 |
| Train candidate-separation term | -2.0310283e-05 |
| Train gradient norm | 0.0274109 |
| Train expert score mean | 40.3351746 |
| Train candidate score mean | 40.3351517 |

The last validation record at step 10,000 was:

| Metric | Value |
|---|---:|
| Validation total loss | 3.3440776e-05 |
| Validation TD loss | 3.3479542e-05 |
| Validation candidate-separation term | -3.8765280e-06 |
| Validation expert score mean | 40.3334503 |
| Validation candidate score mean | 40.3334465 |
| Validation candidate-minus-expert mean | approximately -3.88e-06 |

The candidate and expert scores are almost identical on this cached sample.
That means the sampled policy candidates are close to the demonstrations in
the critic representation; it does not establish that coverage is useful or
that the policy succeeds online. It is a diagnostic result, not a success
rate or a probability estimate.

No non-finite-value failure was recorded, and no score-limit violation was
observed during this run.

## 4. Recorded artifacts

The training artifacts remain on the Ubuntu host and are intentionally not
committed to GitHub:

```text
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap/config.json
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap/metrics.jsonl
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap/step_0010000.pt
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap/last.pt
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap.train.log
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap.pid
```

The final checkpoint and `last.pt` were written at 2026-09-21 01:36:47
(Asia/Shanghai). The final process exited after completing step 10,000.

For provenance, the remote records have these hashes:

```text
config.json  5847643c440ee693870b76db6901b035c2f7bb3d232cdfbdb225eaee081053ac
metrics.jsonl 513f5b744e1f2bc1c62c8ee09f9f10df3702df2be05459269718557b36c37fcd
```

## 5. Limitations

The cached transition is formed from adjacent saved observations and the next
saved joint target. It is not a verified sequence of executed low-level
commands. This offline critic report therefore does not replace simulation
rollouts, recovery-branch evaluation, or a real success-rate measurement.
The report also does not upload the demonstration data, candidate cache, or
model weights to GitHub.

