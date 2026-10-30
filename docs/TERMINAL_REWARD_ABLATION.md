# Recorded-end terminal reward ablation

For the diagnosis of why the earlier B score was near 40, see
[Coverage B 分数约 40 的成因与验证](COVERAGE_SCORE_40_DEBUG_REPORT.md).

This experiment asks what happens to the learned `B(o,a)` curve when a
terminal boundary is introduced. It is an **assumption-based diagnostic**, not
an assertion that LeRobot's final saved frame contains a verified success flag.
The prompt-fixed LeRobot v2.1 metadata contains episode lengths but no `done`
or `success` field. The final *real adjacent* transition of each demonstration
(from saved frame `n-2` to `n-1`) is treated as terminal. The last saved
frame still has no training transition; no self-loop or cross-episode transition
is manufactured. The approximate demonstration action is the next saved joint
state target, as in the existing cache.

For each `R` in {1, 10, 100}, independently train from the same seed:

\[
y_t^{(R)} =
\begin{cases}
R, & t=n-2 \quad \text{(assumed recorded end)},\\
1+0.99\,\operatorname{mean}_{k=1}^4
 B_{\bar\phi}(o_{t+1},\tilde a_{t+1}^{(k)}), & \text{otherwise}.
\end{cases}
\]

The terminal reward **replaces** the ordinary `+1`; it is not an additional
bonus. The conservative term remains the same current-observation candidate
mean minus expert score with `alpha=1.0`. Network, initialization seed 42,
training/validation split (405/45 full episodes), candidate cache, batch 256,
AdamW `1e-4`, target EMA 0.005, and 10,000 optimizer steps are held fixed.
No policy sampling or environment execution is performed during training.

With the constructed rewards, the return along an *executed recorded path* is

\[
G_t^{(R)}=\sum_{j=0}^{n-t-3}0.99^j+0.99^{n-t-2}R.
\]

For `R=100=1/(1-0.99)`, this is identically 100. That arm is an important
control: a flat curve is mathematically expected under this reward convention.
The recorded-path Monte Carlo curve is **not** ground-truth policy `Q`, because
the training TD target uses policy candidates at the next observation, not the
next recorded expert action.

On a host with the completed SFT candidate cache and PyTorch/Matplotlib:

```bash
python RoboTwin/scripts/run_tail_terminal_ablation.py \
  --cache-dir /path/to/outputs/tail/cache/sft_450 \
  --output-dir /path/to/outputs/tail/terminal_reward_ablation_20261006 \
  --device cuda:0
```

The runner refuses existing output directories, trains sequentially, and
records `status.json`, three separate checkpoints and logs, and validation and
training reports. Each report has `b_vs_absolute_step.{png,pdf}` and
`b_vs_progress.{png,pdf}` plus `metrics.json`. The absolute-step plot shows
which saved transition index is being scored; its late bins include fewer
episodes. The normalized-progress plot puts trajectories of different lengths
on a common 0–100% axis. Curves show median `B(o,a^E)` with 10–90% episode
bands, alongside the constructed recorded-path return. The last saved frame
is excluded because it has no verified successor action transition.

Do not interpret a low/high `B` as calibrated success probability or a
verified online recovery score. In particular, a terminal-ablation curve may
change because of an imposed recorded-end boundary, not because the network
has learned a better notion of expert support.

## 2026-10-06 run (10,000 steps per arm)

The controlled run finished on Host `new_server_my_2` using four CPU threads
per run, sequentially, without occupying a shared GPU. It used the complete
cached 450-episode SFT set, with 405 training and 45 critic-validation
episodes. The three training logs record the same initial model SHA-256:
`1b2c40d44241d295eb77dc02191e35068235dc402543049c64ea86e8dddd8308`.
The cache fingerprint was
`e93de3b4b29eccd38a3e4bdd004c3488b87018305b5d6ae3971c1e0cbfe43a05`.

| Terminal reward | Val first-step median B | Val middle-step median B | Val final valid-step median B | Final-step MAE to R | Val TD MSE | Expert minus candidate mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.695 | 40.764 | 1.017 | 0.470 | 0.248 | -0.000516 |
| 10 | 40.764 | 40.755 | 10.587 | 0.607 | 0.172 | -0.000534 |
| 100 | 40.617 | 40.549 | 98.002 | 2.005 | 0.314 | +0.000963 |

The validation plot shows a nearly shared ~40 plateau until roughly 80% of
recorded progress; only near the recorded end do the arms separate toward
their assigned terminal rewards. The training plot has the same qualitative
shape. Thus the boundary is learned locally, but at 10,000 optimizer steps it
has **not** made the whole trajectory match its constructed recorded-path MC
curve. In particular, R=100 has an exactly flat MC reference of 100, while its
first-step predicted median remains 40.617. This run does not establish that
the long-horizon Bellman objective has converged. The tiny action-wise
expert-minus-candidate means also do not support a claim that terminal rewards
improved same-state coverage discrimination.

Run artifacts (not committed to Git):

`/root/data/my/robotwin-hil/outputs/tail/terminal_reward_ablation_20261006/`

- `terminal_{1,10,100}/last.pt` and `metrics.jsonl`
- `report_val/{b_vs_absolute_step,b_vs_progress}.{png,pdf}` and `metrics.json`
- `report_train/{b_vs_absolute_step,b_vs_progress}.{png,pdf}` and `metrics.json`
- `status.json`

The 45 critic-validation episodes were not used to train B, but the underlying
SFT policy had seen the original demonstration set. This is not an independent
policy heldout or a success-rate evaluation.

## Continue to 30,000 steps

The following continuation uses each arm's existing `last.pt` at step 10,000;
the immutable `step_0010000.pt` is retained for the before/after comparison.
Only the reporting interval changes (validation and checkpoint every 2,000
steps, to conserve disk); optimizer, target network, sampler RNG, and all
training hyperparameters are restored from the checkpoint.

```bash
python RoboTwin/scripts/resume_tail_terminal_ablation.py \
  --cache-dir /path/to/outputs/tail/cache/sft_450 \
  --runs-root /path/to/outputs/tail/terminal_reward_ablation_20261006 \
  --device cpu --cpu-threads 4
```

The script refuses to run unless all three source checkpoints are exactly at
step 10,000 and their reward, seed, alpha, and cache fingerprint agree with
the experiment. It writes `resume_to_30000_status.json`, three continuation
logs, and new `report_val_step30000` and `report_train_step30000` directories
without replacing the step-10,000 reports.

## 2026-10-06 continuation result (30,000 steps per arm)

All three jobs resumed from their own step-10,000 checkpoints and finished
successfully at step 30,000. The 10,000-step checkpoints and reports remain
available. On the same 45-episode critic-validation split (13,388 real
adjacent transitions), the diagnostics changed as follows:

| Terminal reward | First B, 10k -> 30k | Middle B, 10k -> 30k | Final B, 10k -> 30k | Recorded-path MC MAE, 10k -> 30k | TD MSE, 10k -> 30k |
|---:|---:|---:|---:|---:|---:|
| 1 | 40.695 -> 78.721 | 40.764 -> 78.193 | 1.017 -> 1.135 | 31.021 -> 5.434 | 0.248 -> 0.177 |
| 10 | 40.764 -> 79.009 | 40.755 -> 78.861 | 10.587 -> 10.086 | 32.706 -> 5.770 | 0.172 -> 0.144 |
| 100 | 40.617 -> 78.893 | 40.549 -> 89.483 | 98.002 -> 100.047 | 53.346 -> 10.801 | 0.314 -> 0.051 |

The training split shows the same shape: at 30k, first-step B is 78.723,
79.005, and 78.889 for rewards 1, 10, and 100 respectively. The previous
~40 plateau was therefore not a converged fixed point. The imposed terminal
condition propagates farther backward by 30k, and the constructed-path MC
MAE drops markedly. However, the early part of all three predictions remains
near 79, whereas the R=100 MC reference is identically 100. The R=1/10
predictions also flatten too early relative to their decreasing MC references.
Do not call this Bellman or value convergence from the lower MAE alone:
policy-next-action bootstrap and expert-path MC measure different quantities.

The action-discrimination gap remains tiny at 30k: on validation,
`mean[B(expert)-mean_k B(candidate_k)]` is -0.00147, -0.00140, and +0.00036
for R=1, 10, and 100. This experiment shows learning of the imposed
recorded-end boundary, not evidence that B meaningfully separates actions
at the same observation or improves online takeover decisions. Recorded end
is still an assumption rather than a verified success/done label.

New artifacts under the same remote run root:

- `terminal_{1,10,100}/step_0030000.pt` (and updated `last.pt`)
- `report_val_step30000/` and `report_train_step30000/`, each with
  `b_vs_progress.{png,pdf}`, `b_vs_absolute_step.{png,pdf}`, and `metrics.json`
- `resume_to_30000_status.json` and the three continuation logs

## Continue to 60,000 steps

After verifying all three step-30,000 checkpoints, resume each arm without
changing the reward convention or training objective:

```bash
python RoboTwin/scripts/resume_tail_terminal_ablation.py \
  --cache-dir /path/to/outputs/tail/cache/sft_450 \
  --runs-root /path/to/outputs/tail/terminal_reward_ablation_20261006 \
  --starting-step 30000 --target-steps 60000 \
  --device cpu --cpu-threads 4
```

The runner checks that `last.pt` matches the immutable
`step_0030000.pt` by SHA-256 before writing. It retains the earlier
checkpoints and reports, writes `resume_to_60000_status.json`, and places the
new curves in `report_val_step60000` and `report_train_step60000`.

## 2026-10-06 continuation result (60,000 steps per arm)

All three resumed jobs and both reports completed successfully. The 60k
validation results use the same 45 episodes and 13,388 real adjacent
transitions as the 10k/30k comparisons:

| Terminal reward | First B, 30k -> 60k | Middle B, 30k -> 60k | Final B, 30k -> 60k | Recorded-path MC MAE, 30k -> 60k | TD MSE, 30k -> 60k |
|---:|---:|---:|---:|---:|---:|
| 1 | 78.721 -> 95.260 | 78.193 -> 77.858 | 1.135 -> 1.336 | 5.434 -> 0.918 | 0.177 -> 0.123 |
| 10 | 79.009 -> 95.651 | 78.861 -> 80.152 | 10.086 -> 10.425 | 5.770 -> 0.964 | 0.144 -> 0.114 |
| 100 | 78.893 -> 99.954 | 89.483 -> 100.268 | 100.047 -> 100.026 | 10.801 -> 0.437 | 0.051 -> 0.002 |

The training split exhibits essentially the same shape (60k first-step
medians 95.251, 95.647, and 99.972). Thus the large ~40 plateau at 10k and
the ~79 early plateau at 30k were primarily incomplete propagation of the
imposed terminal target. By 60k, the predicted expert-path profile closely
tracks the constructed MC profile, including the flat R=100 control. This is
good evidence for convergence toward this **diagnostic** target, not proof
that the policy-next-action Bellman equation has globally converged or that
the critic is a calibrated coverage function.

Indeed, the 60k validation mean same-observation expert-minus-candidate B
gaps are -0.001882, -0.001691, and +0.000029 for R=1, 10, and 100. The
large improvement in time/terminal calibration has not produced meaningful
action separation. Also, the recorded end remains an assumed terminal rather
than a verified task success/done event; no online policy or takeover
evaluation was performed.

The preserved remote artifacts are `terminal_{1,10,100}/step_0060000.pt`,
`report_val_step60000/`, `report_train_step60000/`, and
`resume_to_60000_status.json` under the original run root.
