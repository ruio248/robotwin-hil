# Recorded-end terminal reward ablation

This experiment asks what happens to the learned (B(o,a)) curve when a
terminal boundary is introduced. It is an **assumption-based diagnostic**, not
an assertion that LeRobot's final saved frame contains a verified success flag.
The prompt-fixed LeRobot v2.1 metadata contains episode lengths but no `done`
or `success` field. The final *real adjacent* transition of each demonstration
(from saved frame (n-2) to (n-1)) is treated as terminal. The last saved
frame still has no training transition; no self-loop or cross-episode transition
is manufactured. The approximate demonstration action is the next saved joint
state target, as in the existing cache.

For each (R\in\{1,10,100\}), independently train from the same seed:

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
AdamW (10^{-4}), target EMA 0.005, and 10,000 optimizer steps are held fixed.
No policy sampling or environment execution is performed during training.

With the constructed rewards, the return along an *executed recorded path* is

\[
G_t^{(R)}=\sum_{j=0}^{n-t-3}0.99^j+0.99^{n-t-2}R.
\]

For (R=100=1/(1-0.99)), this is identically 100. That arm is an important
control: a flat curve is mathematically expected under this reward convention.
The recorded-path Monte Carlo curve is **not** ground-truth policy (Q), because
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
on a common 0–100% axis. Curves show median (B(o,a^E)) with 10–90% episode
bands, alongside the constructed recorded-path return. The last saved frame
is excluded because it has no verified successor action transition.

Do not interpret a low/high (B) as calibrated success probability or a
verified online recovery score. In particular, a terminal-ablation curve may
change because of an imposed recorded-end boundary, not because the network
has learned a better notion of expert support.
