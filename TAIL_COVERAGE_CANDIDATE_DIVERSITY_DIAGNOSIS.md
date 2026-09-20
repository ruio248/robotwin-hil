# Tail Coverage Candidate-Diversity Diagnosis

**Date:** 2026-09-21 (Asia/Shanghai)  
**Related implementation:** `785115f` (`Use expert next-action coverage objective`)  
**Related run:** `expert_bootstrap`, 10,000 training steps

## Summary

The critic did not produce a visible expert-versus-candidate score gap on the
first formal run. The evidence points primarily to candidate collapse, not to
a failure to execute the coverage term:

1. The four policy calls for one observation produce actions that are almost
   identical to one another and very close to the demonstration action.
2. The current objective has no explicit score margin, so an action-insensitive
   solution `C(o, a) approximately equals C(o)` can fit the data.
3. With `alpha=0.01`, the weighted coverage term is much smaller than the TD
   term on this run.
4. Candidate actions are not executed in this offline dataset, so the critic
   receives no verified counterfactual next state for a candidate action.

This is a diagnosis of the current cache and objective. It is not evidence
that coverage is useless, and it is not a success-rate measurement.

## Evidence from the candidate cache

The statistics below were computed on the 45-episode validation split of the
completed 450-episode cache. The cache contains four candidate actions per
observation.

| Quantity | Value |
|---|---:|
| Raw candidate-to-expert RMS action distance, mean | 0.0022605 |
| Raw candidate-to-expert RMS distance, median | 0.0011241 |
| Raw candidate-to-expert RMS distance, 90th percentile | 0.0038984 |
| Raw candidate-to-expert RMS distance, maximum | 0.0553836 |
| Mean per-observation candidate action standard deviation | 0.0006446 |
| Normalized candidate-to-expert RMS distance, mean | 0.0059613 |
| Normalized candidate-to-expert RMS distance, median | 0.0031093 |
| Normalized candidate-to-expert RMS distance, 90th percentile | 0.0127764 |
| Normalized candidate-to-expert RMS distance, 99th percentile | 0.0445844 |
| Normalized candidate-to-expert RMS distance, maximum | 0.1145197 |

The normalized distance uses the action scale learned from the 405-episode
training split. The raw distance is nonzero, but its typical magnitude is
small; the four repeated policy calls therefore provide little effective
action diversity.

## Evidence from the critic metrics

At step 10,000, the validation metrics were:

```text
validation total loss       = 3.3440776e-05
validation TD loss          = 3.3479542e-05
validation conservative term = -3.8765280e-06
validation expert score     = 40.3334503
validation candidate score  = 40.3334465
```

The candidate-minus-expert score difference is approximately
`-3.88e-06`. Because the loss multiplies the conservative term by
`alpha=0.01`, its actual contribution to the total loss is approximately
`-3.88e-08`, compared with a TD contribution of about `3.35e-05`.

The final training record showed the same pattern:

```text
train TD loss           = 2.3259643e-05
train conservative term = -2.0310283e-05
alpha-weighted term     = -2.0310283e-07
```

There was no non-finite-value failure, no score-limit violation, and no
evidence that the coverage code path was skipped. The small gap is consistent
with the small action differences in the cache.

## Why the current objective can show no gap

The implemented loss is:

\[
\mathcal L_C =
\mathbb E[(C(o_t,a_t^E)-y_t^E)^2]
+ \alpha\mathbb E\left[
\frac{1}{K}\sum_k C(o_t,\tilde a_t^{(k)})
- C(o_t,a_t^E)\right].
\]

If the candidate action is almost the expert action, then the candidate term
is approximately zero. There is also no explicit margin requiring the score
difference to exceed a fixed value. Consequently, the critic can fit the TD
target while using mostly the visual/state features and very little action
sensitivity:

\[
C(o,a) \approx C(o).
\]

This is especially plausible because the current data contains expert
transitions only. Candidate actions are scored offline but are not executed,
so the cache does not contain a verified candidate-specific next observation
or outcome. The critic can therefore learn trajectory progress from the
observation without learning a strong action-dependent consequence.

## What should change before changing the loss

The first intervention should be candidate generation, not longer training:

1. Use genuine stochastic policy inference, if supported by the policy
   adapter, with a controlled sampling temperature or independent inference
   noise.
2. Alternatively, generate bounded joint-space perturbations around the
   policy action, enforcing joint, gripper, velocity, and safety limits.
3. Recompute and report candidate distances before training. Increasing `K`
   alone will not help if every call returns the same deterministic action.
4. Keep the TD target on the next expert action while testing candidate
   diversity; this preserves the current algorithm definition.
5. Only after candidates have meaningful separation should `alpha` be tuned.
   With no output regularizer, a diverse candidate set can cause scores to
   grow, so the existing non-finite and absolute-score guards must remain.

An explicit margin or a different contrastive objective could be a later
experiment, but it would be a new algorithm variant and should not be mixed
with the current formula without a separate comparison.

## Run artifacts and scope

The source run remains on the Ubuntu host at:

```text
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap/config.json
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap/metrics.jsonl
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap/step_0010000.pt
/root/data/my/robotwin-hil/outputs/tail/runs/expert_bootstrap/last.pt
```

The report does not upload the candidate cache, demonstration data, or model
weights. The diagnosis is offline and does not claim a policy success rate or
an online recovery improvement.

