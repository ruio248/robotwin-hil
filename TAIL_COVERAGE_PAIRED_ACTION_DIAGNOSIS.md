# Tail Coverage Paired-Action Diagnosis

**Date:** 2026-09-23 (Asia/Shanghai)
**Checkpoint:** `expert_bootstrap/step_0010000.pt`
**Checkpoint SHA-256:** `0e046ab3f8c8f0a65d8e36f336a88d891d2502639f6f7da155ec670a24771817`

## Executive summary

The critic shows a clear response to synthetic, increasingly large joint
perturbations, but it does **not** show useful ranking behavior on the actual
cached policy candidates at held-out expert states. Grouping those real
candidates by action distance does not reveal a monotonic increase in the
critic's preference for the expert action. This is a partial synthetic
diagnostic pass, not validation of coverage detection on executed policy
behavior, task success, or HIL intervention timing.

## Evaluation setup

- Dataset: the 45-episode held-out validation split from the 450-episode
  demonstration cache; 405 episodes are used for critic training.
- Scored data: 13,388 valid adjacent-frame transitions. No terminal self-loop
  was added.
- At each validation observation, the cache provides four policy candidate
  actions. Candidate and expert actions are compared at the same observation.
- Scores are online `C_phi(s_t, a_t)`. The expert action is the next saved
  absolute-joint target proxy, not a verified low-level executed command.
- Action distance is normalized RMS distance using the checkpoint action
  scales. A positive margin means
  `C_phi(s_t, a_expert) > C_phi(s_t, a_candidate)`.
- The validation episodes are held out from critic gradient updates, but come
  from the same task and data-collection setup. They are not an independent
  task/domain test.

## Actual policy candidates

Across the four cached candidates per observation:

| Metric | Result |
|---|---:|
| Transition-weighted expert win rate | 53.30% |
| Episode-macro expert win rate | 53.38% |
| 95% episode-bootstrap CI for win rate | 52.31%–54.42% |
| Mean expert-minus-candidate score margin, transition-weighted | `+3.90e-6` |
| Median margin | `0` |
| Episode-macro mean margin | `+4.62e-6` |
| 95% episode-bootstrap CI for macro margin | `[-4.73e-6, +1.35e-5]` |
| Mean normalized candidate-to-expert distance, episode-macro | 0.00596 |
| Spearman correlation: distance vs. margin | -0.0062 |

The win rate is only slightly above chance, the average margins are tiny, and
the episode-macro margin interval includes zero. The distance-margin
correlation is effectively absent. The actual cached candidates are also very
close to the expert actions, so this comparison may be under-informative; it
does not establish that the critic fails on larger, realistic policy errors.

## Distance-stratified paired comparison

To check whether more distant real candidates are ranked as worse, validation
states were sorted by their mean normalized distance across the four cached
policy candidates and divided into five equal-count bins. Each bin contains
about 2,678 states and includes all 45 episodes. The win-rate interval below
uses a 10,000-resample episode-cluster bootstrap (seed 42), so transitions
within an episode are not treated as independent samples.

| Candidate-distance quintile | Median normalized distance | Episode-macro expert win rate | 95% episode-bootstrap CI |
|---|---:|---:|---:|
| Lowest 20% | 0.001401 | 53.72% | 52.18%–55.28% |
| 20%–40% | 0.002177 | 55.47% | 53.75%–57.11% |
| 40%–60% | 0.003374 | 53.31% | 51.76%–54.87% |
| 60%–80% | 0.005565 | 54.21% | 52.32%–56.10% |
| Highest 20% | 0.013131 | 51.73% | 49.55%–53.82% |

The highest-distance quintile is not the one with the strongest expert
preference; its interval includes 50%. Thus the real-candidate results do not
support the expected monotonic relationship between action distance and
expert-vs-policy ranking. The highest-distance bin is broad and includes a
small tail of much larger deviations (up to 0.0906), so its median is more
representative than its maximum.

For the highest-distance bin, transition-weighted mean margin is
`-1.16e-6`, while episode-macro mean margin is `+2.61e-5` with a bootstrap
interval spanning zero (`[-1.90e-5, +7.01e-5]`). This weighting disagreement
is a reason not to interpret that bin's mean margin as robust evidence.

## Synthetic perturbation diagnostic

As a separate offline-only diagnostic, random perturbations were applied to
the 12 arm-joint dimensions while grippers were held fixed. Perturbations were
scaled using the checkpoint action scales. No perturbed action was executed,
and these actions were not clipped to robot joint limits.

| Perturbation RMS level | Expert win rate | Episode-macro margin, expert minus perturbation |
|---|---:|---:|
| 0.25 scale | 57.03% | `+0.000263` (95% CI: `[0.000202, 0.000320]`) |
| 0.5 scale | 62.13% | `+0.001098` (95% CI: `[0.000975, 0.001221]`) |
| 1.0 scale | 70.26% | `+0.004204` (95% CI: `[0.003943, 0.004469]`) |
| 2.0 scale | 77.79% | `+0.016523` (95% CI: `[0.015911, 0.017127]`) |

This monotonic synthetic trend is evidence that the critic responds to these
perturbations. They are not labeled unsafe actions or known task failures, so
the result cannot be read as a safety, success-prediction, or recovery metric.

## Judgment and next test

**Judgment:** partial pass on synthetic action perturbations; not validated for
ranking actual policy behavior. The current cache's true policy candidates are
too close to expert actions to be a strong stress test, and the distance bins
show no reliable trend even within that cache.

The next informative test is to score more diverse, realistic policy actions
at states that can be aligned to expert observations, or to collect executed
policy deviations with their resulting observations and outcomes. Candidate
generation should first be checked for meaningful diversity. Any synthetic
negative actions used for further diagnostics should also obey robot joint,
gripper, and velocity limits. Final validation should test real success and
failure trajectories while controlling for task phase and time. The scalar
`C` value is not a success probability and this experiment does not establish
early HIL-warning capability.

## Scope and artifacts

This report uses the cached validation scores and checkpoint identified
above. It does not include or upload the demonstration cache, policy candidate
cache, per-frame score CSVs, videos, or model weights. Results are offline and
do not claim an online policy improvement.
