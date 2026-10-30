# Baseline 100-Episode Action-Diversity Diagnosis

**Date:** 2026-09-23 (Asia/Shanghai)

**Policy:** SFT baseline checkpoint `9999`

**Evaluation set:** `sft_policy_eval_100`, seeds 31000–31099

**Result:** 31 successful episodes, 69 failed episodes

## Question

Do repeated stochastic policy samples at the same saved observation diverge
more on episodes that ultimately fail, and is the divergence concentrated
near the end of a trajectory?

This is an offline action-diversity measurement. It does not execute sampled
actions in the simulator and does not measure whether an alternative action
would have changed the episode outcome.

## Method

- Loaded the SFT baseline `9999` checkpoint and its configured normalization
  statistics on the Ubuntu evaluation host.
- Read each episode's saved HDF5 observations: three RGB camera views and the
  14-dimensional joint state.
- Evaluated four fixed observations per episode: approximately 10%, 50%, and
  90% of the saved frames, plus the common saved-frame index 10.
- Drew 8 independent action chunks for each fixed observation, for 400
  observations and 3,200 chunks total. Each chunk has shape `[10, 14]`.
- For every observation, compared all 28 pairs of chunks. The reported arm
  distance is the RMS difference over the 12 arm-joint dimensions, excluding
  both gripper dimensions. The first-action metric uses the first row of each
  chunk; the full-chunk metric uses all ten rows. Distances are in radians.
- “Early”, “middle”, and “late” mean relative positions in the saved episode,
  not semantic task stages. The common frame-10 comparison reduces episode
  length mismatch but does not guarantee identical task phases.

No observation had eight bit-identical action chunks.

## Results

Values below average each observation's mean pairwise distance, then average
over episodes within the outcome group. Lower and upper variation across
observations is shown by the median where useful; it is not a confidence
interval.

| Observation position | Outcome | Episodes | First arm action RMS | Full 10-step arm chunk RMS | Chunk diversity ratio, failure/success |
|---|---|---:|---:|---:|---:|
| Early, about 10% | Failed | 69 | 0.001410 | 0.009940 | 3.95× |
| Early, about 10% | Successful | 31 | 0.000818 | 0.002517 | — |
| Middle, about 50% | Failed | 69 | 0.002265 | 0.010207 | 2.22× |
| Middle, about 50% | Successful | 31 | 0.000989 | 0.004609 | — |
| Late, about 90% | Failed | 69 | 0.002286 | 0.010603 | 1.54× |
| Late, about 90% | Successful | 31 | 0.001756 | 0.006898 | — |
| Common saved frame 10 | Failed | 69 | 0.000925 | 0.003615 | 3.42× |
| Common saved frame 10 | Successful | 31 | 0.000589 | 0.001057 | — |

The episode-level overlap is substantial. For the full-chunk metric, the
probability that a randomly selected failed episode has greater diversity
than a randomly selected successful episode (AUC) was:

| Position | AUC | Failed episodes above the successful-group 90th percentile |
|---|---:|---:|
| Early | 0.725 | 39 / 69 |
| Middle | 0.673 | 43 / 69 |
| Late | 0.574 | 28 / 69 |
| Common saved frame 10 | 0.704 | 25 / 69 |

Within-episode comparisons also do not show a failure-specific late spike:
full-chunk diversity was higher late than early in 40 of 69 failed episodes
and 26 of 31 successful episodes. The median late-minus-early change was
0.001576 rad for failures and 0.003838 rad for successes.

The eight episodes with the largest late-position chunk diversity in this
measurement were all failures (episode indices 72, 10, 46, 2, 54, 84, 68,
and 32). This is a tail observation, while the group-level late AUC is only
0.574; it should not be treated as a reliable failure detector.

## Interpretation

Repeated samples from failed episodes are more diverse on average, especially
when comparing the full action chunk. The current first action remains much
closer across samples; most of the measured spread appears later in the
10-step chunk.

The difference is strongest in early and common-frame comparisons, not at the
late relative position. Successful episodes also often become more diverse
near their end. Therefore this experiment does not support the claim that
sampling diversity is specifically a final-stage failure signal.

Action diversity is also not a calibrated measure of uncertainty or task
failure. A diverse set can contain several good actions, several bad actions,
or a few outliers. The measurement does not associate any sampled chunk with
an executed outcome.

## Limitations and next use

- HDF5 observations are saved sparsely (every 15 control steps in this
  evaluation setup). The experiment covers those observations, not every
  control step.
- The saved `stage_id` field is zero throughout these records, so semantic
  phase matching was unavailable. Relative trajectory positions and frame 10
  are only approximate controls.
- These files do not provide a complete simulator snapshot for restoring
  contacts, object velocities, and other hidden simulator state. This is not
  a counterfactual rollout experiment.
- The evaluation outcomes are useful anchors for a later study, but the 100
  episodes must be split by episode or seed before training and validation.
  Reusing them for fitting means they can no longer serve as an untouched
  final test set.

The most useful follow-up is to log policy outputs and actually applied
actions at each saved decision point, annotate failure onset/task phase, and
then compare success and failure states at matched phases. Branch rollouts
from restored simulator states are needed to test whether sampled action
diversity causes different outcomes.

This report contains aggregate measurements only. It does not include the
evaluation videos, HDF5 data, model weights, or other per-episode observations.
