# Fixed-window coverage-guided enhanced sampling

Based on `tail-coverage-value-v2` (`74af6f1`). Entry point:
`RoboTwin/scripts/eval_policy_xpolicylab.py`, opt in with `--es-mode enhanced`.
Default `off` preserves the existing single-policy-call evaluation path.

## Algorithm and experiment contract

Choose an inclusive **zero-based policy decision window** `[t_on, t_off]`
before running either experiment. One decision generates and executes one
chunk; these indices are neither physics steps, saved frames, nor seconds.
The initial window must be chosen from development trajectories, not inferred
from coverage or adjusted separately for the two experiment arms.

Outside the window, draw and execute one ordinary policy chunk. Inside:

1. Send the current observation to Pi0.5 once, then call `get_action` N times
   without resetting policy noise between calls. The supported
   `Pi_05_RobotTwin` adapter calls Flow inference on every call. This first
   version uses sequential inference; it does not claim batched inference.
2. Capture the live simulator state. Starting from that state for each
   candidate, execute its full H-step chunk through the existing
   `Base_Task.take_action(qpos)` and get fresh observations at every step.
3. Compute `c[i,h] = C(o[i,h], a[i,h])` **before** executing that step and
   `J[i] = min_h c[i,h]`. Thus h=0 uses the current observation, and h>0 uses
   genuinely simulated future observations. Restore the live state after
   every branch, including exception paths.
4. Enhanced: `w = softmax(-beta * J)`, sample `I ~ Categorical(w)`, then
   execute candidate I on the restored live scene. Vanilla: perform the same
   candidate inference and branch scoring, but select uniformly. No uniform
   component is mixed into Enhanced weights; beta=0 naturally gives uniform
   selection. `J-min(J)` is used solely for numerical stability.

This is finite-N self-normalized importance resampling, approximating
`q_beta(A|s) ∝ pi(A|s) exp(-beta J(A;s))`. It is not exact sampling from
q_beta at finite N. Since proposals already come from pi, do not multiply
the weights by a second policy-likelihood factor.

The two arms use identical policy/critic weights, task config, initial scene
seed list, window, N, H and execution code. Identical scene seeds do not imply
identical remote Flow RNG streams or states after actions diverge. All actual
candidate arrays are logged; the local resampling RNG is independently seeded
per episode. Outside the window, policy inference is called once in both arms.

## Action, observation, and checkpoint conventions

- Candidates are physical **absolute 14D joint/gripper targets**, ordered as
  `[left arm 6, left gripper, right arm 6, right gripper]`. There is no extra
  delta conversion or policy normalization. Runtime gripper handling remains
  in the existing executor.
- The critic uses the same three-view frozen ResNet18 ImageNet V1 encoder,
  RGB aspect-preserving resize/padding and normalization as tail training.
  Its state is `joint_action.vector`, the recorded drive-target representation
  used by training, not a newly substituted EEF state.
- Supply the critic checkpoint and local encoder weight file explicitly.
  Full training and inference-only critic payloads are accepted. Record their
  SHA-256, training step, alpha and bootstrap type in every episode log.
  The tested current alpha=0.1 inference artifact is step 10000,
  `bootstrap=policy_next_action_mean`, SHA-256
  `0c1dbf97ca0ece2c6e8154c59f5a43a9a9616f38f8dd87b867846b26c586f670`.
- The expected full chunk length defaults to H=10. A differently shaped
  response fails explicitly; it is not silently truncated or reinterpreted
  as N candidates. Candidate count defaults to N=4 and beta to 10; beta is a
  configurable sampling parameter, distinct from the critic-training alpha.

## Simulator isolation and limits

V1 supports serial `handover_to_tray` absolute-joint evaluation with SAPIEN 3
CPU PhysX state pack/unpack. It restores physics state, drive position and
velocity targets, generalized forces/accelerations, Python gripper state,
step counters, success flags, cached observations and Python/NumPy RNG.
Lookahead disables evaluation video writes, dataset saves and viewer refresh.
It never updates policy observations with hypothetical frames.

At the first active decision of every episode, replay candidate zero after
the other branches and compare endpoint physics and the entire score curve.
The default absolute tolerance is 1e-4. A mismatch aborts evaluation with a
diagnostic, rather than allowing unverified restoration to affect the trial.
This adds one extra branch to that decision in both experiment arms. CPU
pack/unpack is not asserted to capture every hidden PhysX solver cache;
the replay guard and predicted-versus-executed logs make drift observable.

Both success and the normal episode action budget terminate a branch early.
Its minimum uses only actually reached pre-action observations; no terminal
self-loop or invented final score is added. `scored_steps` is logged because
terminal branches can have unequal lengths. A single anomalous low score can
dominate this minimum objective; inspect full curves alongside its minimum.

Randomly changing lights and other tasks/action spaces are rejected. The
existing action executor, evaluation frequency setting, task termination and
policy adapter are retained. The sampler does not create an automatic HIL
trigger or change the DAgger recorder. The first experiment measures deviation
under frozen-critic scores; additional human-intervention collection is a
separate experiment.

## Run on Ubuntu

Use an isolated checkout of this branch with access to the installed RoboTwin
assets and existing Python environments. Do not switch an active evaluation
checkout underneath its process. Run the following when a simulator/policy
service is available for this experiment.

Start a dedicated Pi0.5 policy service using the existing baseline config on
an unused port, e.g. 18311 (run from the isolated checkout's `RoboTwin/`):

```bash
POLICY_PY=/hdd/robotwin-hil/RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/openpi/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
  "$POLICY_PY" XPolicyLab/setup_policy_server.py \
  --config_path ../local_serving/pi05_robotwin_handover_to_tray_v2_promptfix_9999.yml \
  --host 127.0.0.1 --port 18311
```

The config contains the existing baseline checkpoint's absolute path. Check
that path and its `repo_id` norm-stats asset before starting the service.
Both experiment arms must point at that same policy. For alternate
checkpoints use their own serving config and record the actual checkpoint
path in `ckpt_setting` below.

From the same `RoboTwin/` directory, set local artifact paths and a **new**
report root. The window 20..29 below is an example, not a measured handover
window; replace it with the development-set window chosen for your experiment.

```bash
EVAL_PY=/hdd/miniconda3/envs/robotwin_hil/bin/python
CRITIC=/path/to/alpha_0p1.pt
ENCODER=/home/ruio/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
SEEDS=/path/to/fixed_seed_manifest.json
POLICY_CKPT=/path/to/the/served/baseline/9999
REPORT_ROOT=../outputs/enhanced_sampling/paired_run_001

for mode in vanilla enhanced; do
  "$EVAL_PY" scripts/eval_policy_xpolicylab.py \
    --bench_name RobotTwin --task_name handover_to_tray \
    --env_cfg_type aloha_agilex --policy_name Pi_05_RobotTwin \
    --host 127.0.0.1 --port 18311 --protocol ws --eval_batch false \
    --root_dir "$PWD" --device_id 0 --seed 0 \
    --task_config handover_to_tray_v2_promptfix --expert_check false --frequency 30 \
    --seed_manifest "$SEEDS" --seed_split test \
    --additional_info "ckpt_setting=$POLICY_CKPT,action_type=joint" \
    --es-mode "$mode" --es-window-start 20 --es-window-end 29 \
    --es-num-candidates 4 --es-horizon 10 --es-beta 10 --es-seed 42 \
    --es-critic "$CRITIC" --es-encoder-weights "$ENCODER" --es-device cpu \
    --es-log-dir "$REPORT_ROOT/$mode" || break
done

"$EVAL_PY" scripts/summarize_enhanced_sampling.py \
  --vanilla-dir "$REPORT_ROOT/vanilla" --enhanced-dir "$REPORT_ROOT/enhanced" \
  --output "$REPORT_ROOT/comparison.json"
```

Use the existing fixed seed manifest format. A seed manifest determines the
episode count in this evaluator (it takes precedence over `--test_num`).
For a short smoke run, provide a manifest with just the selected smoke seeds.
The `frequency=30` field is retained exactly as in the baseline; it is sent
to the policy adapter, and does not change TOPP physics execution into a new
fixed-duration controller. The actual executor remains `take_action`.

## Recorded metrics and interpretation

Each `episode_*.jsonl` records:

- Artifact hashes/config, task seed, instruction and executor.
- Each decision's window flag, all candidate action arrays, all simulated
  coverage curves, J, categorical probabilities, selected index, effective
  sample size, inference/selection timing and replay check errors.
- Every actual executed action's pre-action coverage, predicted coverage,
  prediction error, action index, rollout step and success flag. Actual
  scores are recomputed from actual observations, not copied from lookahead.
- Episode completion/error, whether the window was reached, active-window
  executed coverage minimum/mean and task outcome.

To report low-coverage trajectory hit rates, pass the same preselected
`--es-low-threshold` to both arms. Without a calibrated threshold these fields
are null. The comparison script rejects unfinished/error logs, unmatched
episode seed sets, critics, policy identifiers, instructions or configs. It
reports all outcomes plus paired coverage differences among episodes that
reach the window in both arms; missing-window episodes remain visible.

A lower selected J is encouraged by construction and does not on its own
prove useful deviation. Inspect actual execution curves and task outcomes.
The comparison does not infer HIL labels or report unobserved interventions.
There is no online performance claim from unit tests or synthetic-image smoke.

## Verification

```bash
python -m unittest discover -s RoboTwin/scripts/tests -p 'test_coverage_sampling*.py' -v
python -m unittest discover -s RoboTwin/scripts/tests -p 'test_tail_value.py' -v
```

The SAPIEN test skips when SAPIEN is unavailable. It constructs a real CPU
articulation pushing a box, compares branch replay under contact, and checks
that the selected candidate can be executed from the restored scene with the
predicted trajectory. It needs no renderer, robot assets or policy service.

Validation completed on 2026-09-24:

- Local Python 3.11: 45 tests passed (18 new sampler/integration/report tests
  and all 27 existing tail tests). The optional SAPIEN test was skipped on
  macOS and run separately on Ubuntu.
- Ubuntu Python 3.10 / SAPIEN 3.0.0b1: real CPU articulation/contact snapshot,
  replay and selected-execution test passed without a renderer or policy call.
- Current alpha=0.1 / step10000 critic loaded successfully. On synthetic RGB,
  `OnlineCoverage` equaled a direct call using the training preprocessing and
  model exactly: both returned `40.34922790527344`. This is an interface
  check, not an evaluated task trajectory or a performance result.
- Source compilation and `git diff --check` passed. The existing Ubuntu
  evaluation process and its policy service were not modified or restarted.

A complete Pi0.5 + visual handover A/B episode has not yet been run with this
branch. The first online smoke should use a short fixed seed list, inspect
replay checks and execution/prediction errors, and then expand the experiment.
