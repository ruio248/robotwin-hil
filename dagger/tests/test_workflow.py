"""CPU regressions for the migrated collector, without a simulator/server."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dagger import recording, runner, runtime, takeover
from dagger.adapters import get_adapter, registered_tasks
from dagger.adapters.handover_to_tray import HandoverRecoveryMixin


@pytest.fixture(autouse=True)
def source_only_default_runtime(monkeypatch, tmp_path):
    # Unit tests must not depend on a developer's generated runtime snapshot.
    monkeypatch.setattr(runner, "ROBOTWIN_ROOT", tmp_path / "not_assembled")


class FakeEnv:
    def __init__(self):
        self.actions = []
        self.expert_stages = 0
        self.recoveries = 0
        self.closed = False

    def setup_demo(self, now_ep_num, seed, **args):
        self.episode_seed = seed
        self.eval_success = False
        self.plan_success = True
        self.episode_metadata = {"task": "handover_to_tray", "seed": seed}
        self.info = {}
        self.save_freq = args["save_freq"]

    def set_instruction(self, instruction):
        self.instruction = instruction

    def _take_picture(self):
        cache = Path(self.save_dir) / "cache"
        cache.mkdir(exist_ok=True)
        self.folder_path["cache"] = str(cache)
        (cache / f"{self.FRAME_IDX}.pkl").write_bytes(b"test frame")
        self.control_mask.append(self.current_control_source)
        self.FRAME_IDX += 1

    def remove_data_cache(self):
        cache = Path(self.folder_path["cache"])
        for path in cache.iterdir():
            path.unlink()
        cache.rmdir()

    def get_obs(self):
        return {"state": np.zeros(14)}

    def take_action(self, action, **kwargs):
        self.actions.append(int(action[0]))

    def infer_recovery_state(self):
        return dict(branch="receiver_place", stage_id=6, source_holds=False,
                    receiver_holds=True, placed=False, recoverable=True)

    def recovery_stage_iterator(self, start_stage_id=None):
        # This body is lazy: constructing the iterator cannot move the scene.
        self.recoveries += 1
        self.expert_stages += 1
        self._take_picture()
        yield {"phase": "stage", "plan_success": True}
        self.eval_success = True
        yield {"phase": "done", "result": {"success": True, "plan_success": True}}

    def planned_joints_legal(self):
        return True, 0.0

    def success_metrics(self):
        return {"fake_metric": self.eval_success}

    def check_success(self):
        return self.eval_success


class FakeClient:
    def __init__(self):
        self.chunks = 0
        self.resets = 0
        self.closed = False

    def call(self, func_name, **kwargs):
        if func_name == "get_action":
            self.chunks += 1
            return np.full((3, 14), self.chunks)
        assert func_name == "update_obs"


class FakeKeyboard:
    def __init__(self, env, key_map=None):
        self.env = env
        self.sent = set()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read_line(self, prompt):
        return "y"

    def poll(self):
        env = self.env
        if len(env.actions) == 1 and "first_i" not in self.sent:
            self.sent.add("first_i")
            return "intervene"
        if env.expert_stages == 1 and "r" not in self.sent:
            self.sent.add("r")
            return "handback"
        if len(env.actions) == 2 and "second_i" not in self.sent:
            self.sent.add("second_i")
            return "intervene"
        return None


def run_fake(monkeypatch, tmp_path, keyboard=FakeKeyboard):
    env, client = FakeEnv(), FakeClient()
    helpers = dict(
        class_decorator=lambda task: env,
        build_policy_client=lambda args: client,
        close_policy_client=lambda c: setattr(c, "closed", True),
        prepare_policy_case=lambda *args: None,
        notify_trial_end=lambda *args: None,
        reset_policy=lambda c: setattr(c, "resets", c.resets + 1),
        is_episode_end=lambda e: e.eval_success,
        normalize_action_chunk=lambda chunk: chunk,
        robotwin_obs_to_xpolicylab=lambda obs, **kwargs: obs,
        xpolicylab_action_to_robotwin=lambda action, **kwargs: (action, "qpos"),
        safe_close_env=lambda e, **kwargs: setattr(e, "closed", True),
    )
    for name, value in helpers.items():
        monkeypatch.setattr(runner, name, value, raising=False)
    monkeypatch.setattr(runner, "_load_runtime_helpers", lambda: None)
    monkeypatch.setattr(runner, "activate_runtime", lambda path: path)
    monkeypatch.setattr(runner, "ROBOTWIN_ROOT", tmp_path)
    monkeypatch.setattr(runner, "build_runtime_args", lambda cli: (
        {"left_arm_dim": 6, "right_arm_dim": 6}, {"save_freq": cli.save_freq},
    ))
    monkeypatch.setattr(runner, "HumanInterventionInput", keyboard)
    for name in ("render_initial_frame", "setup_viewer_diagnostics", "install_viewer_frame_limit"):
        monkeypatch.setattr(runner, name, lambda env: None)
    result = runner.main([
        "--runtime-root", str(tmp_path), "--output-dir", str(tmp_path / "session"),
        "--episodes", "1", "--target-saved", "1", "--seed-mode", "sequential",
        "--save-freq", "1", "--auto-save", "true",
    ])
    reports = list((tmp_path / "session").glob("session_*.json"))
    assert len(reports) == 1
    return result, env, client, json.loads(reports[0].read_text())


def test_takeover_handback_and_second_takeover_discard_old_chunk(monkeypatch, tmp_path):
    code, env, client, report = run_fake(monkeypatch, tmp_path)
    assert code == 0
    # Three actions were returned each time, but only the first of each was run.
    assert env.actions == [1, 2]
    assert client.chunks == 2
    assert client.resets == 2  # initial reset and handback
    assert env.recoveries == 2
    assert env.closed and client.closed
    record = report["records"][0]
    assert record["intervention_count"] == 2
    assert record["interventions"][0]["handback_step"] is not None
    assert record["interventions"][1]["expert_done"]
    assert record["control_mask"] == ["policy", "policy", "hil", "policy", "hil"]
    assert [segment["source"] for segment in record["segments"]] == ["policy", "hil", "policy", "hil"]
    raw = Path(record["raw_dir"])
    assert len(list((raw / "frames").glob("*.pkl"))) == 5
    assert json.loads((raw / "episode.json").read_text())["record"]["interventions"] == record["interventions"]
    assert report["task"] == "handover_to_tray"


@pytest.mark.parametrize("event,code", [("abort", 0), ("quit", 130)])
def test_abort_and_quit_never_save_partial_episode(monkeypatch, tmp_path, event, code):
    class Keyboard(FakeKeyboard):
        def poll(self):
            return event if self.env.actions else None
    actual, env, client, report = run_fake(monkeypatch, tmp_path, Keyboard)
    assert actual == code
    assert env.actions == [1]
    assert client.closed and env.closed
    assert not list((tmp_path / "session").glob("raw/episode_*"))
    assert report["saved_episodes"] == 0


@pytest.mark.parametrize("answers,chosen", [(["y"], None), (["6"], 6), (["invalid", "4"], 4), (["q"], "cancel")])
def test_stage_confirmation_does_not_execute_expert(answers, chosen):
    env = FakeEnv()
    keyboard = SimpleNamespace(read_line=lambda prompt: answers.pop(0))
    recovery, stage = takeover.choose_recovery_stage(env, keyboard, get_adapter("handover_to_tray"))
    assert env.recoveries == 0
    if chosen == "cancel":
        assert recovery is None and stage is None
    else:
        assert stage == chosen
        assert recovery is not None


def test_raw_save_refuses_overwrite_and_keeps_cache(tmp_path):
    env = FakeEnv()
    env.setup_demo(0, 40000, save_freq=15)
    env.save_dir = str(tmp_path)
    env.FRAME_IDX, env.folder_path, env.control_mask = 0, {}, []
    env.current_control_source = "hil"
    env._take_picture()
    env._take_picture()
    (tmp_path / "raw/episode_0000000").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        recording.finalize_episode(env, tmp_path, 0, "test", save=True,
                                   supervisor_label="success", segments=[])
    assert (tmp_path / "cache/0.pkl").exists()


def test_compatibility_entry_reexports_same_implementation():
    path = Path(__file__).resolve().parents[2] / "robotwin_hil/overlay/scripts/hg_dagger_handover.py"
    spec = importlib.util.spec_from_file_location("source_handover_compat", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    assert legacy.main is runner.main
    assert legacy.finalize_episode is recording.finalize_episode
    assert legacy.HumanInterventionInput is takeover.HumanInterventionInput
    cli = legacy.parse_args([])
    assert (cli.task, cli.task_config, cli.frequency, cli.policy_name) == (
        "handover_to_tray", "handover_to_tray_v2_promptfix", 30, "Pi_05",
    )
    assert cli.ckpt_name == "v2_promptfix_9999"
    assert cli.es_horizon == 10


def test_unmigrated_task_is_not_silently_routed_to_handover():
    assert registered_tasks() == ("handover_to_tray",)
    with pytest.raises(ValueError, match="No shared-runner adapter"):
        get_adapter("stack_blocks_three")
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", "handover_block"])


def test_explicit_runtime_sets_cwd_and_missing_runtime_fails_before_chdir(monkeypatch, tmp_path):
    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setattr(sys, "path", list(sys.path))
        patch.setattr(runtime, "configure_imports", lambda root: None)
        missing = tmp_path / "missing"
        with pytest.raises(RuntimeError, match="Incomplete HIL runtime"):
            runtime.activate_runtime(missing)
        assert Path.cwd() == tmp_path
        built = tmp_path / "runtime"
        for name in (
            ".robotwin_hil_runtime.json", "envs/_base_task.py", "scripts/eval_policy_xpolicylab.py",
            "scripts/coverage_sampling/options.py", "scripts/hg_dagger_resume.py",
            "scripts/one_stage_dagger.py", "scripts/policy_rollout_archive.py",
        ):
            path = built / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        (built / "XPolicyLab").mkdir()
        patch.setenv("ROBOTWIN_ROOT", str(built))
        assert runtime.activate_runtime(built) == built
        assert Path.cwd() == built


def test_package_help_does_not_import_simulator(tmp_path):
    script = "from dagger.runner import parse_args; import sys; " \
             "assert 'sapien' not in sys.modules; parse_args(['--help'])"
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                            env={**os.environ, "ROBOTWIN_ROOT": str(tmp_path / "not_assembled")})
    assert result.returncode == 0, result.stderr
    assert "--task" in result.stdout


def test_relative_cli_paths_resolve_before_runtime_chdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cli = runner.parse_args([
        "--runtime-root", "generated/RoboTwin", "--output-dir", "sessions/new",
        "--paired-baseline-report", "reports/baseline.json", "--es-critic", "weights/d.pt",
        "--es-calibration", "weights/calibration.json", "--es-encoder-weights", "weights/r3m.pt",
        "--es-log-dir", "scores",
    ])
    assert cli.runtime_root == tmp_path / "generated/RoboTwin"
    assert cli.output_dir == tmp_path / "sessions/new"
    assert cli.paired_baseline_report == tmp_path / "reports/baseline.json"
    assert cli.es_critic == str(tmp_path / "weights/d.pt")
    assert cli.es_calibration == str(tmp_path / "weights/calibration.json")
    assert cli.es_encoder_weights == str(tmp_path / "weights/r3m.pt")
    assert cli.es_log_dir == str(tmp_path / "scores")
    cli = runner.parse_args(["--runtime-root", "generated/RoboTwin"])
    assert cli.output_dir == tmp_path / "generated/RoboTwin/data/handover_to_tray_hg_dagger_r1/handover_to_tray/aloha_agilex"


def test_selected_runtime_helpers_are_not_default_or_cached_snapshot(tmp_path):
    source = Path(__file__).resolve().parents[2] / "robotwin_hil/overlay/scripts"
    for name in ("selected", "other"):
        scripts = tmp_path / name / "scripts"
        scripts.mkdir(parents=True)
        shutil.copytree(source / "coverage_sampling", scripts / "coverage_sampling",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for filename in ("hg_dagger_resume.py", "one_stage_dagger.py", "policy_rollout_archive.py"):
            shutil.copy2(source / filename, scripts / filename)
    script = '''
from pathlib import Path
import sys
from dagger.runner import parse_args
assert 'coverage_sampling' not in sys.modules
selected = Path(sys.argv[1])
parse_args(['--runtime-root', str(selected)])
for name in ('coverage_sampling', 'coverage_sampling.options', 'hg_dagger_resume',
             'one_stage_dagger', 'policy_rollout_archive'):
    assert Path(sys.modules[name].__file__).is_relative_to(selected)
assert 'sapien' not in sys.modules
try:
    parse_args(['--runtime-root', sys.argv[2]])
except RuntimeError as error:
    assert 'already loaded' in str(error)
else:
    raise AssertionError('cached runtime modules must not be reused across roots')
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "selected"),
                             str(tmp_path / "other")], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    preload = '''
import sys
sys.path.insert(0, sys.argv[1])
import tail_value
from dagger.runner import parse_args
try:
    parse_args(['--runtime-root', sys.argv[2]])
except RuntimeError as error:
    assert 'tail_value' in str(error) and 'already loaded' in str(error)
else:
    raise AssertionError('cached critic/encoder dependency must be rejected')
'''
    result = subprocess.run([sys.executable, "-c", preload, str(source),
                             str(tmp_path / "selected")], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_moved_recovery_preserves_unrecoverable_state_guard():
    class Env(HandoverRecoveryMixin):
        def infer_recovery_state(self):
            return {"recoverable": False, "branch": "restart_source_grasp"}
    env = Env()
    result = env.recover_from_current_state()
    assert result["success"] is False
    assert result["reason"] == "bar_outside_recovery_workspace"
    events = list(env.recovery_stage_iterator())
    assert len(events) == 1
    assert events[0]["phase"] == "done"
    assert events[0]["plan_success"] is False


def test_moved_recovery_at_target_only_releases_grippers():
    class Env(HandoverRecoveryMixin):
        def __init__(self):
            self.source_arm_tag, self.receiver_arm_tag = "left", "right"
            self.episode_metadata, self.info = {}, {}
            self.stage_snapshots, self.executed = [], []
        def infer_recovery_state(self):
            return dict(recoverable=True, branch="release_at_tray", placed=True,
                        source_holds=False, receiver_holds=False)
        def open_gripper(self, side):
            return ("open", side)
        def _run_stage(self, stage, *actions):
            self.executed.append((stage, actions))
            return True
        def check_success(self):
            return True
    env = Env()
    events = list(get_adapter("handover_to_tray").recovery_stage_iterator(env))
    assert env.executed == [(6, (("open", "left"), ("open", "right")))]
    assert events[-1]["result"]["success"] is True
    assert events[-1]["result"]["executed_stage_ids"] == [6]
