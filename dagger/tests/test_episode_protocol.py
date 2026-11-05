"""Screen/reset/prompt contracts, no simulator or policy GPU required."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from dagger import runner
from dagger.episode_protocol import prepare_screened_scene, screening_history
from dagger.tests.test_workflow import FakeEnv, run_fake


class Scene:
    def __init__(self, *, plan=True, success=True, legal=True, error=None, joint_error=None):
        self.plan, self.success, self.legal = plan, success, legal
        self.error, self.joint_error = error, joint_error
        self.events = []
        self.setups = 0

    def setup_demo(self, **kw):
        self.events.append(("setup", deepcopy(kw)))
        self.setups += 1
        self.position = "initial"
        self.plan_success = True
        self.episode_metadata = {}
        np.random.seed(kw["seed"])

    def play_once(self):
        self.events.append(("expert", self.position))
        self.position = "expert_terminal"
        if self.error:
            raise self.error
        self.plan_success = self.plan
        return {"info": {"{a}": "left", "{b}": "right", "{c}": "left"}}

    def check_success(self):
        self.events.append(("check_success", self.position))
        return self.success

    def planned_joints_legal(self):
        self.events.append(("planned_joints_legal", self.position))
        if self.joint_error:
            raise self.joint_error
        return self.legal, 2.5 if self.legal else 3.5


class Adapter:
    def initialize_episode(self, env):
        env.events.append(("adapter", env.position))
        env.episode_metadata = {}


def prepare(tmp_path, env, build=None):
    args = dict(task_name="stack_blocks_three", render_freq=10, save_data=False,
                save_video=False, eval_mode=True, need_plan=True, save_freq=15)
    original = deepcopy(args)
    def build_prompt(args, info, split, count):
        env.events.append(("prompt", env.position, info, split, count))
        return "Use left, right, left to stack red, green, blue."
    result = prepare_screened_scene(
        env, args, seed=40001, episode_index=0, adapter=Adapter(),
        build_instruction=build or build_prompt, instruction_pool_size=50,
        close_env=lambda env: env.events.append(("close", env.position)), output_dir=tmp_path,
    )
    assert args == original
    return result


def test_expert_then_reset_then_official_prompt_without_recording(tmp_path):
    env = Scene()
    result = prepare(tmp_path, env)
    assert [event[0] for event in env.events] == [
        "setup", "expert", "check_success", "planned_joints_legal", "close", "setup", "prompt", "adapter"]
    first, replay = env.events[0][1], env.events[5][1]
    assert first["seed"] == replay["seed"] == 40001
    assert first["is_test"] is replay["is_test"] is True
    assert first["save_data"] is replay["save_data"] is False
    assert first["render_freq"] == 0 and replay["render_freq"] == 10
    assert first["save_video"] is False and first["eval_video_save_dir"] is None
    assert env.events[6][1] == "initial"
    assert env.events[6][3:] == ("seen", 50)
    assert result["status"] == "policy_ready"
    assert result["joints_legal"] is True and result["joint_absmax"] == 2.5
    assert env.episode_metadata["initial_scene_screening"]["instruction"] == result["instruction"]
    assert screening_history(tmp_path)[40001] == result
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("plan,success", [(False, False), (False, True), (True, False)])
def test_either_failed_expert_predicate_rejects_without_policy_setup(tmp_path, plan, success):
    env = Scene(plan=plan, success=success)
    assert prepare(tmp_path, env) is None
    assert env.setups == 1
    row = screening_history(tmp_path)[40001]
    assert row["status"] == "expert_rejected"
    assert row["policy_steps"] == 0 and row["counted_as_policy_rollout"] is False
    assert row["joints_legal"] is None and row["joint_absmax"] is None
    assert not any(event[0] == "planned_joints_legal" for event in env.events)


def test_joint_limit_failure_rejects_seed_before_policy_setup(tmp_path):
    env = Scene(legal=False)
    assert prepare(tmp_path, env) is None
    assert env.setups == 1
    row = screening_history(tmp_path)[40001]
    assert row["status"] == "expert_rejected"
    assert row["plan_success"] is True and row["check_success"] is True
    assert row["joints_legal"] is False and row["joint_absmax"] == 3.5
    assert row["policy_steps"] == 0 and row["counted_as_policy_rollout"] is False
    assert [event[0] for event in env.events][-1] == "close"


def test_joint_limit_check_error_stops_screening_as_technical_error(tmp_path):
    env = Scene(joint_error=RuntimeError("joint path unreadable"))
    with pytest.raises(RuntimeError, match="joint path unreadable"):
        prepare(tmp_path, env)
    row = screening_history(tmp_path)[40001]
    assert row["status"] == "expert_error"
    assert row["exception_type"] == "RuntimeError"
    assert env.setups == 1 and env.events[-1][0] == "close"


def test_expert_error_is_durable_and_not_a_policy_failure(tmp_path):
    env = Scene(error=RuntimeError("planner disconnected"))
    with pytest.raises(RuntimeError, match="planner disconnected"):
        prepare(tmp_path, env)
    row = screening_history(tmp_path)[40001]
    assert row["status"] == "expert_error"
    assert "RuntimeError: planner disconnected" in row["traceback"]
    assert env.setups == 1 and env.events[-1][0] == "close"


def test_rebuild_error_does_not_select_prompt_or_enter_policy(tmp_path):
    class BrokenReplay(Scene):
        def setup_demo(self, **kw):
            if self.setups == 1:
                raise RuntimeError("renderer unavailable")
            super().setup_demo(**kw)
    with pytest.raises(RuntimeError, match="renderer unavailable"):
        prepare(tmp_path, BrokenReplay())
    assert screening_history(tmp_path)[40001]["status"] == "policy_setup_error"


def test_instruction_fallback_is_not_silently_used_as_seen(tmp_path):
    with pytest.raises(ValueError, match="fell back"):
        prepare(tmp_path, Scene(), build=lambda args, *unused: args["task_name"])
    assert screening_history(tmp_path)[40001]["status"] == "instruction_error"


def test_exact_official_builder_called_with_expert_info_after_seed_reset(tmp_path):
    # Execute the actual SFT builder, avoiding only its simulator imports.
    source = Path(__file__).resolve().parents[2] / "robotwin_hil/overlay/scripts/eval_policy_xpolicylab_official_trace.py"
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_instruction"]
    seen = ["left/right/left red-green-blue", "red base, green middle, blue top"]
    calls = []
    def descriptions(task, episodes, count):
        calls.append((task, episodes, count))
        return [{"seen": seen, "unseen": ["not selected"]}]
    namespace = {"np": np, "Any": object, "generate_episode_descriptions": descriptions}
    exec(compile(tree, str(source), "exec"), namespace)
    np.random.seed(40001)
    expected = namespace["build_instruction"]({"task_name": "stack_blocks_three"},
        {"info": {"{a}": "left", "{b}": "right", "{c}": "left"}}, "seen", 50)
    result = prepare(tmp_path, Scene(), build=namespace["build_instruction"])
    assert result["instruction"] == expected
    assert calls[0] == calls[1]


def test_runner_rejection_does_not_consume_rollout_or_save_expert_frames(monkeypatch, tmp_path):
    class RejectFirst(FakeEnv):
        def play_once(self):
            info = super().play_once()
            self.plan_success = self.episode_seed != 40000
            return info
    result, env, client, report = run_fake(
        monkeypatch, tmp_path, env_override=RejectFirst(),
        cli_extra=("--episode-protocol", "official-seen", "--max-scene-attempts", "3"),
    )
    assert result == 0
    assert env.expert_checks == 2
    assert [seed for seed, _ in env.setups] == [40000, 40001, 40001]
    assert report["seeds"] == [40001] and report["rollouts"] == 1
    assert report["scene_attempts"] == 2
    assert report["instruction"] is None
    assert report["records"][0]["instruction"] == "official seen seed 40001"
    assert "expert_screen" not in report["records"][0]["control_mask"]
    assert client.chunks == 2


def test_runner_rejects_illegal_expert_joints_without_consuming_rollout(monkeypatch, tmp_path):
    class IllegalFirst(FakeEnv):
        def planned_joints_legal(self):
            return (self.episode_seed != 40000), (3.5 if self.episode_seed == 40000 else 2.5)

    result, env, client, report = run_fake(
        monkeypatch, tmp_path, env_override=IllegalFirst(),
        cli_extra=("--episode-protocol", "official-seen", "--max-scene-attempts", "3"),
    )
    assert result == 0
    assert [seed for seed, _ in env.setups] == [40000, 40001, 40001]
    assert report["seeds"] == [40001] and report["rollouts"] == 1
    assert report["scene_attempts"] == 2 and client.chunks == 2
    rejected = screening_history(tmp_path / "session")[40000]
    assert rejected["status"] == "expert_rejected"
    assert rejected["joints_legal"] is False and rejected["joint_absmax"] == 3.5


def test_candidate_budget_stops_without_policy_calls(monkeypatch, tmp_path):
    class RejectAll(FakeEnv):
        def play_once(self):
            info = super().play_once()
            self.plan_success = False
            return info
    result, env, client, report = run_fake(
        monkeypatch, tmp_path, env_override=RejectAll(),
        cli_extra=("--episode-protocol", "official-seen", "--max-scene-attempts", "2"),
    )
    assert result == 4 and report["scene_budget_exhausted"] is True
    assert report["rollouts"] == 0 and client.chunks == 0
    assert env.expert_checks == 2


def test_no_fixed_prompt_override_and_legacy_tray_defaults_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROBOTWIN_ROOT", tmp_path / "missing")
    with pytest.raises(SystemExit):
        runner.parse_args(["--task", "stack_blocks_three", "--instruction", "fixed"])
    rotate = runner.parse_args(["--task", "rotate_qrcode"])
    assert rotate.episode_protocol == "official-seen" and rotate.instruction is None
    cli = runner.parse_args(["--task", "handover_to_tray"])
    assert cli.episode_protocol == "legacy-fixed" and cli.instruction


def test_seed_pools_are_bounded_and_reserved_seeds_never_screened(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROBOTWIN_ROOT", tmp_path / "missing")
    cli = runner.parse_args(["--task", "stack_blocks_three", "--seed-mode", "random",
                             "--seed-min", "98", "--seed-max", "101", "--rng-seed", "42"])
    assert sorted(runner.build_seed_stream(cli)) == [100, 101]
    cli = runner.parse_args(["--task", "stack_blocks_three", "--seed-mode", "sequential",
                             "--seed-start", "99999", "--max-scene-attempts", "2"])
    stream = runner.build_seed_stream(cli)
    assert next(stream) == 99999
    with pytest.raises(ValueError, match="reserved seed"):
        next(stream)


def test_existing_screening_journal_requires_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROBOTWIN_ROOT", tmp_path / "missing")
    runner.parse_args([])  # Load the actual resume helper.
    prepare(tmp_path, Scene(plan=False))
    with pytest.raises(ValueError, match="use --resume"):
        runner.inspect_collection(tmp_path, target_mode="hil")


def test_resume_skips_previously_rejected_seeds_and_keeps_episode_prompt(monkeypatch, tmp_path):
    class RejectThenAllow(FakeEnv):
        reject = True
        def play_once(self):
            info = super().play_once()
            self.plan_success = not self.reject
            return info
    env = RejectThenAllow()
    result, env, client, report = run_fake(
        monkeypatch, tmp_path, env_override=env,
        cli_extra=("--episode-protocol", "official-seen", "--max-scene-attempts", "2"),
    )
    assert result == 4
    session = tmp_path / "session"
    next(session.glob("session_*.json")).rename(session / "session_20000101_000000.json")
    env.reject = False
    result = runner.main([
        "--runtime-root", str(tmp_path), "--output-dir", str(session), "--resume",
        "--episodes", "1", "--target-saved", "1", "--seed-mode", "sequential",
        "--save-freq", "1", "--auto-save", "true", "--episode-protocol", "official-seen",
        "--max-scene-attempts", "4",
    ])
    assert result == 0
    assert [seed for seed, _ in env.setups] == [40000, 40001, 40002, 40002]
    assert env.instruction == "official seen seed 40002"
    assert sorted(screening_history(session)) == [40000, 40001, 40002]


def test_legacy_report_cannot_be_resumed_as_official_seen(monkeypatch, tmp_path):
    run_fake(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        runner.main([
            "--runtime-root", str(tmp_path), "--output-dir", str(tmp_path / "session"),
            "--resume", "--episode-protocol", "official-seen", "--seed-mode", "sequential",
        ])


def test_prompts_change_between_episodes_but_not_within_one_episode(monkeypatch, tmp_path):
    class OneActionSuccess(FakeEnv):
        def take_action(self, action, **kw):
            super().take_action(action, **kw)
            self.eval_success = True
    class NoKeys:
        def __init__(self, *args, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def poll(self): return None
    uploaded = []
    result, env, client, report = run_fake(
        monkeypatch, tmp_path, keyboard=NoKeys, auto_save="false",
        env_override=OneActionSuccess(),
        helper_overrides={"robotwin_obs_to_xpolicylab":
            lambda obs, **kw: uploaded.append(kw["instruction"]) or obs},
        cli_extra=("--episode-protocol", "official-seen", "--episodes", "2"),
    )
    assert result == 0
    assert [r["instruction"] for r in report["records"]] == uploaded == [
        "official seen seed 40000", "official seen seed 40001"]
