"""CPU contracts for the three adapters; NOT simulator rescue-rate evidence."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dagger.adapters import get_adapter
from dagger import runner
from dagger.tests.test_workflow import (
    run_fake, WireFormatClient, action_wire_helpers, source_only_default_runtime,
)

TASKS = ("rotate_qrcode", "place_empty_cup", "handover_block")


def test_initialize_episode_creates_missing_metadata_for_plain_scene():
    env = Env("rotate_qrcode")
    del env.episode_metadata
    env.adapter.initialize_episode(env)
    assert env.episode_metadata["task"] == "rotate_qrcode"
    assert env.episode_metadata["seed"] == 40000
    assert env.episode_metadata["live_recovery_events"] == []


class Entity:
    def __init__(self, uid, name="box"):
        self.uid, self.name = uid, name
        self.body = NS(linear_velocity=np.zeros(3), angular_velocity=np.zeros(3))

    def get_id(self): return self.uid
    def get_components(self): return [self.body]


class Actor:
    def __init__(self, uid, point, q=(1, 0, 0, 0), offset=0.):
        self.actor = Entity(uid)
        self.p, self.q, self.offset = np.array(point, dtype=float), np.array(q, dtype=float), offset

    def get_pose(self): return NS(p=self.p.copy(), q=self.q.copy())
    def get_functional_point(self, index, mode=None):
        p = self.p + [0, 0, self.offset]
        return NS(p=p, q=self.q.copy()) if mode == "pose" else [*p, *self.q]


class Env:
    def __init__(self, task):
        self.adapter = get_adapter(task)
        self.kind = task
        q = (.5, .5, .5, .5) if task == "place_empty_cup" else (1, 0, 0, 0)
        self.object = Actor(1, [-.2, 0, .84 if task == "handover_block" else .76], q,
                            -.1 if task == "handover_block" else 0)
        setattr(self, self.adapter.actor_name, self.object)
        self.target = Actor(2, [.2, .15, .75])
        self.coaster = self.target_box = self.target
        self.target_pose = [-.2, -.15, .74, 1, 0, 0, 0]
        self.block_middle_pose = [0, 0, .9, 0, 1, 0, 0]
        self.table = Entity(3, "table")
        self.fingers = {s: Entity(i, s + "_finger") for i, s in enumerate(("left", "right"), 4)}
        self.grip = {s: 1. for s in self.fingers}
        self.held = set()
        self.on_table, self.on_target = True, False
        self.robot = NS(**{s + "_gripper": [(NS(child_link=NS(entity=f)),)] for s, f in self.fingers.items()})
        self.robot.get_left_arm_jointState = lambda: [0.] * 6 + [self.grip["left"]]
        self.robot.get_right_arm_jointState = lambda: [0.] * 6 + [self.grip["right"]]
        self.scene = NS(get_contacts=self.contacts, get_timestep=lambda: 1/250)
        self.table_z_bias = 0.
        self.current_control_source = "policy"
        self.plan_success = True
        self.episode_seed = 40000
        self.episode_metadata = {"existing": "preserved"}
        self.moves, self.move_batches, self.hold_calls = [], [], 0
        self.fail = self.construction_error = None
        self.drop_on_lift = self.miss_grasp = False
        self.illegal = False
        self.hold_drift = 0.
        self.adapter.initialize_episode(self)

    def contacts(self):
        pairs = [(self.object.actor, self.fingers[s]) for s in self.held]
        if self.on_table: pairs.append((self.object.actor, self.table))
        if self.on_target: pairs.append((self.object.actor, self.target.actor))
        return [NS(bodies=[NS(entity=a), NS(entity=b)]) for a, b in pairs]

    def is_left_gripper_open(self): return self.grip["left"] > .8
    def is_right_gripper_open(self): return self.grip["right"] > .8
    def is_left_gripper_close(self): return self.grip["left"] < .2
    def is_right_gripper_close(self): return self.grip["right"] < .2
    def check_success(self):
        return bool(self.adapter.geometry(self)["at_target"] and all(v > .8 for v in self.grip.values()))
    def planned_joints_legal(self): return not self.illegal, 1.
    def open_gripper(self, arm): return ("open", str(arm), {})
    def close_gripper(self, arm, pos): return ("close", str(arm), {"pos": pos})
    def back_to_origin(self, arm): return ("park", str(arm), {})
    def move_by_displacement(self, arm_tag, **kw): return ("lift", str(arm_tag), kw)
    def grasp_actor(self, actor, arm_tag, **kw):
        if self.construction_error == "grasp": raise ValueError("no_feasible_grasp_pose")
        return ("grasp", str(arm_tag), kw)
    def place_actor(self, actor, arm_tag, target_pose, **kw):
        if self.construction_error == "place": raise ValueError("no_feasible_place_pose")
        p = list(target_pose.p) if hasattr(target_pose, "p") else list(target_pose[:3])
        return ("place", str(arm_tag), {"target": p, **kw})

    def move(self, *commands):
        assert self.current_control_source == "hil"
        self.move_batches.append(tuple((kind, side) for kind, side, _ in commands))
        for kind, side, kw in commands:
            self.moves.append((kind, side, kw))
            if self.fail == kind:
                self.plan_success = False
                return
            if kind == "open":
                self.grip[side] = 1.
                self.held.discard(side)
            elif kind == "close": self.grip[side] = kw["pos"]
            elif kind == "grasp":
                self.grip[side] = 0.
                if not self.miss_grasp: self.held.add(side)
            elif kind == "lift" and side in self.held:
                self.object.p[2] += kw["z"]
                self.on_table = self.on_target = False
                if self.drop_on_lift: self.held.clear()
            elif kind == "place":
                assert side in self.held
                self.object.p[:] = np.asarray(kw["target"]) - [0, 0, self.object.offset]
                if kw.get("is_open") is False:
                    self.object.q[:] = [0, 1, 0, 0]
                    self.on_table = self.on_target = False
                else:
                    self.grip[side] = 1.
                    self.held.discard(side)
                    self.on_table = self.kind == "rotate_qrcode"
                    self.on_target = not self.on_table
                    if self.on_table:
                        self.object.q[:] = [2**-.5, 2**-.5, 0, 0]

    def take_dense_action(self, control):
        assert self.current_control_source == "hil"
        assert control["left_gripper"] is None and control["right_gripper"] is None
        assert all(control[s + "_arm"]["position"].shape == (10, 6) for s in self.fingers)
        self.hold_calls += 1
        self.object.p[0] += self.hold_drift


@pytest.fixture(autouse=True)
def fake_armtag(monkeypatch):
    monkeypatch.setitem(sys.modules, "envs.utils", NS(ArmTag=lambda side: side))


@pytest.mark.parametrize("task", TASKS)
def test_full_suffix_contract(task):
    env = Env(task)
    assert env.episode_metadata["existing"] == "preserved"
    initial = env.adapter.infer_recovery_state(env)
    assert initial["stage_id"] == 1 and initial["recoverable"]
    it = env.adapter.recovery_stage_iterator(env)
    assert env.moves == []  # construction/confirmation must not move anything
    events = list(it)
    assert events[-1]["result"]["success"], events[-1]
    assert env.hold_calls == 6
    assert env.adapter.success_metrics(env)["recovery_ok"]
    assert env.adapter.success_metrics(env)["physical_quality_gate"]
    assert {e["stage_id"] for e in events if e["phase"] == "stage"} == set(env.adapter.stage_labels)
    if task == "place_empty_cup":
        assert next(kw for k, s, kw in env.moves if k == "close")["pos"] == .6
        assert next(kw for k, s, kw in env.moves if k == "grasp")["contact_point_id"] == 2


@pytest.mark.parametrize("task", ("rotate_qrcode", "place_empty_cup"))
def test_pick_place_matches_official_grasp_lift_place_without_homing(task):
    env = Env(task)
    events = list(env.adapter.recovery_stage_iterator(env))
    assert events[-1]["result"]["success"]
    actions = [(kind, side) for kind, side, _ in env.moves]
    assert not any(kind == "park" for kind, _ in actions)
    assert not any(side == "right" for _, side in actions)
    prefix = [("grasp", "left"), ("lift", "left"), ("place", "left")]
    assert [action for action in actions if action[0] in {"grasp", "lift", "place"}] == prefix + [("lift", "left")]
    if task == "place_empty_cup":
        assert actions[:2] == [("close", "left"), ("grasp", "left")]
    else:
        assert actions[0] == ("grasp", "left")


def test_handover_only_homes_source_in_parallel_with_final_place():
    env = Env("handover_block")
    events = list(env.adapter.recovery_stage_iterator(env))
    assert events[-1]["result"]["success"]
    actions = [(kind, side) for kind, side, _ in env.moves]
    assert actions[:4] == [("grasp", "left"), ("lift", "left"),
                           ("place", "left"), ("grasp", "right")]
    assert [action for action in actions if action[0] == "park"] == [("park", "left")]
    assert env.move_batches[-1] == (("park", "left"), ("place", "right"))


def test_empty_closed_gripper_is_opened_without_homing():
    for task, side in (("rotate_qrcode", "left"), ("handover_block", "left")):
        env = Env(task)
        env.grip[side] = 0.
        events = list(env.adapter.recovery_stage_iterator(env))
        assert events[-1]["result"]["success"]
        assert env.moves[0][0:2] == ("open", side)
        if task == "rotate_qrcode":
            assert not any(kind == "park" for kind, _, _ in env.moves)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("problem", ["plan", "build", "miss", "drop", "illegal", "drift"])
def test_failure_never_becomes_training_success(task, problem):
    env = Env(task)
    if problem == "plan": env.fail = "grasp"
    if problem == "build": env.construction_error = "grasp"
    if problem == "miss": env.miss_grasp = True
    if problem == "drop": env.drop_on_lift = True
    if problem == "illegal": env.illegal = True
    if problem == "drift": env.hold_drift = .01
    events = list(env.adapter.recovery_stage_iterator(env))
    assert events[-1]["phase"] == "done" and not events[-1]["result"]["success"]
    assert not env.adapter.success_metrics(env)["recovery_ok"]
    if problem in {"plan", "build", "miss", "drop", "illegal"}:
        assert not any(k == "place" for k, s, kw in env.moves)


@pytest.mark.parametrize("task", TASKS)
def test_forced_stage_cannot_skip_grasp_and_rejection_is_motionless(task):
    env = Env(task)
    events = list(env.adapter.recovery_stage_iterator(env, start_stage_id=3))
    assert "stage_precondition_mismatch" in events[-1]["result"]["reason"]
    assert env.moves == []
    env.object.p[2] = .5
    events = list(env.adapter.recovery_stage_iterator(env))
    assert "workspace" in events[-1]["result"]["reason"] and env.moves == []


@pytest.mark.parametrize("task", TASKS)
def test_stage_selection_rejects_ambiguous_grasp_before_takeover(task):
    env = Env(task)
    env.held.add("left")
    env.grip["left"] = .5
    state = env.adapter.infer_recovery_state(env)
    assert not state["recoverable"]
    assert "partial" in env.adapter.recovery_stage_rejection(state, 1)
    assert env.moves == []

    env.grip["left"] = 1.
    env.held.clear()
    state = env.adapter.infer_recovery_state(env)
    assert state["stage_id"] == 1
    assert "stage_precondition_mismatch" in env.adapter.recovery_stage_rejection(state, 3)
    assert env.adapter.recovery_stage_rejection(state, 1) is None


@pytest.mark.parametrize("task", TASKS)
def test_handback_and_new_takeover_do_not_regrasp(task):
    env = Env(task)
    it = env.adapter.recovery_stage_iterator(env)
    for event in it:
        if event.get("substage") in {"lift", "source_lift"}:
            assert event["verified"]
            break
    else: pytest.fail("did not reach lift")
    it.close()
    env.current_control_source = "policy"
    count = len(env.moves)
    assert env.adapter.infer_recovery_state(env)["stage_id"] == 3
    events = list(env.adapter.recovery_stage_iterator(env))
    assert events[-1]["result"]["success"]
    assert not any(k == "grasp" and s == "left" for k, s, kw in env.moves[count:])


@pytest.mark.parametrize("task", TASKS)
def test_pending_placement_not_certified_when_interrupted(task):
    env = Env(task)
    it = env.adapter.recovery_stage_iterator(env)
    for event in it:
        if event.get("substage") == "place_and_release":
            assert event["ok"] and not event["verified"]
            break
    it.close()
    assert not env.adapter.success_metrics(env)["recovery_ok"]
    env.current_control_source = "policy"
    list(env.adapter.recovery_stage_iterator(env))
    # A later takeover cannot silently certify the previously interrupted one.
    assert not env.adapter.success_metrics(env)["recovery_ok"]


@pytest.mark.parametrize("task", TASKS)
def test_identity_and_ambiguous_grasp(task):
    env = Env(task)
    env.held.add("left")
    env.grip["left"] = .5
    assert "partial" in env.adapter.infer_recovery_state(env)["reason"]
    env.grip["left"] = 0.
    assert env.adapter.snapshot(env)["held"] == ["left"]
    # Every actor uses name box. Target contact must not count as object grasp.
    env.scene.get_contacts = lambda: [NS(bodies=[NS(entity=env.target.actor), NS(entity=env.fingers["left"])])]
    assert env.adapter.snapshot(env)["held"] == []


def test_transfer_requires_receiver_attachment_before_source_release():
    env = Env("handover_block")
    it = env.adapter.recovery_stage_iterator(env)
    for event in it:
        if event.get("substage") == "receiver_grasp": break
    assert env.adapter.infer_recovery_state(env)["stage_id"] == 5
    env.held.remove("right")  # real-world loss simulated by fake contacts
    event = next(it)
    assert not event["ok"] and "grasp_not_verified" in event["error"]
    assert env.grip["left"] == 0. and "left" in env.held


def test_two_arm_transfer_resume_and_right_only_resume():
    for interrupt in ("receiver_grasp", "source_release"):
        env = Env("handover_block")
        it = env.adapter.recovery_stage_iterator(env)
        for event in it:
            if event.get("substage") == interrupt: break
        it.close()
        assert env.adapter.infer_recovery_state(env)["stage_id"] == (5 if interrupt == "receiver_grasp" else 6)
        assert list(env.adapter.recovery_stage_iterator(env))[-1]["result"]["success"]


def test_cup_tipped_and_handover_wrong_side_are_rejected():
    cup = Env("place_empty_cup")
    cup.cup.q[:] = [1, 0, 0, 0]
    assert "tipped_cup" in cup.adapter.infer_recovery_state(cup)["reason"]
    block = Env("handover_block")
    block.box.p[0] = .2
    assert "right_side" in block.adapter.infer_recovery_state(block)["reason"]


def test_held_cup_uses_actual_right_arm_not_initial_side():
    env = Env("place_empty_cup")
    env.held.add("right")
    env.grip["right"] = 0.
    env.on_table = False
    events = list(env.adapter.recovery_stage_iterator(env))
    assert events[-1]["result"]["success"]
    assert [(k, s) for k, s, kw in env.moves if k == "place"] == [("place", "right")]
    assert not any(k == "grasp" for k, s, kw in env.moves)


@pytest.mark.parametrize("task", TASKS)
def test_h50_defaults_guards_and_nonfinite(task):
    adapter = get_adapter(task)
    adapter.validate_action_chunk(np.zeros((50, 14)))
    for shape in ((10, 14), (50,), (50, 7)):
        with pytest.raises(ValueError): adapter.validate_action_chunk(np.zeros(shape))
    bad = np.zeros((50, 14)); bad[-1, -1] = np.nan
    with pytest.raises(ValueError): adapter.validate_action_chunk(bad)
    cli = runner.parse_args(["--task", task])
    assert cli.task_config == "demo_clean" and cli.frequency == 50
    assert cli.ckpt_name == adapter.default_checkpoint_name
    if task == "rotate_qrcode":
        assert cli.episode_protocol == "official-seen" and cli.instruction is None
    else:
        assert "{" not in cli.instruction
    assert runner.reserved_seed_label(100100, task)
    assert runner.reserved_seed_label(40000, task) is None
    for opt in (["--one-stage"], ["--es-mode", "enhanced"]):
        with pytest.raises(SystemExit): runner.parse_args(["--task", task, *opt])


@pytest.mark.parametrize("task,port", zip(TASKS, (18511, 18512, 18513)))
def test_shell_entry_dry_run(task, port, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "dagger").mkdir()
    (tmp_path / "dagger/runner.py").write_text("")
    monkeypatch.setenv("ROBOTWIN_ROOT", str(tmp_path))
    monkeypatch.setenv("HIL_DRY_RUN", "1")
    monkeypatch.setenv("MANUAL_MAX_ROLLOUTS", "2")
    output = subprocess.check_output(["bash", str(root / f"dagger/run_{task}.sh")], text=True)
    assert f"--task {task}" in output and f"--port {port}" in output
    assert "--frequency 50" in output and "--policy-name Pi_05" in output


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("physical_ok", (False, True))
def test_shared_runner_recording_and_handback(task, physical_ok, tmp_path, monkeypatch, action_wire_helpers):
    # Actual runner + real wire decoder; fake physics here isolates ownership,
    # discarded chunk, terminal labels, export eligibility and resume counting.
    base = type(get_adapter(task))

    class Adapter(base):
        def infer_recovery_state(self, env):
            return {**env.infer_recovery_state(), "reason": None}
        def recovery_stage_iterator(self, env, start_stage_id=None):
            return env.recovery_stage_iterator(start_stage_id)
        def success_metrics(self, env):
            return {"physical_quality_gate": physical_ok, "recovery_ok": True}

    monkeypatch.setattr(runner, "get_adapter", lambda _: Adapter())
    code, env, client, report = run_fake(monkeypatch, tmp_path, task=task,
        client_override=WireFormatClient("wrapped_dicts"), helper_overrides=action_wire_helpers)
    assert code == 0 and env.actions == [1, 2] and client.chunks == 2
    record = report["records"][0]
    assert record["task"] == task and record["accepted_for_training"] is physical_ok
    assert record["diagnostic_only"] is not physical_ok
    assert report["saved_valid_hil_episodes"] == int(physical_ok)
    resumed = runner.inspect_collection(tmp_path / "session", target_mode="hil", resume=True)
    assert resumed.saved_valid_hil == int(physical_ok)
