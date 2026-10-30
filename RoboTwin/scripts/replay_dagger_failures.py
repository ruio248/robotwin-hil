#!/usr/bin/env python3
"""Diagnostic-only failed-scene replay, terminal rescue, and paired early rescue.

Historical evaluations did not retain actions/RNG state: these are same-scene
reruns, NOT claims of exact historical trajectory reproduction. A fresh control
rollout records every executed action. Early rescue replays that exact prefix
in a separate process, and is gated on agreement of physical state snapshots.
No raw HIL/training data, expert prefilter, scene replacement or automatic retry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


EARLY_STEPS = {"handover_block": 150, "move_can_pot": 50, "stack_blocks_two": 100}


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_differences(reference, actual, prefix=""):
    """Compare every logged scalar/bool, not just the object position."""
    differences = []
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or reference.keys() != actual.keys():
            return [(prefix, "keys_mismatch")]
        for key in reference:
            differences += state_differences(reference[key], actual[key], prefix + "/" + key)
    elif isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            return [(prefix, "length_mismatch")]
        for i, value in enumerate(reference):
            differences += state_differences(value, actual[i], prefix + f"/{i}")
    elif isinstance(reference, bool) or reference is None or isinstance(reference, str):
        if reference != actual:
            differences.append((prefix, "value_mismatch"))
    else:
        delta = abs(float(reference) - float(actual))
        if not math.isfinite(delta) or delta > 0.001:
            differences.append((prefix, delta))
    return differences


def make_manifest(repo, output):
    base = Path(repo) / "outputs/robotwin_dagger_screen_20261024/results"
    tasks = {}
    for task in ("handover_block", "move_can_pot"):
        folder = base / (task + "_h50")
        aggregate, protocol = read(folder / "aggregate_summary.json"), read(folder / "control/protocol.json")
        assert aggregate["complete"] and len(aggregate["records"]) == 50
        failures = [{"seed": r["seed"], "prompt": protocol["prompt"], "historical_record": r}
                    for r in aggregate["records"] if r["success"] is False]
        tasks[task] = {"step_limit": protocol["step_limit"], "transport": "xpolicy",
                       "source": str(folder), "source_sha256": digest(folder / "aggregate_summary.json"),
                       "source_protocol": protocol, "cases": failures}
    task = "stack_blocks_two"
    folder = base / "three_added_public_pi05_h50_20261025" / task
    meta = read(folder / "metadata.json")
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines() if line]
    assert meta["status"] == "complete" and len(rows) == len({r["scene_seed"] for r in rows}) == 50
    tasks[task] = {"step_limit": 800, "transport": "native", "source": str(folder),
                   "source_sha256": digest(folder / "episodes.jsonl"), "source_protocol": meta,
                   "cases": [{"seed": r["scene_seed"], "prompt": r["instruction"], "historical_record": r}
                             for r in rows if r["success"] is False]}
    for task, data in tasks.items():
        data["early_step"] = EARLY_STEPS[task]
        assert len({c["seed"] for c in data["cases"]}) == len(data["cases"])
    save(output, {"diagnostic_only": True, "accepted_for_training": False,
                  "historical_exact_action_replay_possible": False, "tasks": tasks})


def adapter_for(task):
    if task == "handover_block":
        from handover_block_recovery import handover_snapshot, classify_handover_state, iter_expert_suffix
        return handover_snapshot, classify_handover_state, iter_expert_suffix
    if task == "move_can_pot":
        from move_can_pot_recovery import move_can_snapshot, classify_move_can_state, iter_move_can_expert_suffix
        return move_can_snapshot, lambda cur, prev: classify_move_can_state(cur), iter_move_can_expert_suffix
    from hg_dagger_extra_task_recovery import (stack_blocks_two_snapshot,
        classify_stack_blocks_two_state, iter_stack_blocks_two_expert_suffix)
    return stack_blocks_two_snapshot, classify_stack_blocks_two_state, iter_stack_blocks_two_expert_suffix


def physical_quality(env, task, snap):
    official = bool(env.check_success())
    if task == "handover_block":
        physical = bool(env.check_actors_contact(env.box.get_name(), env.target_box.get_name())
                        and env.is_left_gripper_open() and env.is_right_gripper_open())
    elif task == "move_can_pot":
        physical = bool(snap["can_table_contact"] and not snap["assigned_gripper_can_contact"])
    else:
        physical = snap["strict_physical_quality"] is True
    if hasattr(env, "planned_joints_legal"):
        legal, absmax = env.planned_joints_legal()
    else:
        # Same ±pi unwrapped-plan guard used by the HIL Base_Task, without
        # modifying the publisher's original simulator used for stack_two.
        import numpy as np
        values = [np.asarray(item["position"]) for path in (env.left_joint_path, env.right_joint_path)
                  for item in path if isinstance(item, dict) and item.get("position") is not None]
        absmax = max([float(np.max(np.abs(v))) for v in values if v.size] or [0.0])
        legal = math.isfinite(absmax) and absmax <= math.pi
    return {"official_success": official, "physical_gate": physical,
            "joints_legal": bool(legal), "joint_absmax": float(absmax),
            "strict_success": bool(official and physical and legal)}


def worker(cli, cfg, spec, case):
    # Deployment supplies the original simulator/helper roots. Never replace
    # shared environment modules or edit task physics for this diagnostic.
    import numpy as np
    if cfg.get("pace_environment"):
        sys.path.insert(0, cfg["sim_cwd"])
        import envs  # freeze the publisher's environment package before loading HIL helpers
    if cfg.get("renderer_device"):
        import sapien.core as sapien
        original_renderer = sapien.SapienRenderer
        def pinned_renderer(*a, **kw):
            kw["device"] = sapien.Device(cfg["renderer_device"])
            return original_renderer(*a, **kw)
        sapien.SapienRenderer = pinned_renderer
    if cfg.get("eval_helper"):
        import importlib.util
        module_spec = importlib.util.spec_from_file_location("eval_policy_xpolicylab", cfg["eval_helper"])
        ev = importlib.util.module_from_spec(module_spec)
        sys.modules["eval_policy_xpolicylab"] = ev
        module_spec.loader.exec_module(ev)
    else:
        import eval_policy_xpolicylab as ev
    task_snapshot, classify, recovery = adapter_for(cli.task)
    def snapshot(env, steps):
        result = task_snapshot(env, steps)
        result["robot_qpos"] = np.asarray(env.get_obs()["joint_action"]["vector"]).tolist()
        names = {"handover_block": ["box"], "move_can_pot": ["can", "pot"],
                 "stack_blocks_two": ["block1", "block2"]}[cli.task]
        result["actor_quaternions"] = {name: np.asarray(getattr(env, name).get_pose().q).tolist() for name in names}
        return result
    root = Path(cfg["robotwin_root"])
    user = {"task_name": cli.task, "task_config": "demo_clean", "policy_name": "Pi_05_RobotTwin",
            "host": "127.0.0.1", "port": cfg["port"], "protocol": "ws", "action_type": "joint",
            "xpolicylab_root": cfg.get("xpolicylab_root", str(root / "XPolicyLab")),
            "ckpt_setting": "public_pi05_" + cli.task}
    if cfg.get("pace_environment"):
        from pace.robotwin import load_environment
        env, args = load_environment(cli.task)
    else:
        args, _ = ev.load_task_args(user)
        env = ev.class_decorator(cli.task)
    args.update(eval_instruction="seen", eval_mode=True, render_freq=0, need_plan=True,
                save_data=False, save_video=False, save_path=str(cli.output))
    user["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    user["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    client = None
    out = Path(cli.output) / cli.task / f"seed_{case['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / (cli.stage + ".json")
    if result_path.exists():
        raise FileExistsError(result_path)
    marker = {"task": cli.task, "seed": case["seed"], "diagnostic_only": True,
              "accepted_for_training": False, "stage": cli.stage, "prompt": case["prompt"]}
    trace = {"actions": [], "snapshots": {}}
    steps = 0
    try:
        env.setup_demo(now_ep_num=0, seed=case["seed"], is_test=bool(cfg.get("pace_environment")), **args)
        env.step_lim = spec["step_limit"]
        env.set_instruction(case["prompt"])
        env.save_data = False
        if hasattr(env, "_update_render"):
            env._update_render()
        previous = snapshot(env, 0)
        trace["snapshots"]["0"] = previous
        save(out / (cli.stage + "_initial.json"), previous)
        if cli.stage == "control":
            if spec["transport"] == "native":
                from hg_dagger_task_collector import build_openpi_policy_client, openpi_observation_from_robotwin
                client, metadata = build_openpi_policy_client("127.0.0.1", cfg["port"], cli.task)
                marker["policy_metadata"] = metadata
            else:
                client = ev.build_policy_client(user)
                ev.prepare_policy_case(client, cli.task, case["seed"], case["prompt"], "joint")
                ev.reset_policy(client)
            while not ev.is_episode_end(env):
                obs = env.get_obs()
                if spec["transport"] == "native":
                    response = client.infer(openpi_observation_from_robotwin(obs, case["prompt"]))
                else:
                    client.call(func_name="update_obs", obs=ev.robotwin_obs_to_xpolicylab(
                        obs, instruction=case["prompt"], env_idx=0, frequency=50, task_env=env))
                    response = client.call(func_name="get_action")
                chunk = ev.normalize_action_chunk(response)
                if len(chunk) != 50:
                    raise ValueError(f"Expected H50, got {len(chunk)}")
                previous = snapshot(env, steps)
                for action in chunk:
                    flat, action_type = ev.xpolicylab_action_to_robotwin(action, action_type="joint", current_observation=obs)
                    flat = np.asarray(flat)
                    if flat.shape != (14,) or not np.isfinite(flat).all():
                        raise ValueError(f"Invalid action: {flat}")
                    trace["actions"].append({"action": flat.tolist(), "type": action_type})
                    env.current_control_source = "policy"
                    env.take_action(flat, action_type=action_type)
                    steps += 1
                    if ev.is_episode_end(env):
                        break
                    obs = env.get_obs()
                    if spec["transport"] == "xpolicy":
                        client.call(func_name="update_obs", obs=ev.robotwin_obs_to_xpolicylab(
                            obs, instruction=case["prompt"], env_idx=0, frequency=50, task_env=env))
                trace["snapshots"][str(steps)] = snapshot(env, steps)
                save(out / "action_trace.json", trace)
                print(f"CONTROL task={cli.task} seed={case['seed']} steps={steps}", flush=True)
            success = bool(env.eval_success or env.check_success())
            marker.update(policy_steps=steps, control_success=success,
                          failure_reproduced=not success, historical_success=False,
                          historical_exact_action_replay=False, final_snapshot=snapshot(env, steps))
            save(out / "control_rollout.json", marker)
            if success:
                marker["terminal_recovery"] = {"status": "not_needed_control_succeeded", "rescued": False}
            else:
                marker["terminal_recovery"] = rescue(env, cli.task, steps, previous, snapshot, classify, recovery)
        else:
            control = read(out / "control.json")
            trace = read(out / "action_trace.json")
            target = spec["early_step"]
            marker["control_failure_reproduced"] = control["failure_reproduced"]
            if len(trace["actions"]) < target:
                marker["early_recovery"] = {"status": "control_ended_before_fixed_intervention", "rescued": False}
                save(result_path, marker)
                return
            differences = state_differences(trace["snapshots"]["0"], previous)
            for i, action in enumerate(trace["actions"][:target]):
                if steps % 50 == 0:
                    previous = snapshot(env, steps)
                env.get_obs()  # match the control observation/render cadence
                env.current_control_source = "policy"
                env.take_action(np.asarray(action["action"]), action_type=action["type"])
                steps += 1
                if ev.is_episode_end(env):
                    break
                if steps % 50 == 0:
                    differences += state_differences(trace["snapshots"][str(steps)], snapshot(env, steps), f"step{steps}")
            marker.update(policy_steps=steps, prefix_matches=not differences and steps == target,
                          prefix_differences=differences, intervention_step=target)
            if differences or steps != target:
                marker["early_recovery"] = {"status": "prefix_replay_mismatch", "rescued": False}
            else:
                marker["early_recovery"] = rescue(env, cli.task, steps, previous, snapshot, classify, recovery)
        save(result_path, marker)
    finally:
        if client is not None:
            if spec["transport"] == "native":
                from hg_dagger_task_collector import close_openpi_policy_client
                close_openpi_policy_client(client)
            else:
                ev.close_policy_client(client)
        ev.safe_close_env(env, clear_cache=False)


def rescue(env, task, steps, previous, snapshot, classify, recovery):
    current = snapshot(env, steps)
    decision = classify(current, previous)
    result = {"takeover_snapshot": current, "decision": decision, "events": [], "rescued": False}
    if not decision.get("ready"):
        result.update(status="rejected", reason=decision.get("reason"))
        return result
    # Expert suffix uses move(), not policy take_action(); no counter reset,
    # scene reset, teleportation or expansion of classifier acceptance gates.
    for event in recovery(env, decision):
        result["events"].append(event)
        if len(result["events"]) > 40:
            raise RuntimeError("expert suffix exceeded diagnostic stage-action budget")
        if event["phase"] == "done":
            result["expert_result"] = event["result"]
            break
    if "expert_result" not in result:
        raise RuntimeError("expert suffix exited without a done event")
    result["immediate_quality"] = physical_quality(env, task, snapshot(env, steps))
    # Extra physics-only settling is a stricter diagnostic, not a changed
    # policy evaluation success threshold. Record both checks separately.
    for _ in range(100):
        env.scene.step()
    final = snapshot(env, steps)
    result.update(final_snapshot=final, settled_quality=physical_quality(env, task, final),
                  settling_physics_steps=100)
    result["rescued"] = bool(result["expert_result"].get("success") and env.plan_success
                              and result["immediate_quality"]["strict_success"]
                              and result["settled_quality"]["strict_success"])
    result["status"] = "rescued" if result["rescued"] else "expert_failed"
    return result


def aggregate(task, cases, output):
    rows = []
    for case in cases:
        folder = Path(output) / task / f"seed_{case['seed']}"
        if (folder / "control.json").exists() and (folder / "early.json").exists():
            rows.append({"seed": case["seed"], "control": read(folder / "control.json"),
                         "early": read(folder / "early.json")})
    reproduced = [r for r in rows if r["control"]["failure_reproduced"]]
    return {"task": task, "diagnostic_only": True, "accepted_for_training": False,
            "complete": len(rows) == len(cases), "expected": len(cases), "completed": len(rows),
            "failure_reproduced": len(reproduced), "control_succeeded": len(rows) - len(reproduced),
            "terminal_rescued": sum(r["control"]["terminal_recovery"]["rescued"] for r in reproduced),
            "early_rescued_on_reproduced_failures": sum(r["early"]["early_recovery"]["rescued"] for r in reproduced),
            "early_status": dict(Counter(r["early"]["early_recovery"]["status"] for r in rows)), "records": rows}


def run(cli, cfg, spec):
    root = Path(cli.output)
    control = root / cli.task / "control"
    control.mkdir(parents=True, exist_ok=True)
    def state(phase, **kw):
        save(control / "state.json", {"phase": phase, "time": time.strftime("%FT%T%z"), **kw})
    with (control / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (control / "state.json").exists():
            raise FileExistsError("refusing automatic restart/resume of an existing batch")
        save(control / "deployment.json", cfg)
        save(control / "source_protocol.json", spec)
        save(control / "runner.json", {"pid": os.getpid(), "argv": sys.argv})
        if cfg.get("mount"):
            source = subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", cfg["mount"]], text=True).strip()
            if source != "/dev/sda1":
                raise RuntimeError(f"unexpected data disk {source}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", cfg["port"]))
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                   "--format=csv,noheader,nounits", "-i", str(cfg["gpu"])], text=True).strip())
        if free < cfg["minimum_free_mib"]:
            raise RuntimeError(f"insufficient free GPU memory: {free}")
        server = None
        try:
            env = os.environ | cfg["common_env"]
            for name in cfg.get("mkdirs", []):
                Path(name).mkdir(parents=True, exist_ok=True)
            state("starting_server")
            with (control / "server.log").open("w") as log:
                server = subprocess.Popen(cfg["server_command"], cwd=cfg["server_cwd"],
                    env=env | cfg["server_env"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(control / "server_process.json", {"pid": server.pid, "command": cfg["server_command"]})
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited rc={server.returncode}")
                with socket.socket() as sock:
                    sock.settimeout(1)
                    ready = sock.connect_ex(("127.0.0.1", cfg["port"])) == 0
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup timeout")
                time.sleep(2)
            for case in spec["cases"]:
                for stage in ("control", "early"):
                    state("running", seed=case["seed"], stage=stage)
                    folder = root / cli.task / f"seed_{case['seed']}"
                    folder.mkdir(parents=True, exist_ok=True)
                    command = [cfg["sim_python"], "-u", str(Path(__file__).resolve()),
                        "worker", "--manifest", cli.manifest, "--deployment", cli.deployment,
                        "--task", cli.task, "--output", cli.output, "--seed", str(case["seed"]), "--stage", stage]
                    with (folder / (stage + ".log")).open("w") as log:
                        completed = subprocess.run(command, cwd=cfg["sim_cwd"], env=env | cfg["sim_env"],
                            stdout=log, stderr=subprocess.STDOUT, timeout=5400)
                    if completed.returncode:
                        raise RuntimeError(f"{stage} seed={case['seed']} infrastructure error rc={completed.returncode}")
                    if not (folder / (stage + ".json")).is_file():
                        raise RuntimeError("worker exited without durable result")
                save(root / cli.task / "summary.json", aggregate(cli.task, spec["cases"], root))
            state("complete")
        except Exception:
            state("failed", error=traceback.format_exc())
            raise
        finally:
            if server is not None and server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Report cleanup failure rather than kill unrelated PIDs.
                    state("cleanup_failed", server_pid=server.pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("manifest", "run", "worker"))
    parser.add_argument("--repo")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--deployment")
    parser.add_argument("--task", choices=tuple(EARLY_STEPS))
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=("control", "early"))
    cli = parser.parse_args()
    if cli.mode == "manifest":
        make_manifest(cli.repo, cli.manifest)
        return
    cfg = read(cli.deployment)
    manifest = read(cli.manifest)
    assert manifest["diagnostic_only"] and manifest["accepted_for_training"] is False
    spec = manifest["tasks"][cli.task]
    if cli.mode == "run":
        run(cli, cfg, spec)
    else:
        case = next(c for c in spec["cases"] if c["seed"] == cli.seed)
        worker(cli, cfg, spec, case)


if __name__ == "__main__":
    main()
