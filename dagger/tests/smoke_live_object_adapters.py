"""Opt-in real-simulator diagnostics; no policy service, no training samples.

Use a fresh output directory. Not invoked by pytest or the collector launcher.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dagger.adapters import get_adapter
from dagger.recording import write_json
from dagger.runtime import activate_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("rotate_qrcode", "place_empty_cup", "handover_block"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[800300])
    parser.add_argument("--interrupt-after", help="Substage name, e.g. lift / receiver_grasp; no forced scene edits")
    cli = parser.parse_args()
    output = cli.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    activate_runtime(cli.runtime_root)
    from eval_policy_xpolicylab import load_task_args, class_decorator, safe_close_env

    args, _ = load_task_args({"task_name": cli.task, "task_config": "demo_clean", "policy_name": "Pi_05"})
    args.update(eval_mode=False, need_plan=True, save_data=False, save_video=False,
                render_freq=0, save_path=str(output), save_freq=15,
                render_max_num_materials=256, render_max_num_textures=256)
    adapter = get_adapter(cli.task)
    records = []
    for seed in cli.seeds:
        env = class_decorator(cli.task)
        record = {"task": cli.task, "seed": seed, "diagnostic_only": True,
                  "accepted_for_training": False, "events": [], "passed": False}
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=False, **args)
            adapter.initialize_episode(env)
            record["initial"] = adapter.infer_recovery_state(env)
            interrupted = False
            iterator = adapter.recovery_stage_iterator(env)
            for event in iterator:
                record["events"].append(event)
                print(seed, event.get("substage"), event.get("ok"), event.get("result", {}).get("reason"), flush=True)
                if cli.interrupt_after and event.get("substage") == cli.interrupt_after and event.get("ok"):
                    iterator.close()
                    env.current_control_source = "policy"  # No simulated policy action is invented.
                    record["handoff_state"] = adapter.infer_recovery_state(env)
                    interrupted = True
                    break
            if cli.interrupt_after and not interrupted:
                raise RuntimeError("Requested handback boundary was not reached")
            if interrupted:
                record["events"].extend(adapter.recovery_stage_iterator(env))
            record["final"] = adapter.success_metrics(env)
            record["passed"] = bool(record["events"][-1].get("result", {}).get("success")
                                     and record["final"]["physical_quality_gate"] and record["final"]["recovery_ok"])
        except Exception as error:
            record["error"] = repr(error)
        finally:
            safe_close_env(env)
        records.append(record)
        write_json(output / f"seed_{seed}.json", record)
    passed = all(row["passed"] for row in records)
    write_json(output / "summary.json", {"task": cli.task, "diagnostic_only": True,
               "accepted_for_training": False, "passed": passed, "results": records})
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
