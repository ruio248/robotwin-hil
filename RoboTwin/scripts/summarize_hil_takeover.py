#!/usr/bin/env python3
"""Compare real i-key takeover requests from matched HIL rollout sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODES = ("off", "vanilla", "enhanced")


def valid_record(record: dict) -> bool:
    return int(record.get("policy_steps", 0)) > 0 or int(record.get("manual_takeover_requests", 0)) > 0


def load_session(path: Path, mode: str) -> dict:
    session = json.loads(path.read_text(encoding="utf-8"))
    if session.get("sampling_mode") != mode or session.get("takeover_eval") is not True:
        raise ValueError(f"{path}: expected a {mode} human takeover evaluation session")
    if session.get("aborted") or len(session.get("records", [])) != int(session.get("max_rollouts", -1)):
        raise ValueError(f"{path}: incomplete session; finish every scheduled rollout before comparing")
    seeds = [int(record["seed"]) for record in session["records"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{path}: duplicate episode seeds")
    return session


def summarize(sessions: dict[str, dict]) -> dict:
    if any(sessions[mode].get("sampling_activation", "fixed") != "fixed" for mode in MODES):
        raise ValueError("Manual e-key timing is operator-selected; paired takeover comparison requires fixed activation")
    baseline = sessions["off"]
    common = ("policy_name", "policy_host", "policy_port", "checkpoint_name", "task_config", "instruction",
              "frequency", "step_limit", "sampling_window", "seed_mode")
    seed_order = [int(record["seed"]) for record in baseline["records"]]
    for mode in MODES[1:]:
        session = sessions[mode]
        for field in common:
            if session.get(field) != baseline.get(field):
                raise ValueError(f"{mode}: {field} differs from off")
        if [int(record["seed"]) for record in session["records"]] != seed_order:
            raise ValueError(f"{mode}: seed list/order differs from off")
    vanilla, enhanced = sessions["vanilla"], sessions["enhanced"]
    if vanilla.get("critic_sha256") != enhanced.get("critic_sha256"):
        raise ValueError("Vanilla and Enhanced must use the same critic")
    for field in ("window_start", "window_end", "num_candidates", "horizon", "beta", "seed", "replay_atol"):
        if vanilla["sampling_config"][field] != enhanced["sampling_config"][field]:
            raise ValueError(f"Vanilla and Enhanced differ in {field}")

    rows = {mode: {int(record["seed"]): record for record in sessions[mode]["records"]}
            for mode in MODES}
    paired = [seed for seed in seed_order if all(valid_record(rows[mode][seed]) for mode in MODES)]
    if not paired:
        raise ValueError("No seed yielded a valid started rollout in all three arms")

    by_mode = {}
    for mode in MODES:
        records = sessions[mode]["records"]
        valid = [record for record in records if valid_record(record)]
        requested = sum(int(record.get("manual_takeover_requests", 0)) > 0 for record in valid)
        recovered = sum(int(record.get("intervention_count", 0)) > 0 for record in valid)
        paired_requested = sum(int(rows[mode][seed].get("manual_takeover_requests", 0)) > 0 for seed in paired)
        by_mode[mode] = {
            "valid_rollouts": len(valid),
            "manual_request_episodes": requested,
            "manual_request_probability": requested / len(valid) if valid else None,
            "recovery_trigger_episodes": recovered,
            "recovery_trigger_probability": recovered / len(valid) if valid else None,
            "paired_valid_rollouts": len(paired),
            "paired_manual_request_episodes": paired_requested,
            "paired_manual_request_probability": paired_requested / len(paired),
            "aborted_valid_rollouts": sum(record.get("rollout_status") == "aborted" for record in valid),
            "final_task_successes": sum(bool(record.get("final_check_success")) for record in valid),
        }

    return {
        "definition": "A real operator i-key request during policy control; cancelled stage confirmation still counts",
        "paired_seeds": paired,
        "excluded_from_paired": [seed for seed in seed_order if seed not in paired],
        "modes": by_mode,
        "paired_probability_differences": {
            "enhanced_minus_off": by_mode["enhanced"]["paired_manual_request_probability"]
                                  - by_mode["off"]["paired_manual_request_probability"],
            "enhanced_minus_vanilla": by_mode["enhanced"]["paired_manual_request_probability"]
                                      - by_mode["vanilla"]["paired_manual_request_probability"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument(f"--{mode}", required=True, type=Path, help=f"{mode} session_*.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize({mode: load_session(getattr(args, mode), mode) for mode in MODES})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for mode in MODES:
        result = report["modes"][mode]
        print(f"{mode}: i-key requests {result['paired_manual_request_episodes']}/{result['paired_valid_rollouts']} "
              f"= {result['paired_manual_request_probability']:.1%}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
