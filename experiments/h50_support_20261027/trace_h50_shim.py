#!/usr/bin/env python3
"""Route the existing PACE worker to the isolated full-H50 query recorder."""
import os
import sys
from pathlib import Path


def main() -> None:
    arguments = sys.argv[1:]
    if len(arguments) < 2 or arguments[0] != "-u":
        raise ValueError("expected '-u pace/scripts/evaluate.py ...'")
    requested = Path(arguments[1]).resolve()
    if requested.name != "evaluate.py" or requested.parent.name != "scripts" or requested.parent.parent.name != "pace":
        raise ValueError(f"unexpected evaluator: {requested}")
    recorder = Path(__file__).with_name("trace_policy_queries_h50.py")
    if not recorder.is_file():
        raise FileNotFoundError(recorder)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(requested.parent), environment.get("PYTHONPATH", "")])
    os.execve(sys.executable, [sys.executable, "-u", str(recorder), *arguments[2:]], environment)


if __name__ == "__main__":
    main()
