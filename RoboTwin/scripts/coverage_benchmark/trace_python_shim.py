#!/usr/bin/env python3
"""Use the unmodified PACE worker lifecycle with the opt-in trace evaluator.

Pass this executable as the worker's ``--client-python`` only for a separate
diagnostic run.  The worker still owns server health checks and cleanup.
"""

import os
import sys
from pathlib import Path


def main() -> None:
    arguments = sys.argv[1:]
    if len(arguments) < 2 or arguments[0] != "-u":
        raise ValueError("trace shim expects '-u pace/scripts/evaluate.py ...'")
    requested = Path(arguments[1]).resolve()
    if requested.name != "evaluate.py" or requested.parent.name != "scripts" or requested.parent.parent.name != "pace":
        raise ValueError(f"not the PACE evaluator: {requested}")
    traced = requested.with_name("trace_policy_queries.py")
    if not traced.is_file():
        raise FileNotFoundError(traced)
    os.execv(sys.executable, [sys.executable, "-u", str(traced), *arguments[2:]])


if __name__ == "__main__":
    main()
