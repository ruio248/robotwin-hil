"""Restrict the frozen official-data Pi0.5 server to this evaluation pair."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


SOURCE = Path(
    '/media/ruio/hdd/official_nine_task_pi05_sft_5000_20261101/'
    'eval_open_lift_move_min_trace_20261102_retry1/control/serve_official_three.py'
)
SOURCE_SHA = '421e617397cc8245a11f6277c7364abcef545d496704349faf6988965af69471'
TASKS = {'beat_block_hammer', 'click_alarmclock'}


def main() -> None:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError('frozen official-data policy server changed')
    spec = importlib.util.spec_from_file_location('frozen_official_policy_server', SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {SOURCE}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TASKS = TASKS
    module.main()


if __name__ == '__main__':
    main()
