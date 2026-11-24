#!/usr/bin/env python3
"""Resample, train, diagnose, and run K32 online evaluation for click/shoes."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path

from run_online20 import EVAL, OUT, REPO, SIM_PY, SPECS, start_server


def run_logged(command: list[str], log: Path, env=None):
    print('RUN', ' '.join(command), 'log=', log, flush=True)
    with log.open('w') as stream:
        subprocess.run(command, cwd=REPO, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def one(task: str):
    spec = SPECS[task]
    task_output = OUT / task
    task_output.mkdir(parents=True, exist_ok=True)
    server_output = task_output / 'matched_sft_server'
    server_output.mkdir(exist_ok=True)
    server = start_server(task, spec, server_output)
    try:
        env = dict(os.environ)
        env['PYTHONPATH'] = f'{EVAL}:{EVAL / "XPolicyLab"}:{REPO / "robotwin_hil/overlay/scripts"}'
        command = [str(SIM_PY), '-u', str(Path(__file__).parent / 'resample_click_shoes_sft.py'),
                   '--task', task, '--port', str(spec['port']),
                   '--policy-checkpoint', str(spec['checkpoint']),
                   '--output-root', str(OUT / 'matched_sft_data')]
        run_logged(command, task_output / 'resample_matched_sft.log', env)
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    command = [str(SIM_PY), '-u', str(Path(__file__).parent / 'train.py'),
               '--task', task, '--data-root', str(OUT / 'matched_sft_data'),
               '--output-root', str(OUT / 'matched_sft_models'),
               '--steps', '10000', '--batch-size', '256', '--device', 'cuda:0']
    run_logged(command, task_output / 'train_matched_sft.log')
    command = [str(SIM_PY), '-u', str(Path(__file__).parent / 'score_outcomes.py'),
               '--task', task, '--trace-manifest', str(spec['baseline']),
               '--training-root', str(OUT / 'matched_sft_models'),
               '--encoder-asset', str(spec['encoder']),
               '--output', str(OUT / 'matched_sft_models' / task / 'sft_outcomes.json'),
               '--device', 'cuda:0']
    run_logged(command, task_output / 'score_matched_sft.log')
    command = [str(SIM_PY), '-u', str(Path(__file__).parent / 'run_online20.py'),
               '--task', task, '--count', '20',
               '--training-root', str(OUT / 'matched_sft_models')]
    run_logged(command, task_output / 'run_online20.log')
    print('COMPLETE', task, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('click_alarmclock', 'place_dual_shoes'), required=True)
    args = parser.parse_args()
    one(args.task)


if __name__ == '__main__':
    main()
