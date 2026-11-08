#!/usr/bin/env python3
"""Sequential beat_block_hammer and click_alarmclock official-control SFT eval."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path


SFT_ROOT = Path('/media/ruio/hdd/official_nine_task_pi05_sft_5000_20261101')
FROZEN = SFT_ROOT / 'eval_open_lift_move_min_trace_20261102_retry1/control'
ROOT = SFT_ROOT / 'eval_beat_click_min_trace_20261103'
CORE = FROZEN / 'core_runner.py'
CORE_SHA = 'c4e9f037528b945f04e19fa24069a0da67d7705e7b486ec218560e26fc17b1c4'
SERVER = Path(__file__).with_name('serve_official_beat_click.py')
PORT = 18505
TASKS = {
    'beat_block_hammer': {
        'params': 'f4607930f849f3d8362e5c9f1262b124f4102f6cd2ab922319ad77b470f8dc07',
        'norm': 'f95f84b8fe981c5bd4701bac170c8873e60b10fab45e62f53517df211f0a54e7',
        'task': '90269f15f9ce274fb5bcad703c47b0b552ff5a2f0a3ff62d12dbe9a553018014',
        'instruction': 'f2e419c385c80a3d8dac0bec0516b19eea6cdfa1985d754466afaff7211e8262',
    },
    'click_alarmclock': {
        'params': 'cd077fc91faf138452db93627982e1534e5d20ffc50893fa7122538720dae828',
        'norm': '693767f8dce0d99c15e06e8d3e13778dffc0c00fcbfe4c41474e17b0',
        'task': 'f69a1ad4f1b483ab0dd82d8bad872e12047ed8d898f1fa4058de9e382ddeef49',
        'instruction': '46cc55f23d3224ff3bbdd5958b90d23fba6f2dbda0ac72471e0c5744697e0424',
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)


def state(phase: str, **detail: object) -> None:
    write(ROOT / 'control/state.json', {
        'phase': phase, 'time_utc': datetime.now(timezone.utc).isoformat(), **detail,
    })


def load_core():
    if sha(CORE) != CORE_SHA:
        raise RuntimeError('frozen official-control evaluator changed')
    spec = importlib.util.spec_from_file_location('frozen_beat_click_eval_core', CORE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {CORE}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> None:
    if ROOT.exists():
        raise FileExistsError(f'refusing to reuse evaluation output: {ROOT}')
    ROOT.joinpath('control').mkdir(parents=True)
    with (ROOT / 'control/queue.pid').open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    state('preflight', tasks=list(TASKS))
    core = load_core()
    complete = {}
    for task, digest in TASKS.items():
        checkpoint = SFT_ROOT / 'checkpoints' / task / '4999'
        repo_id = f'{task}_official50_v30_20261101'
        if sha(checkpoint / 'params/manifest.ocdbt') != digest['params']:
            raise RuntimeError(f'{task}: params manifest changed')
        if sha(checkpoint / 'assets' / repo_id / 'norm_stats.json') != digest['norm']:
            raise RuntimeError(f'{task}: norm_stats changed')
        if (checkpoint / 'train_state').exists():
            raise RuntimeError(f'{task}: evaluation checkpoint contains train_state')
        task_root = ROOT / task
        setup = task_root / 'control/server_setup'
        setup.mkdir(parents=True, exist_ok=False)
        shutil.copy2(SERVER, setup / 'stack_server_entry.py')
        with (setup / 'server.yaml').open('x') as stream:
            stream.write(
                'policy_name: Pi_05\n'
                f'task_name: {task}\n'
                'action_type: joint\n'
                'env_cfg_type: aloha_agilex\n'
                f'train_config_name: pi05_{task}_official50_h50_state_5000_20261101\n'
                f'repo_id: {repo_id}\n'
                f'model_path: {checkpoint}\n'
                f'host: 127.0.0.1\nport: {PORT}\nprotocol: ws\n'
            )
        core.ROOT = task_root
        core.PRIOR = setup
        core.CHECKPOINT = checkpoint
        core.TASK = task
        core.REPO_ID = repo_id
        core.PORT = PORT
        core.RUN_ID = f'{task}_official_min_trace_20261103_beat_click'
        core.CHECKPOINT_SHA = digest['params']
        core.NORM_SHA = digest['norm']
        core.TASK_SHA = digest['task']
        core.INSTRUCTION_SHA = digest['instruction']
        state('running', task=task, completed=list(complete))
        try:
            core.main()  # smoke, then 50 formal; validates trace and official result.
        except BaseException as error:
            core.state('failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            raise
        complete[task] = json.loads((task_root / 'formal/summary.json').read_text())
        write(ROOT / 'control/completed.json', complete)
    state('complete', completed=list(complete))


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        if ROOT.exists():
            state('failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
