#!/usr/bin/env python3
"""Run a paired 20-scene SFT vs K32-minimum-support success experiment."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'outputs/support_three_tasks_20261123'
RUNTIME = Path('/hdd/robotwin-hil-enhanced-sampling/outputs/robotwin_hil_runtime/RoboTwin')
EVAL = Path('/hdd/h10_dagger_lora_5050_68_ubuntu_20261115/runtime/eval_runtime')
OPENPI = Path('/hdd/robotwin-hil/RoboTwin/XPolicyLab/policy/Pi_05_RobotTwin/openpi')
SIM_PY = Path('/hdd/miniconda3/envs/robotwin_hil/bin/python')

SPECS = {
    'click_alarmclock': {
        'baseline': REPO / 'data/evaluations/h50_retest_20261122/click_alarmclock/sft/manifest.json',
        'scenes': REPO / 'outputs/h50_paired_20261123/fixed50_click_alarmclock.json',
        'checkpoint': Path('/hdd/official_nine_task_pi05_sft_5000_20261101/checkpoints/click_alarmclock/4999'),
        'asset': 'click_alarmclock_official50_v30_20261101',
        'config': 'pi05_click_alarmclock_official50_h50_state_5000_20261101',
        'encoder': OUT / 'assets/r3m18_pytorch_model.bin',
        'horizon': 50,
        'port': 18831,
    },
    'place_dual_shoes': {
        'baseline': REPO / 'data/evaluations/h10_paired_20261122/place_dual_shoes/sft/manifest.json',
        'scenes': REPO / 'outputs/h10_retest_20261122/fixed50_place_dual_shoes.json',
        'checkpoint': REPO / 'checkpoints/h10/tju_a800_eval_20261117/checkpoints/place_dual_shoes/full_h10_sft/4999',
        'asset': 'place_dual_shoes_official50_h10_v30_20261117',
        'config': 'pi05_place_dual_shoes_official50_h10_state_5000_20261117',
        'encoder': OUT / 'assets/r3m18_pytorch_model.bin',
        'horizon': 10,
        'port': 18832,
    },
    'place_empty_cup': {
        'baseline': REPO / 'data/evaluations/h10_paired_20261122/place_empty_cup/sft/manifest.json',
        'scenes': REPO / 'outputs/place_empty_cup_3epoch_20261122/control/fixed50.json',
        'checkpoint': REPO / 'checkpoints/h10/tju_h10_sft_fixed100_20261113/checkpoints/place_empty_cup/full_h10_sft/4999',
        'encoder': Path('/home/ruio/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'),
        'horizon': 10,
        'port': 18816,
    },
}


def sha(path: Path) -> str:
    from train import digest
    return digest(path)


def write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temp, path)


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def start_server(task: str, spec: dict, output: Path):
    port = spec['port']
    if port_open(port):
        if task == 'place_empty_cup':
            # The matching SFT server is already running for paired-data resampling.
            return None
        raise RuntimeError(f'Policy port {port} is already in use')
    if task == 'place_empty_cup':
        server_script = REPO / 'outputs/place_empty_cup_3epoch_20261122/control/run_eval_local.py'
        dagger = Path('/hdd/h10_dagger_lora_5050_68_ubuntu_20261115/runtime')
        command = ['/hdd/Openpi-PACE-eval-20261022/.venv/bin/python', '-u', str(server_script),
                   '--root', str(REPO / 'outputs/place_empty_cup_3epoch_20261122'),
                   '--dagger-root', str(dagger), '--task', task, '--mode', 'full',
                   '--port', str(port), '--serve-only', '--checkpoint', str(spec['checkpoint'])]
        env = dict(os.environ)
        env['PYTHONPATH'] = ':'.join([
            '/hdd/Openpi-PACE-eval-20261022/pace/src',
            '/hdd/Openpi-PACE-eval-20261022/packages/openpi-client/src',
            str(dagger / 'server_deps'), str(dagger / 'runtime_openpi/src'),
            str(EVAL), str(EVAL / 'XPolicyLab')])
        env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
        cwd = REPO
    else:
        config = output / 'policy.yml'
        config.write_text(
            f'policy_name: Pi_05_RobotTwin\ntask_name: {task}\naction_type: joint\n'
            f'env_cfg_type: aloha_agilex\ntrain_config_name: {spec["config"]}\n'
            f'repo_id: {spec["asset"]}\nmodel_path: {spec["checkpoint"]}\n'
            f'lora: false\nhost: 127.0.0.1\nport: {port}\nprotocol: ws\n')
        command = [str(OPENPI / '.venv/bin/python'), '-u', str(RUNTIME / 'XPolicyLab/setup_policy_server.py'),
                   '--config_path', str(config), '--host', '127.0.0.1', '--port', str(port)]
        env = dict(os.environ)
        env.update(PYTHONPATH=f'{RUNTIME}:{RUNTIME / "XPolicyLab"}:{REPO / "robotwin_hil/compat/XPolicyLab/policy/Pi_05_RobotTwin/openpi/src"}:{OPENPI / "src"}',
                   CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   XLA_PYTHON_CLIENT_MEM_FRACTION='0.65', PYTHONUNBUFFERED='1', MUJOCO_GL='egl',
                   ROBOTWIN_ROOT=str(RUNTIME))
        cwd = RUNTIME
    with (output / 'server.log').open('w') as stream:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(600):
        if proc.poll() is not None:
            raise RuntimeError(f'Policy server exited {proc.returncode}; see {output / "server.log"}')
        if port_open(port):
            return proc
        time.sleep(1)
    raise TimeoutError(f'Policy server did not bind port {port}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(SPECS), required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--training-root', type=Path, default=OUT)
    parser.add_argument('--rank-distance', action='store_true')
    parser.add_argument('--run-suffix', default=None)
    args = parser.parse_args()
    args.training_root = args.training_root.resolve()
    if not 1 <= args.count <= 50:
        parser.error('--count must be 1..50')
    task, spec = args.task, SPECS[args.task]
    output = OUT / task / f'online_k32_{args.count}'
    suffix = '_rank_distance' if args.rank_distance else ''
    if args.run_suffix is not None:
        if not args.run_suffix.replace('_', '').isalnum():
            parser.error('run suffix must be alphanumeric with underscores')
        suffix = '_' + args.run_suffix
    if suffix:
        output = output.with_name(output.name + suffix)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    baseline = json.loads(spec['baseline'].read_text())
    frozen = json.loads(spec['scenes'].read_text())
    old = [(e['scene_seed'], e['instruction'], e['success']) for e in baseline['episodes']
           if e.get('counted_in_official_result')]
    wanted = [(e['scene_seed'], e['instruction']) for e in frozen['episodes']]
    if len(old) != 50 or [(s, i) for s, i, _ in old] != wanted:
        raise ValueError('SFT baseline and frozen scene/instruction sequence differ')
    if baseline['infra_error_episodes'] or baseline['query_covered_official_episodes'] != 50:
        raise ValueError('SFT baseline contains missing or infrastructure-error episodes')
    frozen['episodes'] = frozen['episodes'][:args.count]
    scenes = output / 'scene_manifest.json'
    write(scenes, frozen)
    checkpoint = spec['checkpoint']
    norm = (checkpoint / 'assets' / spec['asset'] / 'norm_stats.json') if task != 'place_empty_cup' else (
        checkpoint / 'assets/place_empty_cup_official50_v30_20261101/norm_stats.json')
    if not (checkpoint / 'params/manifest.ocdbt').is_file() or not norm.is_file():
        raise FileNotFoundError('SFT checkpoint or norm stats missing')
    protocol = {'task': task, 'count': args.count, 'candidate_count': 32,
                'selection': 'argmin raw full-chunk expert-support logit',
                'checkpoint': str(checkpoint.resolve()), 'checkpoint_manifest_sha256': sha(checkpoint / 'params/manifest.ocdbt'),
                'norm_stats_sha256': sha(norm), 'baseline_trace_sha256': sha(spec['baseline']),
                'scene_manifest_sha256': sha(scenes), 'training_report_sha256': sha(args.training_root / task / 'report.json')}
    write(output / 'protocol.json', protocol)
    server = None
    try:
        server = start_server(task, spec, output)
        eval_root = EVAL if task != 'place_empty_cup' else REPO / 'outputs/place_empty_cup_3epoch_20261122/eval_runtime'
        policy_name = 'Pi_05_RobotTwin' if task != 'place_empty_cup' else 'Pi_05'
        command = [str(SIM_PY), '-u', str(Path(__file__).parent / 'eval_min_support.py'),
                   '--task', task, '--training-root', str(args.training_root),
                   '--encoder-asset', str(spec['encoder']), '--log', str(output / 'decisions.jsonl'),
                   '--microbatch', '4' if task != 'place_empty_cup' else '1',
                   '--evaluator', str(eval_root / 'scripts/eval_fixed100.py'), '--',
                   '--task_name', task, '--env_cfg_type', 'aloha_agilex', '--policy_name', policy_name,
                   '--host', '127.0.0.1', '--port', str(spec['port']), '--protocol', 'ws',
                   '--root_dir', str(eval_root), '--device_id', '0', '--task_config', 'demo_clean',
                   '--seed', '0', '--test_num', str(args.count), '--eval_batch', 'false',
                   '--expert_check', 'true', '--instruction_type', 'seen', '--frequency', '50',
                   '--additional_info', f'ckpt_setting=support_{task}_k32_{args.count}_20261123{suffix},action_type=joint',
                   '--trace_dir', str(output / 'trace'), '--scene_manifest', str(scenes)]
        write(output / 'command.json', {'argv': command, 'cwd': str(eval_root)})
        env = dict(os.environ)
        env.update(PYTHONPATH=f'{eval_root}:{eval_root / "XPolicyLab"}:{REPO / "robotwin_hil/overlay/scripts"}',
                   MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES='0',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false')
        with (output / 'eval.log').open('w') as stream:
            result = subprocess.run(command, cwd=eval_root, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=40 * 3600)
        if result.returncode:
            raise RuntimeError(f'Evaluator exited {result.returncode}; see {output / "eval.log"}')
        trace = json.loads((output / 'trace/manifest.json').read_text())
        rows = [e for e in trace['episodes'] if e.get('counted_in_official_result')]
        actual = [(e['scene_seed'], e['instruction']) for e in rows]
        expected = wanted[:args.count]
        if (trace['status'] != 'complete' or len(rows) != args.count or actual != expected
                or trace['infra_error_episodes'] or trace['query_covered_official_episodes'] != args.count):
            raise ValueError('Online trace is incomplete or scene sequence differs')
        before = [bool(e[2]) for e in old[:args.count]]
        after = [bool(e['success']) for e in rows]
        paired = np.asarray(after, dtype=np.int8) - np.asarray(before, dtype=np.int8)
        rng = np.random.default_rng(42)
        paired_bootstrap = paired[rng.integers(args.count, size=(10000, args.count))].mean(axis=1)
        decisions = [json.loads(line) for line in (output / 'decisions.jsonl').read_text().splitlines()]
        if not decisions or any(len(d['scores']) != 32 for d in decisions):
            raise ValueError('K32 decision audit missing')
        run_id = f'support_{task}_k32_{args.count}_20261123{suffix}'
        official_files = list((eval_root / 'eval_result' / task / policy_name / 'demo_clean' / run_id).glob('*/_result.txt'))
        if len(official_files) != 1:
            raise ValueError(f'Expected one official result, found {official_files}')
        official_rate = float(official_files[0].read_text().strip().splitlines()[-1])
        if abs(official_rate - sum(after) / args.count) > 1e-9:
            raise ValueError('Official result and trace success counts disagree')
        summary = {'task': task, 'count': args.count, 'baseline_successes': sum(before),
                   'k32_min_support_successes': sum(after), 'baseline_rate': sum(before)/args.count,
                   'min_support_rate': sum(after)/args.count,
                   'paired_rate_change': (sum(after)-sum(before))/args.count,
                   'paired_change_bootstrap_ci95': np.quantile(paired_bootstrap, [.025, .975]).tolist(),
                   'success_to_failure': sum(a and not b for a,b in zip(before,after)),
                   'failure_to_success': sum(not a and b for a,b in zip(before,after)),
                   'decisions': len(decisions), 'mean_ess_over_k_beta1': sum(d['ess_over_k_beta1'] for d in decisions)/len(decisions),
                   'mean_candidate_score_range': sum(d['score_range'] for d in decisions)/len(decisions),
                   'mean_min_minus_candidate_score': sum(d['min_support_score'] - d['candidate_score_mean'] for d in decisions)/len(decisions),
                   'mean_joint_rms_mrad': sum(d['joint_rms_mrad'] for d in decisions)/len(decisions),
                   'scene_results': [{'scene_seed': expected[i][0], 'baseline_success': before[i],
                                      'min_support_success': after[i]} for i in range(args.count)],
                   'protocol': str(output / 'protocol.json'), 'trace': str(output / 'trace/manifest.json'),
                   'official_result': str(official_files[0])}
        write(output / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


if __name__ == '__main__':
    main()
