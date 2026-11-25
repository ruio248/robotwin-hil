"""Collect or evaluate cup rollouts through the frozen official evaluator."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/'experiments/support_three_tasks_20261123'))
from run_online20 import SPECS, SIM_PY, start_server, port_open, sha, write

TASK = 'place_empty_cup'
SPEC = SPECS[TASK]
ROOT = Path(os.environ.get('OCCUPANCY_OUTPUT_ROOT', REPO/'outputs/occupancy_cup_20261125'))
EVAL = REPO/'outputs/place_empty_cup_3epoch_20261122/eval_runtime'

def run(mode, server_port):
    output = ROOT/mode
    output.mkdir(exist_ok=True)
    scenes = json.loads(SPEC['scenes'].read_text())
    pool = scenes['episodes']
    scenes['episodes'] = pool[20:40] if mode == 'collect' else pool[:20]
    assert len({r['scene_seed'] for r in pool[:40]}) == 40
    write(output/'scenes.json', scenes)
    argv = [str(SIM_PY), '-u', str(HERE/'evaluate.py'), mode, str(output),
            str(EVAL/'scripts/eval_fixed100.py'), '--task_name', TASK, '--env_cfg_type', 'aloha_agilex',
            '--policy_name', 'Pi_05', '--host', '127.0.0.1', '--port', str(server_port), '--protocol', 'ws',
            '--root_dir', str(EVAL), '--device_id', '0', '--task_config', 'demo_clean',
            '--seed', '0', '--test_num', '20', '--eval_batch', 'false', '--expert_check', 'true',
            '--instruction_type', 'seen', '--frequency', '50',
            '--additional_info', f'ckpt_setting=occupancy_20261125_{mode},action_type=joint',
            '--trace_dir', str(output/'trace'), '--scene_manifest', str(output/'scenes.json')]
    write(output/'protocol.json', dict(argv=argv, checkpoint=str(SPEC['checkpoint']),
         checkpoint_manifest_sha256=sha(SPEC['checkpoint']/'params/manifest.ocdbt'),
         norm_stats_sha256=sha(SPEC['checkpoint']/'assets/place_empty_cup_official50_v30_20261101/norm_stats.json'),
         encoder_sha256=sha(SPEC['encoder']),candidate_microbatch=1 if os.getenv('OCCUPANCY_SERIAL') else 8))
    env = dict(os.environ, PYTHONPATH=f'{EVAL}:{EVAL/"XPolicyLab"}:{REPO/"robotwin_hil/overlay/scripts"}',
               MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES='0', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    with (output/'eval.log').open('w') as log:
        subprocess.run(argv, cwd=EVAL, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    trace=json.loads((output/'trace/manifest.json').read_text())
    assert trace['status']=='complete' and not trace['infra_error_episodes']
    assert [(r['scene_seed'],r['instruction']) for r in trace['episodes']] == [
        (r['scene_seed'],r['instruction']) for r in scenes['episodes']]
    for row in trace['episodes']:
        assert (output/'trace'/row['directory']/'terminal.npz').is_file()

def start_batch_server(port):
    runtime=Path('/hdd/h10_dagger_lora_5050_68_ubuntu_20261115/runtime')
    evaluator=runtime/'eval_runtime'
    pace=Path('/hdd/Openpi-PACE-eval-20261022')
    server_script=REPO/'outputs/place_empty_cup_3epoch_20261122/control/run_eval_local.py'
    env=dict(os.environ, PYTHONPATH=':'.join(map(str,(runtime/'runtime_openpi/src',evaluator,
        evaluator/'XPolicyLab',pace/'pace/src',pace/'packages/openpi-client/src',runtime/'server_deps'))),
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.7')
    argv=[str(pace/'.venv/bin/python'),'-u',str(HERE/'serve_batch.py'),str(server_script),
          '--root',str(server_script.parents[1]),'--dagger-root',str(runtime),'--task',TASK,
          '--mode','full','--port',str(port),'--serve-only','--checkpoint',str(SPEC['checkpoint'])]
    with (ROOT/'batch_server.log').open('w') as log:
        server=subprocess.Popen(argv,env=env,start_new_session=True,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            if server.poll() is not None:
                raise RuntimeError('Batch server stopped')
            if port_open(port):
                return server
            time.sleep(1)
        raise TimeoutError('Batch server did not bind')
    except BaseException:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            server.wait()
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['collect','online','online-k32','native','random','max','min'])
    parser.add_argument('--serial',action='store_true')
    args=parser.parse_args()
    modes=['native','random','max','min'] if args.mode=='online' else [args.mode]
    if args.mode == 'online-k32':
        modes = ['random', 'max', 'min']
    if args.serial:
        os.environ['OCCUPANCY_SERIAL']='1'
    port=18926
    if port_open(port):
        raise RuntimeError(f'Unexpected existing policy server at {port}')
    server=(start_server(TASK,dict(SPEC,port=port),ROOT) if args.mode=='collect' or args.serial
            else start_batch_server(port))
    try:
        if args.mode in ('online', 'online-k32'):
            env=dict(os.environ,PYTHONPATH=f'{EVAL}:{EVAL/"XPolicyLab"}:{REPO/"robotwin_hil/overlay/scripts"}')
            with (ROOT/'offline_candidates.log').open('w') as log:
                subprocess.run([str(SIM_PY),'-u',str(HERE/'offline_candidates.py'),
                    str(EVAL/'scripts/eval_fixed100.py'),str(port)],cwd=EVAL,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        for mode in modes:
            run(mode,port)
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            server.wait(timeout=30)

if __name__=='__main__':
    main()
