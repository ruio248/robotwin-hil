"""Score frozen-SFT K32 proposals on held-out real query observations."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

from run import ROOT, write
from selection import install

def main():
    evaluator,port=sys.argv[1:]
    path=Path(evaluator)
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('occupancy_candidates_eval',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    from client_server.ws.model_client import WsModelClient

    out=ROOT/'offline_candidates'
    out.mkdir(exist_ok=False)
    install(module,'max',out)
    training=json.loads((ROOT/'training.json').read_text())
    trace=ROOT/'collect/trace'
    manifest=json.loads((trace/'manifest.json').read_text())
    client=WsModelClient(url=f'ws://127.0.0.1:{port}',evaluation_id='occupancy-cup-offline',
        trial_id='occupancy-cup-offline',action_case_id='occupancy-cup-offline')
    client._robotwin_protocol='ws'
    try:
        for ep in manifest['episodes']:
            if ep['scene_seed'] not in training['sft'][1]:
                continue
            module.prepare_policy_case(client,'place_empty_cup',ep['scene_seed'],ep['instruction'],'joint')
            module.reset_policy(client)
            for ix in np.linspace(0,len(ep['queries'])-1,3,dtype=int):
                q=ep['queries'][ix]
                with np.load(trace/ep['directory']/q['file']) as z:
                    obs=dict(data_format_version='v1.0',env_idx=0,instruction=ep['instruction'],
                        instructions=[ep['instruction']],
                        vision={k[4:]:{'color':z[k]} for k in z.files if k.startswith('rgb_')},
                        state={k[6:]:z[k] for k in z.files if k.startswith('state_')},
                        additional_info=dict(frequency=50,occupancy_policy_steps=q['policy_steps_before_query'],
                            occupancy_step_limit=500))
                client.call(func_name='update_obs',obs=obs)
                client.call(func_name='get_action')
    finally:
        module.close_policy_client(client)
    rows=[json.loads(line) for line in (out/'decisions.jsonl').read_text().splitlines()]
    assert len(rows)==12 and all(len(r['scores'])==32 for r in rows)
    write(out/'summary.json',dict(decisions=len(rows),means={k:float(np.mean([r[k] for r in rows]))
        for k in ('range','pair_gap','ess_over_k','joint_rms_mrad','gripper_rms')}))

if __name__=='__main__':
    main()
