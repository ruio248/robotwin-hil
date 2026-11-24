"""Complete both frozen cohorts to K32, verifying K4 preservation."""
from run_online20 import *
from resample_cup_sft import np
root=OUT/'matched_sft_k32'
for task in ['place_dual_shoes','place_empty_cup']:
    spec=SPECS[task]
    log=root/task
    log.mkdir(parents=True,exist_ok=True)
    proc=None
    try:
        proc=start_server(task,spec,log)
        script='extend_shoes_k32.py' if task=='place_dual_shoes' else 'extend_cup_k32.py'
        cmd=[str(SIM_PY),'-u',str(REPO/'experiments/support_three_tasks_20261123'/script),
             '--port',str(spec['port']),'--output-root',str(root),'--policy-checkpoint',str(spec['checkpoint'])]
        if task=='place_dual_shoes':cmd+=['--task',task]
        env=dict(os.environ)
        env['PYTHONPATH']=str(RUNTIME/'XPolicyLab')
        with (log/'extend.log').open('w') as f:
            subprocess.run(cmd,env=env,check=True,stdout=f,stderr=subprocess.STDOUT)
        old=OUT/'matched_sft_data'/task
        manifest=json.loads((log/'manifest.json').read_text())
        assert manifest['complete'] and len(manifest['episodes'])==70
        for e in manifest['episodes']:
            with np.load(log/e['file']) as a,np.load(old/e['file']) as b:
                assert a['policy_chunks'].shape[1]==32
                assert np.array_equal(a['policy_chunks'][:,:4],b['policy_chunks'])
                for k in ['state','features','expert_chunks']:
                    assert np.array_equal(a[k],b[k])
        print(task,'K32 verified all 70',flush=True)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=30)
