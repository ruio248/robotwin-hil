"""Add terminal observation and optional K32 selection to the official loop."""
import sys
import types
from pathlib import Path
import numpy as np

def main():
    mode, output, path, *args = sys.argv[1:]
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    module = types.ModuleType('occupancy_official_eval')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    source = path.read_text()
    anchor = '        if trace is not None:\n            trace.finish_episode(\n                success=succ, rollout_failed=rollout_failed,'
    assert source.count(anchor)==1
    source = source.replace(anchor, '''        if trace is not None and not rollout_failed:
            terminal = robotwin_obs_to_xpolicylab(task_env.get_obs(), task_env=task_env,
                instruction=instruction, frequency=frequency)
            save_terminal(trace, terminal, task_env.step_lim, rollout_steps, succ)
''' + anchor)
    query = '                action_chunk = normalize_action_chunk(model_client.call(func_name="get_action"))'
    update = '                model_client.call(func_name="update_obs", obs=xpl_obs)\n' + query
    assert source.count(update)==1
    source=source.replace(update, '''                xpl_obs["additional_info"].update(occupancy_policy_steps=rollout_steps,
                    occupancy_step_limit=task_env.step_lim)
                if trace is not None and collect_geometry:
                    archive_geometry(trace, task_env, rollout_steps)
''' + update)
    before_terminal = '            save_terminal(trace, terminal, task_env.step_lim, rollout_steps, succ)'
    source=source.replace(before_terminal, '            if collect_geometry:\n                archive_geometry(trace, task_env, rollout_steps)\n'+before_terminal)
    def archive_geometry(trace, env, steps):
        import json
        cup = env.cup.get_functional_point(0, "pose").p.tolist()
        coaster = env.coaster.get_functional_point(0, "pose").p.tolist()
        with (trace.root/trace.active['directory']/'geometry.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(steps=steps,cup=cup,coaster=coaster))+'\n')
    module.archive_geometry=archive_geometry
    module.collect_geometry=mode!='collect'
    def save_terminal(trace, obs, limit, steps, success):
        values = {f'rgb_{name}':data['color'] for name,data in obs['vision'].items() if 'color' in data}
        values.update({f'state_{name}':value for name,value in obs['state'].items()})
        np.savez_compressed(trace.root/trace.active['directory']/'terminal.npz', **values,
                            step_limit=limit, rollout_steps=steps, success=success,
                            done=True, reason='success' if success else 'step_limit')
    module.save_terminal = save_terminal
    exec(compile(source,str(path),'exec'), module.__dict__)
    if mode in ('random','max','min'):
        from selection import install
        install(module, mode, Path(output))
    sys.argv=[str(path),*args]
    from test_render import Sapien_TEST
    Sapien_TEST()
    module.main(module.parse_args())

if __name__=='__main__':
    main()
