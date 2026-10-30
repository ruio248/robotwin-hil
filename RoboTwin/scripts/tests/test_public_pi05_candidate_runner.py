import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "run_public_pi05_candidate.py"
spec = importlib.util.spec_from_file_location("run_public_pi05_candidate", MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_only_pre_policy_setup_unstable_error_is_replaceable():
    setup_error = """Traceback (most recent call last):
  File "eval.py", line 93, in main
    task_env.setup_demo(now_ep_num=0, seed=31028, is_test=False, **args)
  File "_base_task.py", line 158, in _init_task_env_
    raise UnStableError(
envs.utils.create_actor.UnStableError: Objects is unstable in seed(31028), unstable objects: 105_sauce-can
"""
    assert runner.is_unstable_setup_log(setup_error)
    assert not runner.is_unstable_setup_log(setup_error.replace(
        "task_env.setup_demo(", "task_env.take_action("
    ))
    assert not runner.is_unstable_setup_log(setup_error.replace(
        "UnStableError:", "RuntimeError:"
    ))
    assert not runner.is_unstable_setup_log(setup_error + "task_env.take_action(")
