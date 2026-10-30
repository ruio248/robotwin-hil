"""Exercise the actual synchronous client with a delayed async RPC transport."""
from __future__ import annotations
import ast
import asyncio
from collections.abc import Coroutine
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager, nullcontext
import math
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any
import unittest


def client_class():
    source = Path(__file__).resolve().parents[2] / "XPolicyLab/client_server/ws/model_client.py"
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WsModelClient"]
    namespace = dict(globals())
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["WsModelClient"]


class Transport:
    def __init__(self):
        self.events = []
        self.error = None

    async def connect(self, **kwargs): pass
    async def close(self): pass

    async def call(self, name, payload, **kwargs):
        self.events.append(("start", kwargs["step"]))
        await asyncio.sleep(0.06)
        self.events.append(("end", kwargs["step"]))
        if self.error:
            raise self.error
        return SimpleNamespace(payload={"result": payload})


class ResponsiveClientTests(unittest.TestCase):
    def make_client(self):
        transport = Transport()
        client = client_class()(url="unused", evaluation_id="test", trial_id="test", client=transport)
        self.addCleanup(client.close)
        return client, transport

    def test_ui_runs_on_main_thread_and_requests_stay_sequential(self):
        client, transport = self.make_client()
        original_responsive_wait = client.responsive_wait
        client.responsive_wait = lambda callback, **kwargs: original_responsive_wait(callback, **kwargs)
        ticks = []
        main_thread = threading.get_ident()
        with client.responsive_wait(lambda: ticks.append(threading.get_ident()), interval_s=0.005):
            self.assertEqual(client.call("get_action_candidates", 32), 32)
        self.assertGreater(len(ticks), 2)
        self.assertEqual(set(ticks), {main_thread})
        self.assertIsNone(client._wait_callback)
        self.assertEqual(transport.events, [("start", 0), ("end", 0)])

    def test_takeover_discards_response_after_drain_and_keeps_painting(self):
        client, transport = self.make_client()
        ticks = []
        class Takeover(Exception): pass
        def poll():
            ticks.append(len(ticks))
            if len(ticks) == 2:
                raise Takeover("i")
        with self.assertRaisesRegex(Takeover, "i"):
            with client.responsive_wait(poll, interval_s=0.005):
                client.call("get_action_candidates", 32)
        self.assertGreater(len(ticks), 3)
        self.assertEqual(transport.events[-1], ("end", 0))
        self.assertEqual(client.call("get_action", 1), 1)
        self.assertEqual(transport.events, [("start", 0), ("end", 0), ("start", 1), ("end", 1)])

    def test_rpc_timeout_is_not_retried_as_ui_poll_timeout(self):
        client, transport = self.make_client()
        transport.error = TimeoutError("RPC timeout")
        with self.assertRaisesRegex(TimeoutError, "RPC timeout"):
            with client.responsive_wait(lambda: None, interval_s=0.005):
                client.call("get_action_candidates", 32)
        self.assertEqual(transport.events, [("start", 0), ("end", 0)])
        self.assertIsNone(client._wait_callback)

    def test_hil_sampling_functions_actually_enable_responsive_wait(self):
        source = Path(__file__).resolve().parents[1] / "hg_dagger_handover.py"
        tree = ast.parse(source.read_text())
        functions = {n.name: n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name in {"sample_chunk", "sample_candidates"}}
        self.assertEqual(set(functions), {"sample_chunk", "sample_candidates"})
        client, transport = self.make_client()
        original_responsive_wait = client.responsive_wait
        client.responsive_wait = lambda callback: original_responsive_wait(callback, interval_s=0.001)
        ticks = []
        def poll():
            ticks.append(threading.get_ident())
        def single(model, observation, action_type):
            return model.call("get_action", 1)
        def candidates(model, observation, action_type, count, batch):
            self.assertEqual((count, batch), (32, 32))
            return model.call("get_action_candidates", count)
        namespace = dict(model_client=client, observation={}, poll_live_control=poll,
                         nullcontext=nullcontext,
                         sampling_config=SimpleNamespace(policy_microbatch=32),
                         sample_absolute_joint_chunk=single, sample_absolute_joint_candidates=candidates)
        for function in functions.values():
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        for name, args, expected in (("sample_chunk", (), 1), ("sample_candidates", (32,), 32)):
            ticks.clear()
            self.assertEqual(namespace[name](*args), expected)
            self.assertGreaterEqual(len(ticks), 3)  # before, during RPC, after
            self.assertEqual(set(ticks), {threading.get_ident()})
            self.assertIsNone(client._wait_callback)

        class Takeover(Exception): pass
        def take_over_during_wait():
            if client._wait_callback is not None:
                raise Takeover("discard")
        namespace["poll_live_control"] = take_over_during_wait
        with self.assertRaisesRegex(Takeover, "discard"):
            namespace["sample_candidates"](32)
        self.assertEqual(transport.events[-1], ("end", 2))
        self.assertIsNone(client._wait_callback)


if __name__ == "__main__":
    unittest.main()
