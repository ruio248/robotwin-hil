"""Run: python -m unittest discover -s scripts/tests -p 'test_tail_value.py' -v"""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tail_value.cache import (ACTION_SPACE, PROMPT, atomic_json, atomic_npz, assert_compatible, cache_fingerprint,
                              episode_split, load_episode, load_manifest, sha256, transition_mask, validate_episode)
from tail_value.evaluate import report_rows, score_episode, summarize
from tail_value.model import (CoverageCritic, TransitionTable, action_distances, critic_loss, farthest_indices,
                              finite_guard, load_checkpoint, loss_from_scores, make_target, models_from_checkpoint, soft_update)
from tail_value.prepare import first_action, generate_episode, image_tensor, sample_candidates
from tail_value.sources import (CAMERAS, EpisodeSource, JOINT_FIELDS, control_mask, discover, iter_episode)
import eval_tail_value
import tail_data
import train_tail


def arrays(n=5, k=4, hil=False, offset=0.):
    rng = np.random.default_rng(12)
    state = np.arange(n * 14, dtype=np.float32).reshape(n, 14) * .01 + offset
    action = np.concatenate([state[1:], state[-1:]])
    result = {"features": rng.standard_normal((n, 1536)).astype(np.float32), "state": state,
              "a_pi": action[:, None] + rng.standard_normal((n, k, 14)).astype(np.float32) * .1,
              "frame_index": np.arange(n), "valid_transition": transition_mask(n, not hil)}
    if hil:
        result["control_source"] = np.asarray([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int8)
    else:
        result["a_demo"] = action
    return result


def cache(root, spec=(("train", False, 0.), ("val", False, 100.))):
    entries = []
    for i, (split, hil, offset) in enumerate(spec):
        filename = f"episodes/episode_{i}.npz"
        atomic_npz(root / filename, arrays(hil=hil, offset=offset))
        entries.append({"id": f"episode_{i}", "split": split, "kind": "hil" if hil else "demo", "file": filename,
                        "sha256": sha256(root / filename), "metadata": {"supervisor_label": "success"} if hil else {}})
    manifest = {"schema_version": 1, "complete": True, "config": {"source_format": "lerobot",
                "compatibility": {"num_candidates": 4, "test": True}, "transition_semantics": "approximate"}, "episodes": entries}
    atomic_json(root / "manifest.json", manifest)
    return manifest


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.draw = 0
    def call(self, func_name, obs=None):
        self.calls.append((func_name, obs))
        if func_name == "get_action":
            self.draw += 1
            result = np.full((10, 14), 1000., dtype=np.float32)
            result[0] = self.draw
            return result


class CacheTests(unittest.TestCase):
    def test_split_episode_counts_and_seed(self):
        split = episode_split(list(range(450)))
        self.assertEqual(sum(v == "train" for v in split.values()), 405)
        self.assertEqual(sum(v == "val" for v in split.values()), 45)
        self.assertEqual(split, episode_split(list(reversed(range(450)))))
        with self.assertRaises(ValueError):
            episode_split([1])

    def test_cache_roundtrip_corruption_and_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = cache(root)
            self.assertEqual(cache_fingerprint(manifest), cache_fingerprint(load_manifest(root)))
            data = load_episode(root / manifest["episodes"][0]["file"], "demo", 4)
            self.assertFalse(data["valid_transition"][-1])
            data["valid_transition"][-1] = True
            with self.assertRaises(ValueError):
                validate_episode(data, "demo")
            atomic_npz(root / manifest["episodes"][0]["file"], arrays(offset=4.))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                load_manifest(root)

    def test_hil_has_no_expert_targets(self):
        data = arrays(hil=True)
        validate_episode(data, "hil")
        self.assertFalse(data["valid_transition"].any())
        data["a_demo"] = np.zeros((5, 14))
        with self.assertRaises(ValueError):
            validate_episode(data, "hil")
        with self.assertRaises(ValueError):
            control_mask({"control_mask": ["policy"]}, 2)
        with self.assertRaises(ValueError):
            control_mask({"control_mask": ["unknown"]}, 1)

    def test_train_stats_exclude_val_and_terminal_action(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = cache(root)
            table = TransitionTable(root, manifest, "train")
            norm = table.normalization()
            self.assertLess(norm["state_mean"].max(), 1.)
            self.assertTrue((norm["action_scale"] >= .05).all())
            self.assertEqual(len(table), 4)
            b = table.batch(torch.arange(4), "cpu")
            self.assertTrue(torch.allclose(b["a_demo"], b["next_state"]))
            hroot = root / "hil"
            hm = cache(hroot, (("train", True, 0.),))
            with self.assertRaises(ValueError):
                TransitionTable(hroot, hm, "train")

    def test_representation_mismatch(self):
        with self.assertRaises(ValueError):
            assert_compatible({"k": 4}, {"k": 8})

    def test_incomplete_and_nonfinite_caches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = cache(root)
            manifest["complete"] = False
            atomic_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_manifest(root)
        data = arrays()
        data["a_pi"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_episode(data, "demo")

    def test_native_discovery_only_uses_manifest_heldout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            for eid in (0, 450, 499):
                (root / "data" / f"episode_{eid:07d}.hdf5").touch()
            atomic_json(root / "split_manifest_v1.json", {"train_episodes": [0], "validation_episodes": [450, 499]})
            sources = discover(root, "native")
            self.assertEqual([s.metadata["episode_index"] for s in sources], [450, 499])
            self.assertTrue(all(s.split == "heldout" for s in sources))
            atomic_json(root / "split_manifest_v1.json", {"train_episodes": [450], "validation_episodes": [450]})
            with self.assertRaises(ValueError):
                discover(root, "native")


class SamplingTests(unittest.TestCase):
    def test_absolute_and_delta_joint_distances_are_identical(self):
        state = torch.linspace(-0.4, 0.4, 14)[None]
        policy_abs = state[:, None, :] + torch.tensor(
            [[[0.1] * 14, [0.2] * 14, [0.3] * 14, [0.4] * 14]], dtype=torch.float32
        )
        expert_abs = state + 0.15
        scale = torch.full((14,), 0.05)
        absolute = action_distances(policy_abs, expert_abs, scale)
        delta = action_distances(policy_abs - state[:, None, :], expert_abs - state, scale)
        self.assertEqual(ACTION_SPACE, "absolute_joint_qpos")
        self.assertTrue(torch.allclose(absolute, delta))

    def test_same_obs_independent_calls_first_action_only(self):
        frame = {"state": np.zeros(14), "images": {c: np.zeros((8, 12, 3), dtype=np.uint8) for c in CAMERAS}}
        client = FakePolicy()
        candidates = sample_candidates(client, frame, 4)
        self.assertEqual(candidates.shape, (4, 14))
        np.testing.assert_array_equal(candidates[:, 0], [1, 2, 3, 4])
        self.assertEqual([c[0] for c in client.calls], ["update_obs"] + ["get_action"] * 4)
        self.assertIs(client.calls[0][1]["images"], frame["images"])

    def test_runtime_singular_action_fields(self):
        values = np.arange(14, dtype=np.float32)
        action = {key.removesuffix("s"): value for key, value in zip(JOINT_FIELDS, np.split(values, [6, 7, 13]), strict=True)}
        np.testing.assert_array_equal(first_action([action, action]), values)
        for bad in (np.ones(14), np.ones((10, 32)), np.full((10, 14), np.nan)):
            with self.assertRaises(ValueError):
                first_action(bad)

    def test_rgb_padding_and_channel_order(self):
        red = np.zeros((120, 240, 3), dtype=np.uint8)
        red[..., 0] = 255
        value = image_tensor(red)
        self.assertEqual(value.shape, (3, 224, 224))
        self.assertGreater(value[0, 112, 112], value[2, 112, 112])
        self.assertAlmostEqual(value[0, 0, 0].item(), -.485 / .229, places=5)

    def test_generate_hil_never_derives_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "collection/raw/episode_0000000"
            (root / "frames").mkdir(parents=True)
            atomic_json(root / "episode.json", {"instruction": PROMPT, "control_mask": ["policy", "hil"], "supervisor_label": "success"})
            for i in range(2):
                frame = {"joint_action": {"vector": np.full(14, i)}, "observation": {
                    name: {"rgb": np.zeros((8, 12, 3), np.uint8)} for name in ("head_camera", "left_camera", "right_camera")}}
                with (root / "frames" / f"{i}.pkl").open("wb") as stream:
                    pickle.dump(frame, stream)
            source = discover(root, "hil")[0]
            data = generate_episode(source, FakePolicy(), lambda images: np.zeros(1536), 4)
            self.assertNotIn("a_demo", data)
            self.assertFalse(data["valid_transition"].any())
            np.testing.assert_array_equal(data["control_source"], [0, 1])

    def test_lerobot_row_video_alignment(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episode.parquet"
            table = {"observation.state": [[0.] * 14, [1.] * 14], "action": [[1.] * 14] * 2,
                     "frame_index": [0, 1], "episode_index": [0, 0], "task_index": [0, 0], "timestamp": [0., 1 / 15]}
            pq.write_table(pa.table(table), path)
            source = EpisodeSource("e", "train", "demo", "lerobot", [path] * 4,
                                   {"length": 2, "episode_index": 0, "fps": 15, "task_indices": [0], "camera_shapes": [[3, 4, 6]] * 3})
            @contextmanager
            def reader(*args):
                yield iter([np.zeros((4, 6, 3), np.uint8)] * 2)
            with patch("tail_value.sources.video_frames", reader):
                self.assertEqual(len(list(iter_episode(source))), 2)
            @contextmanager
            def short_reader(*args):
                yield iter([np.zeros((4, 6, 3), np.uint8)])
            @contextmanager
            def long_reader(*args):
                yield iter([np.zeros((4, 6, 3), np.uint8)] * 3)
            for bad_reader in (short_reader, long_reader):
                with patch("tail_value.sources.video_frames", bad_reader), self.assertRaises(ValueError):
                    list(iter_episode(source))
            table["frame_index"] = [0, 2]
            pq.write_table(pa.table(table), path)
            with patch("tail_value.sources.video_frames", reader), self.assertRaises(ValueError):
                list(iter_episode(source))

    def test_resume_skips_policy_and_rejects_config_change(self):
        import argparse
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy_path = root / "policy.yml"
            policy_path.write_text("policy_name: Pi_05_RobotTwin\naction_type: joint\n")
            source_path = root / "source"
            source_path.write_text("fixture")
            source = EpisodeSource("e", "train", "demo", "lerobot", [source_path], {})
            args = argparse.Namespace(source_format="lerobot", dataset_root=root, policy_url="ws://127.0.0.1:18301",
                       policy_config=policy_path, output_dir=root / "cache", num_candidates=4, seed=42,
                       episode_limit=2, encoder_weights=None, device="cpu", resume=False)
            import types
            module = types.ModuleType("client_server.ws.model_client")
            class FakeContext:
                def __init__(self, **kwargs):
                    pass
                def __enter__(self):
                    return FakePolicy()
                def __exit__(self, *args):
                    pass
            module.WsModelClient = FakeContext
            with patch("tail_data.discover", return_value=[source]), patch.dict(sys.modules, {"client_server.ws.model_client": module}), \
                 patch("tail_value.prepare.FrozenEncoder"), patch("tail_value.prepare.generate_episode", return_value=arrays()) as generate:
                tail_data.run(args)
                args.resume = True
                tail_data.run(args)
                self.assertEqual(generate.call_count, 1)
                args.num_candidates = 8
                with self.assertRaises(ValueError):
                    tail_data.run(args)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        norm = {name: torch.zeros(14) if name.endswith("mean") else torch.ones(14)
                for name in ("state_mean", "action_mean", "state_scale", "action_scale")}
        torch.manual_seed(7)
        self.model = CoverageCritic(norm, 16)
        self.target = make_target(self.model)

    def test_farthest_normalized_not_largest_raw_distance(self):
        candidates = torch.zeros(1, 2, 14)
        candidates[0, 0, 0] = 2
        candidates[0, 1, 1] = 1
        scale = torch.ones(14)
        scale[0] = 100
        self.assertEqual(farthest_indices(candidates, torch.zeros(1, 14), scale).item(), 1)

    def test_numeric_losses_and_regularizer_all_candidates(self):
        q_e, q_p, target, far = torch.tensor([2.]), torch.tensor([[1., 5.]]), torch.tensor([3.]), torch.tensor([1])
        options = dict(alpha=.1, output_reg=.01)
        original, _ = loss_from_scores(q_e, q_p, target, far, mode="original", reduction="mean", **options)
        mean, m = loss_from_scores(q_e, q_p, target, far, mode="stabilized", reduction="mean", **options)
        hardest, h = loss_from_scores(q_e, q_p, target, far, mode="stabilized", reduction="farthest", **options)
        self.assertAlmostEqual(original.item(), 1.1, places=6)
        self.assertAlmostEqual(mean.item(), 1.185, places=6)
        self.assertAlmostEqual(hardest.item(), 1.385, places=6)
        self.assertEqual(m["regularizer"], h["regularizer"])

    def test_target_is_unchanged_and_has_no_gradient(self):
        data = arrays()
        batch = {k: torch.from_numpy(data[k][:-1]) for k in ("features", "state", "a_demo", "a_pi")}
        batch.update({"next_" + k: torch.from_numpy(data[k][1:]) for k in ("features", "state", "a_pi")})
        targets = []
        for reduction in ("mean", "farthest"):
            loss, _, values = critic_loss(self.model, self.target, batch, gamma=.99, alpha=.01,
                                          output_reg=.0001, reduction=reduction, mode="stabilized")
            targets.append(values["target"])
            loss.backward()
        torch.testing.assert_close(*targets)
        self.assertTrue(all(p.grad is None for p in self.target.parameters()))
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()))

    def test_ema_and_guards(self):
        before = next(self.target.parameters()).clone()
        with torch.no_grad():
            next(self.model.parameters()).add_(1)
        soft_update(self.target, self.model, .1)
        torch.testing.assert_close(next(self.target.parameters()), before + .1)
        for bad in (torch.tensor(float("nan")), torch.tensor(1e5)):
            with self.assertRaises(FloatingPointError):
                finite_guard({"score": bad}, 1e4)

    def test_hil_evaluation_no_td_and_no_label_input(self):
        data = arrays(hil=True)
        first = score_episode(self.model, self.target, data, "hil", .99, 2, "cpu", 1e4)
        data["control_source"] = 1 - data["control_source"]
        second = score_episode(self.model, self.target, data, "hil", .99, 2, "cpu", 1e4)
        np.testing.assert_array_equal(first["q_pi"], second["q_pi"])
        self.assertNotIn("td_target", first)
        self.assertIsNone(summarize([first])["td_mse"])

    def test_last_demo_frame_has_no_td_target(self):
        data = arrays()
        result = score_episode(self.model, self.target, data, "demo", .99, 2, "cpu", 1e4)
        self.assertTrue(np.isnan(result["td_target"][-1]))
        self.assertTrue(np.isfinite(result["td_target"][:-1]).all())


class EndToEndTests(unittest.TestCase):
    def test_divergence_records_failure_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache(root / "cache")
            args = train_tail.parse_args(["--cache-dir", str(root / "cache"), "--output-dir", str(root / "run"),
                   "--device", "cpu", "--cpu-threads", "1", "--width", "16", "--steps", "1", "--score-limit", "1e-12"])
            with patch("sys.stdout", new_callable=io.StringIO), self.assertRaises(FloatingPointError):
                train_tail.run(args)
            self.assertTrue((root / "run/failure.json").is_file())
            self.assertFalse((root / "run/last.pt").exists())

    def test_cpu_resume_matches_uninterrupted_and_all_suite_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "cache"
            cache(data)
            base = ["--cache-dir", str(data), "--device", "cpu", "--cpu-threads", "1", "--width", "16",
                    "--batch-size", "3", "--eval-every", "1", "--log-every", "1"]
            with patch("sys.stdout", new_callable=io.StringIO):
                train_tail.run(train_tail.parse_args(base + ["--output-dir", str(root / "full"), "--steps", "4"]))
                train_tail.run(train_tail.parse_args(base + ["--output-dir", str(root / "resume"), "--steps", "2"]))
                train_tail.run(train_tail.parse_args(base + ["--output-dir", str(root / "resume"), "--steps", "4",
                                                            "--resume", str(root / "resume/last.pt")]))
            full, resumed = load_checkpoint(root / "full/last.pt"), load_checkpoint(root / "resume/last.pt")
            for key in full["model"]:
                torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
            cache(root / "heldout", (("heldout", False, 0.),))
            cache(root / "hil", (("hil", True, 0.),))
            args = eval_tail_value.parse_args(["--cache-dir", str(data), str(root / "heldout"), str(root / "hil"),
                    "--checkpoint", str(root / "full/last.pt"), str(root / "resume/last.pt"),
                    "--output-dir", str(root / "report"), "--suite", "all", "--device", "cpu", "--cpu-threads", "1", "--max-plots", "1"])
            with patch("sys.stdout", new_callable=io.StringIO):
                metrics = eval_tail_value.run(args)
            self.assertEqual(len(metrics), 2)
            for values in metrics.values():
                self.assertEqual(set(values["suites"]), {"val", "heldout", "hil"})
                self.assertIsNone(values["suites"]["hil"]["td_mse"])
            import pyarrow.parquet as pq
            rows = pq.read_table(root / "report/scores.parquet").to_pylist()
            self.assertTrue(all(row["q_demo"] is None and row["td_target"] is None for row in rows if row["suite"] == "hil"))
            self.assertGreater(len(list((root / "report").rglob("*.png"))), 3)
            with self.assertRaises(FileExistsError):
                eval_tail_value.run(args)


if __name__ == "__main__":
    unittest.main()
