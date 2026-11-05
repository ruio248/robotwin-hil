import importlib.util
import json
import pickle
from pathlib import Path
import sys
import types
from unittest.mock import patch

import h5py
import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[1] / "export_hg_dagger_dataset.py"
fake_envs = types.ModuleType("envs")
fake_utils = types.ModuleType("envs.utils")
fake_convert = types.ModuleType("envs.utils.pkl2hdf5")
fake_convert.process_folder_to_hdf5_video = lambda *_args, **_kwargs: None
with patch.dict(sys.modules, {
    "envs": fake_envs,
    "envs.utils": fake_utils,
    "envs.utils.pkl2hdf5": fake_convert,
}):
    spec = importlib.util.spec_from_file_location("export_hg_dagger_dataset", MODULE)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)


def test_headless_smoke_metadata_is_excluded(tmp_path):
    episode = tmp_path / "episode_0000000"
    episode.mkdir()
    assert exporter.is_diagnostic_episode(episode, {
        "episode_metadata": {"diagnostic_only": True},
        "record": {"accepted_for_training": False},
    })
    assert exporter.is_diagnostic_episode(episode, {
        "record": {"diagnostic_only": "true"},
    })


def test_legacy_test_seed_marker_is_excluded(tmp_path):
    episode = tmp_path / "episode_0000000"
    episode.mkdir()
    (episode / "DIAGNOSTIC_ONLY.json").write_text("{}")
    assert exporter.is_diagnostic_episode(episode, {})


def test_training_episode_remains_exportable(tmp_path):
    episode = tmp_path / "episode_0000000"
    episode.mkdir()
    assert not exporter.is_diagnostic_episode(episode, {
        "episode_metadata": {"diagnostic_only": False},
        "record": {"accepted_for_training": True},
    })
    assert not exporter.is_diagnostic_episode(episode, {})


def test_export_main_skips_diagnostic_but_keeps_formal(tmp_path):
    raw_root = tmp_path / "raw"
    for index, diagnostic in enumerate((True, False)):
        episode = raw_root / f"episode_{index:07d}"
        (episode / "frames").mkdir(parents=True)
        for frame in range(3):
            (episode / "frames" / f"{frame}.pkl").write_bytes(bytes([frame]))
        (episode / "episode.json").write_text(json.dumps({
            "episode_index": index,
            "instruction": "test",
            "control_mask": ["policy", "hil", "hil"],
            "episode_metadata": {"diagnostic_only": diagnostic},
            "record": {"accepted_for_training": not diagnostic},
        }))
    output = tmp_path / "export"
    calls = []
    def convert(folder, hdf5, _video, **_kwargs):
        calls.append((folder, hdf5))
        Path(hdf5).write_bytes(b"converted")
        return 2

    with patch.object(exporter, "process_folder_to_hdf5_video", side_effect=convert), \
         patch.object(sys, "argv", ["export", "--raw-root", str(raw_root), "--output-dir", str(output)]):
        assert exporter.main() == 0
    assert len(calls) == 1
    assert calls[0][0].endswith("episode_0000001/frames")
    manifest = next(output.glob("manifest_full_*.json"))
    assert [episode["episode_index"] for episode in json.loads(manifest.read_text())["episodes"]] == [1]


def test_hil_export_keeps_two_takeovers_separate_without_cross_segment_action(tmp_path):
    raw_root = tmp_path / "raw"
    episode = raw_root / "episode_0000007"
    frames = episode / "frames"
    frames.mkdir(parents=True)
    mask = ["policy", "hil", "hil", "hil", "policy", "hil", "hil", "hil"]
    for index in range(len(mask)):
        (frames / f"{index}.pkl").write_bytes(bytes([index]))
    (episode / "episode.json").write_text(json.dumps({
        "episode_index": 7, "seed": 48302, "instruction": "stack red, green, blue",
        "control_mask": mask, "record": {"accepted_for_training": True},
    }))
    output = tmp_path / "export"
    staged = []

    def inspect_staging(folder, hdf5, _video, **kwargs):
        selected = sorted(Path(folder).glob("*.pkl"), key=lambda item: int(item.stem))
        staged.append(([path.read_bytes()[0] for path in selected], kwargs["episode_metadata"]))
        Path(hdf5).write_bytes(b"converted")
        return len(selected) - 1

    with patch.object(exporter, "process_folder_to_hdf5_video", side_effect=inspect_staging), \
         patch.object(sys, "argv", ["export", "--raw-root", str(raw_root),
                                   "--output-dir", str(output), "--mode", "hil"]):
        assert exporter.main() == 0
    assert [values for values, _ in staged] == [[1, 2, 3], [5, 6, 7]]
    assert [metadata["transition_count"] for _, metadata in staged] == [2, 2]
    assert [metadata["source_frame_start"] for _, metadata in staged] == [1, 5]
    manifest = next(output.glob("manifest_hil_*.json"))
    rows = json.loads(manifest.read_text())["episodes"]
    assert [row["transition_count"] for row in rows] == [2, 2]
    assert [Path(row["hdf5"]).name for row in rows] == [
        "episode_0000007_hil_s00.hdf5", "episode_0000007_hil_s01.hdf5"]
    assert all(row["seed"] == 48302 and row["instruction"] == "stack red, green, blue" for row in rows)


def test_hil_export_refuses_mask_frame_mismatch_before_conversion(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "0.pkl").write_bytes(b"0")
    (frames / "1.pkl").write_bytes(b"1")
    try:
        exporter.validated_frame_count(frames, ["hil"])
    except ValueError as error:
        assert "control_mask/frame alignment" in str(error)
    else:
        raise AssertionError("Mismatched mask must not be exported")


def test_full_export_retains_legacy_episode_without_control_mask(tmp_path):
    raw_root = tmp_path / "raw"
    episode = raw_root / "episode_0000000"
    frames = episode / "frames"
    frames.mkdir(parents=True)
    for index in range(2):
        (frames / f"{index}.pkl").write_bytes(bytes([index]))
    (episode / "episode.json").write_text(json.dumps({
        "episode_index": 0, "instruction": "legacy episode",
    }))
    output = tmp_path / "export"
    calls = []
    def convert(folder, hdf5, _video, **_kwargs):
        calls.append((folder, hdf5))
        Path(hdf5).write_bytes(b"converted")
        return 1

    with patch.object(exporter, "process_folder_to_hdf5_video", side_effect=convert), \
         patch.object(sys, "argv", ["export", "--raw-root", str(raw_root), "--output-dir", str(output)]):
        assert exporter.main() == 0
    assert len(calls) == 1
    assert calls[0][0] == str(frames)
    assert (frames / "0.pkl").is_file() and (frames / "1.pkl").is_file()


def test_hil_export_discards_one_transition_segment_at_shared_default():
    assert list(exporter.hil_runs(["policy", "hil", "hil", "policy", "hil", "hil", "hil"])) == [(4, 7)]


def test_conversion_failure_leaves_no_published_files(tmp_path):
    raw_root = tmp_path / "raw"
    frames = raw_root / "episode_0000000" / "frames"
    frames.mkdir(parents=True)
    for index in range(3):
        (frames / f"{index}.pkl").write_bytes(bytes([index]))
    (frames.parent / "episode.json").write_text(json.dumps({
        "episode_index": 0, "instruction": "stack", "control_mask": ["hil"] * 3,
    }))
    output = tmp_path / "export"

    def fail(_folder, hdf5, _video, **_kwargs):
        Path(hdf5).write_bytes(b"incomplete")
        raise RuntimeError("conversion interrupted")

    with patch.object(exporter, "process_folder_to_hdf5_video", side_effect=fail), \
         patch.object(sys, "argv", ["export", "--raw-root", str(raw_root),
                                   "--output-dir", str(output), "--mode", "hil"]):
        with pytest.raises(RuntimeError, match="conversion interrupted"):
            exporter.main()
    assert not list(output.rglob("*.hdf5"))
    assert not list(output.rglob("*.json"))
    assert [path.read_bytes() for path in sorted(frames.glob("*.pkl"))] == [bytes([i]) for i in range(3)]


def test_real_hdf5_conversion_has_only_within_segment_actions(tmp_path):
    overlay = MODULE.parents[1]
    video_module = types.ModuleType("envs.utils.images_to_video")
    video_module.images_to_video = lambda *_args, **_kwargs: None
    envs_module = types.ModuleType("envs")
    envs_module.__path__ = [str(overlay / "envs")]
    utils_module = types.ModuleType("envs.utils")
    utils_module.__path__ = [str(overlay / "envs" / "utils")]
    converter_path = overlay / "envs" / "utils" / "pkl2hdf5.py"
    with patch.dict(sys.modules, {
        "envs": envs_module, "envs.utils": utils_module,
        "envs.utils.images_to_video": video_module,
    }):
        spec = importlib.util.spec_from_file_location("envs.utils.pkl2hdf5", converter_path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)

    raw_root = tmp_path / "raw"
    frames = raw_root / "episode_0000007" / "frames"
    frames.mkdir(parents=True)
    mask = ["policy", "hil", "hil", "hil", "policy", "hil", "hil", "hil"]
    for index in range(len(mask)):
        record = {
            "joint_action": {
                "left_arm": np.full(6, index, dtype=np.float32),
                "left_gripper": np.array([index], dtype=np.float32),
                "right_arm": np.full(6, index + 20, dtype=np.float32),
                "right_gripper": np.array([index + 20], dtype=np.float32),
            },
            "observation": {"head_camera": {"rgb": np.full((4, 4, 3), index, dtype=np.uint8)}},
        }
        (frames / f"{index}.pkl").write_bytes(pickle.dumps(record))
    (frames.parent / "episode.json").write_text(json.dumps({
        "episode_index": 7, "seed": 48302, "instruction": "stack red green blue",
        "control_mask": mask, "record": {"accepted_for_training": True},
    }))
    output = tmp_path / "export"
    with patch.object(exporter, "process_folder_to_hdf5_video", converter.process_folder_to_hdf5_video), \
         patch.object(sys, "argv", ["export", "--raw-root", str(raw_root),
                                   "--output-dir", str(output), "--mode", "hil"]):
        assert exporter.main() == 0

    expected = [([1, 2], [2, 3], 1, 4), ([5, 6], [6, 7], 5, 8)]
    for segment_index, (states, actions, start, end) in enumerate(expected):
        path = output / "data" / f"episode_0000007_hil_s{segment_index:02d}.hdf5"
        with h5py.File(path, "r") as data:
            np.testing.assert_array_equal(data["state/left_arm_joint_states"][:, 0], states)
            np.testing.assert_array_equal(data["action/left_arm_joint_states"][:, 0], actions)
            assert len(data["vision/cam_head/colors"]) == 2
            assert json.loads(data["instructions"][()]) == ["stack red green blue"]
            metadata = json.loads(data["additional_info/episode_metadata_json"][()])
            assert metadata["source_seed"] == 48302
            assert (metadata["source_frame_start"], metadata["source_frame_end_exclusive"]) == (start, end)
            assert metadata["transition_count"] == 2
    assert len(list(frames.glob("*.pkl"))) == len(mask)
