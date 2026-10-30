import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch


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
        (episode / "episode.json").write_text(json.dumps({
            "episode_index": index,
            "instruction": "test",
            "episode_metadata": {"diagnostic_only": diagnostic},
            "record": {"accepted_for_training": not diagnostic},
        }))
    output = tmp_path / "export"
    calls = []
    with patch.object(exporter, "process_folder_to_hdf5_video", side_effect=lambda *args, **kwargs: calls.append(args)), \
         patch.object(sys, "argv", ["export", "--raw-root", str(raw_root), "--output-dir", str(output)]):
        assert exporter.main() == 0
    assert len(calls) == 1
    assert calls[0][0].endswith("episode_0000001/frames")
    manifest = next(output.glob("manifest_full_*.json"))
    assert [episode["episode_index"] for episode in json.loads(manifest.read_text())["episodes"]] == [1]
