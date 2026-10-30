"""Read-only readers for the three explicitly supported RoboTwin sources."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import tempfile

import numpy as np

from .cache import CAMERAS, PROMPT, episode_split, read_json, sha256

JOINT_FIELDS = ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")
NATIVE_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
RAW_CAMERAS = ("head_camera", "left_camera", "right_camera")


def add_xpolicylab_path():
    # Override only for isolated tests; normal scripts find the enclosing RoboTwin.
    root = Path(os.environ.get("ROBOTWIN_ROOT", Path(__file__).resolve().parents[2]))
    for path in (root, root / "XPolicyLab"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def rgb_image(image):
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError(f"Expected HWC RGB uint8, got {value.shape}/{value.dtype}")
    return value


@dataclass
class EpisodeSource:
    id: str
    split: str
    kind: str
    format: str
    files: list[Path]
    metadata: dict

    def fingerprint(self):
        return [{"path": str(p.resolve()), "sha256": sha256(p)} for p in self.files]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def discover(root, source_format, limit=None, seed=42):
    root = Path(root).resolve()
    if limit is not None and limit < 1:
        raise ValueError("episode-limit must be positive")
    if source_format == "lerobot":
        info = read_json(root / "meta/info.json")
        if info.get("codebase_version") != "v2.1" or info.get("fps", 0) <= 0:
            raise ValueError("Expected LeRobot v2.1 with positive FPS")
        for key in ("observation.state", "action"):
            if info["features"].get(key, {}).get("shape") != [14]:
                raise ValueError(f"Expected 14D {key}")
        tasks = {int(row["task_index"]): row["task"] for row in read_jsonl(root / "meta/tasks.jsonl")}
        if set(tasks.values()) != {PROMPT}:
            raise ValueError("Expected the prompt-fixed single handover task")
        rows = sorted(read_jsonl(root / "meta/episodes.jsonl"), key=lambda r: r["episode_index"])
        if len(rows) != info["total_episodes"] or len({r["episode_index"] for r in rows}) != len(rows):
            raise ValueError("Episode metadata count/IDs mismatch")
        rows = rows[:limit]
        split = episode_split([r["episode_index"] for r in rows], seed)
        sources = []
        for row in rows:
            eid = int(row["episode_index"])
            fmt = {"episode_index": eid, "episode_chunk": eid // int(info["chunks_size"])}
            parquet = root / info["data_path"].format(**fmt)
            videos = [root / info["video_path"].format(**fmt, video_key="observation.images." + c) for c in CAMERAS]
            metadata = {"episode_index": eid, "length": int(row["length"]), "fps": float(info["fps"]),
                        "task_indices": sorted(tasks), "camera_shapes": [info["features"]["observation.images." + c]["shape"] for c in CAMERAS]}
            sources.append(EpisodeSource(f"episode_{eid:07d}", split[str(eid)], "demo", source_format,
                                         [parquet, *videos, root / "meta/info.json", root / "meta/tasks.jsonl", root / "meta/episodes.jsonl"], metadata))
        return sources
    if source_format == "native":
        manifest_path = root / "split_manifest_v1.json"
        split = read_json(manifest_path)
        heldout = split["validation_episodes"]
        if not heldout or len(set(heldout)) != len(heldout) or set(heldout) & set(split["train_episodes"]):
            raise ValueError("Empty/duplicate native heldout episodes or heldout/train overlap")
        sources = []
        for eid in heldout[:limit]:
            paths = [p for p in (root / "data").glob("episode_*.hdf5") if int(p.stem.rsplit("_", 1)[1]) == eid]
            if len(paths) != 1:
                raise ValueError(f"Expected one native HDF5 for episode {eid}")
            sources.append(EpisodeSource(f"episode_{eid:07d}", "heldout", "demo", source_format,
                                         [paths[0], manifest_path], {"episode_index": eid, "prompt_override": PROMPT}))
        return sources
    if source_format != "hil":
        raise ValueError(f"Unknown source format: {source_format}")
    # A collection root, raw root, episode root or outputs root is accepted.
    if (root / "episode.json").is_file():
        paths = [root / "episode.json"]
    else:
        patterns = ("raw/episode_*/episode.json", "episode_*/episode.json", "hg_dagger_collection_r*/raw/episode_*/episode.json")
        paths = sorted({p for pattern in patterns for p in root.glob(pattern)})
    sources = []
    for path in paths[:limit]:
        metadata = read_json(path)
        if metadata.get("instruction") != PROMPT:
            raise ValueError(f"HIL instruction does not match this single-task critic: {path}")
        frames = sorted((path.parent / "frames").glob("*.pkl"), key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))
        if not frames:
            raise ValueError(f"No HIL frames: {path.parent}")
        eid = path.parent.parent.parent.name + "__" + path.parent.name
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", eid):
            raise ValueError("Unsafe episode identifier")
        sources.append(EpisodeSource(eid, "hil", "hil", source_format, [path, *frames], metadata))
    if not sources:
        raise ValueError(f"No episodes found under {root}")
    return sources


@contextmanager
def video_frames(path, height, width):
    """Stream presentation-order RGB frames without FPS resampling or random seeks."""
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
                                 "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                                stdout=subprocess.PIPE, stderr=errors)
        def frames():
            size = height * width * 3
            while True:
                payload = bytearray()
                while len(payload) < size:
                    part = proc.stdout.read(size - len(payload))
                    if not part:
                        break
                    payload.extend(part)
                if not payload:
                    break
                if len(payload) != size:
                    raise ValueError(f"Truncated RGB frame: {path}")
                yield np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            if proc.wait(timeout=30):
                errors.seek(0)
                raise ValueError(f"FFmpeg failed: {errors.read().decode(errors='replace')}")
        try:
            yield frames()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()


def control_mask(metadata, length):
    values = metadata.get("control_mask")
    if not isinstance(values, list) or len(values) != length:
        raise ValueError("HIL control_mask must have one entry per saved frame")
    mapping = {"policy": 0, "hil": 1, "expert": 1}
    if any(not isinstance(v, str) or v not in mapping for v in values):
        raise ValueError("Unknown HIL control source")
    return np.asarray([mapping[v] for v in values], dtype=np.int8)


def iter_episode(source):
    """Yield state, RGB views, optional expert action; metadata never enters critic."""
    if source.format == "lerobot":
        import pyarrow.parquet as pq
        table = pq.read_table(source.files[0]).to_pydict()
        state = np.asarray(table["observation.state"], dtype=np.float32)
        actions = np.asarray(table["action"], dtype=np.float32)
        n = source.metadata["length"]
        if state.shape != (n, 14) or actions.shape != state.shape:
            raise ValueError("Parquet state/action shape or episode length mismatch")
        if table["frame_index"] != list(range(n)) or set(table["episode_index"]) != {source.metadata["episode_index"]}:
            raise ValueError("Non-adjacent frame indices or mixed episodes")
        if set(table["task_index"]) - set(source.metadata["task_indices"]):
            raise ValueError("Unknown task index")
        if not np.allclose(table["timestamp"], np.arange(n) / source.metadata["fps"], atol=1e-4, rtol=1e-5):
            raise ValueError("Frame timestamps do not match declared FPS")
        with ExitStack() as stack:
            readers = []
            for path, shape in zip(source.files[1:4], source.metadata["camera_shapes"], strict=True):
                if len(shape) != 3 or shape[0] != 3:
                    raise ValueError("Expected CHW camera metadata")
                readers.append(stack.enter_context(video_frames(path, shape[1], shape[2])))
            for i in range(n):
                try:
                    images = {c: rgb_image(next(r)) for c, r in zip(CAMERAS, readers, strict=True)}
                except StopIteration as exc:
                    raise ValueError("Video ended before the last parquet row") from exc
                yield {"frame_index": i, "state": state[i], "a_demo": actions[i], "images": images}
            for reader in readers:
                if next(reader, None) is not None:
                    raise ValueError("Video has more frames than parquet")
        return
    if source.format == "native":
        import h5py
        add_xpolicylab_path()
        from XPolicyLab.utils.process_data import decode_image_bit
        with h5py.File(source.files[0], "r") as handle:
            raw_metadata = handle["additional_info/episode_metadata_json"][()]
            metadata = json.loads(raw_metadata.decode() if isinstance(raw_metadata, bytes) else str(raw_metadata))
            if not metadata.get("success") or not metadata.get("plan_success"):
                raise ValueError("Native heldout record is not a successful planned demonstration")
            def pack(group):
                return np.concatenate([np.asarray(handle[f"{group}/{key}"]).reshape(len(handle[f"{group}/{key}"]), -1) for key in JOINT_FIELDS], axis=1).astype(np.float32)
            state, actions = pack("state"), pack("action")
            if state.shape != actions.shape or state.shape[1] != 14:
                raise ValueError("Native state/action shape mismatch")
            images = [handle[f"vision/{c}/colors"] for c in NATIVE_CAMERAS]
            if any(len(x) != len(state) for x in images):
                raise ValueError("Native camera/state frame count mismatch")
            for i in range(len(state)):
                yield {"frame_index": i, "state": state[i], "a_demo": actions[i],
                       "images": {c: rgb_image(decode_image_bit(x[i])) for c, x in zip(CAMERAS, images, strict=True)}}
        return
    frames = source.files[1:]
    mask = control_mask(source.metadata, len(frames))
    for i, path in enumerate(frames):
        # HIL pickle input is trusted local collection data, never downloaded arbitrary pickles.
        with path.open("rb") as stream:
            obs = pickle.load(stream)
        index = int(re.search(r"(\d+)$", path.stem).group(1))
        yield {"frame_index": index, "state": np.asarray(obs["joint_action"]["vector"], dtype=np.float32),
               "images": {c: rgb_image(obs["observation"][r]["rgb"]) for c, r in zip(CAMERAS, RAW_CAMERAS, strict=True)},
               "control_source": mask[i]}
