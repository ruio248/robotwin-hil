#!/usr/bin/env python3
"""Read-only Gradio browser and quality audit for LeRobot v2.1 episodes.

The original dataset is never edited. Review labels/notes are written to a
separate JSONL sidecar chosen with --annotation-path.

Example (on the machine that can read the dataset):
  python sft_data_audit_app.py --dataset-root /path/to/lerobot_dataset \
      --host 127.0.0.1 --port 7860 --annotation-path ./sft_audit_notes.jsonl

For a private remote session, keep --host at 127.0.0.1 and use an SSH tunnel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class LeRobotDataset:
    """Small, lazy reader for LeRobot v2.1 per-episode parquet/video layout."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(
                f"Expected LeRobot metadata at {info_path} and {episodes_path}"
            )
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.episodes = sorted(
            _read_jsonl(episodes_path), key=lambda item: int(item["episode_index"])
        )
        if not self.episodes:
            raise ValueError(f"No episodes listed in {episodes_path}")
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_template = self.info.get(
            "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )
        self.video_template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        self.video_keys = [
            key
            for key, feature in self.info.get("features", {}).items()
            if feature.get("dtype") == "video"
        ]
        self.annotations: dict[int, dict[str, Any]] = {}
        self._data_files = None
        self._data_path_index: dict[int, Path] | None = None
        self._video_path_index: dict[tuple[int, str], Path] | None = None

    def _index_files(self) -> None:
        if self._data_path_index is not None:
            return
        data_paths: dict[int, Path] = {}
        for path in self.root.glob("data/chunk-*/*.parquet"):
            match = re.fullmatch(r"episode_(\d+)\.parquet", path.name)
            if match:
                data_paths[int(match.group(1))] = path
        video_paths: dict[tuple[int, str], Path] = {}
        for path in self.root.glob("videos/chunk-*/*/episode_*.mp4"):
            match = re.fullmatch(r"episode_(\d+)\.mp4", path.name)
            if match:
                video_paths[(int(match.group(1)), path.parent.name)] = path
        self._data_path_index = data_paths
        self._video_path_index = video_paths

    def _template_values(self, episode_index: int, video_key: str = "") -> dict[str, Any]:
        return {
            "episode_index": episode_index,
            "episode_chunk": episode_index // self.chunks_size,
            "file_index": episode_index // int(self.info.get("episodes_per_file", self.chunks_size)),
            "video_key": video_key,
        }

    def episode_meta(self, episode_index: int) -> dict[str, Any]:
        for item in self.episodes:
            if int(item["episode_index"]) == episode_index:
                return item
        raise KeyError(f"episode_index={episode_index} not found")

    def data_path(self, episode_index: int) -> Path | None:
        self._index_files()
        assert self._data_path_index is not None
        if episode_index in self._data_path_index:
            return self._data_path_index[episode_index]
        values = self._template_values(episode_index)
        try:
            candidate = self.root / self.data_template.format(**values)
        except (KeyError, ValueError):
            candidate = None
        if candidate is not None and candidate.is_file():
            return candidate
        matches = list(self.root.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
        if matches:
            return matches[0]
        # Some LeRobot exports store several episodes in one parquet shard.
        return None

    def video_path(self, episode_index: int, key: str) -> Path:
        self._index_files()
        assert self._video_path_index is not None
        indexed = self._video_path_index.get((episode_index, key))
        if indexed is not None:
            return indexed
        values = self._template_values(episode_index, key)
        try:
            candidate = self.root / self.video_template.format(**values)
        except (KeyError, ValueError):
            candidate = self.root / "__missing_video_template__"
        if candidate.is_file():
            return candidate
        # Also accept exports whose camera directory has been sanitized.
        chunk = episode_index // self.chunks_size
        suffix = f"episode_{episode_index:06d}.mp4"
        matches = list((self.root / "videos").glob(f"chunk-{chunk:03d}/**/{suffix}"))
        return matches[0] if matches else candidate

    def has_video(self, episode_index: int, key: str) -> bool:
        self._index_files()
        assert self._video_path_index is not None
        return (episode_index, key) in self._video_path_index

    def existing_data_indices(self) -> list[int]:
        return [
            int(item["episode_index"])
            for item in self.episodes
            if self.data_path(int(item["episode_index"])) is not None
        ]

    def completeness_rows(self) -> list[list[Any]]:
        rows = []
        for item in self.episodes:
            idx = int(item["episode_index"])
            data_ok = self.data_path(idx) is not None
            present = sum(self.has_video(idx, key) for key in self.video_keys)
            label = self.annotations.get(idx, {}).get("label", "unchecked")
            rows.append([idx, item.get("length", ""), data_ok, present, len(self.video_keys) - present, label])
        return rows

    @lru_cache(maxsize=8)
    def dataframe(self, episode_index: int) -> pd.DataFrame:
        path = self.data_path(episode_index)
        if path is not None:
            return pd.read_parquet(path).sort_values("frame_index").reset_index(drop=True)

        if self._data_files is None:
            self._data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        for shard in self._data_files:
            frame = pd.read_parquet(shard, filters=[("episode_index", "=", episode_index)])
            if not frame.empty:
                return frame.sort_values("frame_index").reset_index(drop=True)
        raise FileNotFoundError(f"No parquet rows found for episode {episode_index}")

    def load_annotations(self, path: Path) -> None:
        if path.exists():
            self.annotations = {
                int(row["episode_index"]): row for row in _read_jsonl(path)
            }

    def save_annotation(self, path: Path, episode_index: int, label: str, note: str) -> str:
        previous = self.annotations.get(episode_index, {})
        record = {
            "episode_index": episode_index,
            "label": label,
            "note": note,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(self.root),
            "dataset_codebase_version": self.info.get("codebase_version"),
        }
        self.annotations[episode_index] = record
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self.annotations[idx] for idx in sorted(self.annotations)]
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temp.replace(path)
        return f"Saved review for episode {episode_index}: {label}"


class VideoFrames:
    def __init__(self):
        self._cache: OrderedDict[tuple[str, int, float], np.ndarray | None] = OrderedDict()
        self._lock = threading.Lock()
        self._max_frames = 24

    def get(self, path: Path, frame_index: int, fps: float = 15.0):
        key = (str(path), max(0, int(frame_index)), float(fps) if fps > 0 else 15.0)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        image = None
        try:
            import imageio_ffmpeg
            from PIL import Image

            command = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                "-nostdin", "-threads", "1", "-ss", f"{key[1] / key[2]:.6f}",
                "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, check=False)
            if result.returncode == 0 and result.stdout:
                image = np.asarray(Image.open(BytesIO(result.stdout)).convert("RGB"))
        except (ImportError, OSError, subprocess.SubprocessError, ValueError):
            image = None
        with self._lock:
            self._cache[key] = image
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_frames:
                self._cache.popitem(last=False)
        return image


def _vectors(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame:
        return None
    try:
        return np.stack([np.asarray(value, dtype=np.float64).reshape(-1) for value in frame[column]])
    except (TypeError, ValueError):
        return None


def _diagnostics(ds: LeRobotDataset, idx: int, frame: pd.DataFrame, jump_threshold: float) -> tuple[str, dict[str, np.ndarray]]:
    meta = ds.episode_meta(idx)
    state = _vectors(frame, "observation.state")
    action = _vectors(frame, "action")
    messages: list[str] = []
    metrics: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    expected = int(meta.get("length", len(frame)))
    if len(frame) != expected:
        messages.append(f"⚠ parquet 有 {len(frame)} 帧，episodes.jsonl 记录 {expected} 帧")
    if "timestamp" in frame:
        ts = np.asarray(frame["timestamp"], dtype=np.float64)
        dt = np.diff(ts)
        if not np.all(np.isfinite(ts)):
            messages.append("⚠ timestamp 含非有限值")
        if np.any(dt <= 0):
            messages.append("⚠ timestamp 非严格递增或有重复")
        elif len(dt):
            fps = 1.0 / float(np.median(dt))
            metrics.append(f"记录频率约 {fps:.2f} Hz；时长 {max(0.0, float(ts[-1] - ts[0])):.2f}s")
            if np.max(np.abs(dt - np.median(dt))) > max(0.02, float(np.median(dt)) * 0.25):
                messages.append("⚠ timestamp 间隔有明显抖动")
    if state is None:
        messages.append("⚠ 缺少 observation.state，无法画状态关节曲线")
    else:
        arrays["state"] = state
        if not np.all(np.isfinite(state)):
            messages.append("⚠ observation.state 含 NaN/Inf")
        if state.shape[1] >= 14:
            joint_cols = list(range(6)) + list(range(7, 13))
        else:
            joint_cols = list(range(state.shape[1]))
        joints = state[:, joint_cols]
        if len(joints) > 1 and np.all(np.isfinite(joints)):
            delta = np.diff(joints, axis=0)
            max_step = float(np.max(np.abs(delta)))
            metrics.append(f"状态最大单帧关节变化 {max_step:.3f}（数据单位；阈值 {jump_threshold:.3f}）")
            if max_step > jump_threshold:
                messages.append("⚠ 发现超过阈值的状态单帧跳变，请结合视频确认")
            path_len = float(np.linalg.norm(delta, axis=1).sum())
            endpoint = float(np.linalg.norm(joints[-1] - joints[0]))
            ratio = path_len / max(endpoint, 1e-6)
            metrics.append(f"关节空间路径长度/首尾距离 {ratio:.1f}（仅是关节空间启发式，不代表末端空间回环）")
            reversals = 0
            for col in range(delta.shape[1]):
                d = delta[:, col]
                signs = np.sign(d[np.abs(d) > 1e-3])
                if len(signs) > 1:
                    reversals += int(np.sum(signs[1:] != signs[:-1]))
            metrics.append(f"关节方向反转次数 {reversals}（忽略绝对变化 ≤0.001 的帧）")
    if action is None:
        messages.append("⚠ 缺少 action，无法检查动作曲线")
    else:
        arrays["action"] = action
        if not np.all(np.isfinite(action)):
            messages.append("⚠ action 含 NaN/Inf")
        if state is not None and action.shape == state.shape:
            residual = np.abs(action - state)
            metrics.append(f"|action − observation.state| 中位数/P99 {np.nanmedian(residual):.3f}/{np.nanpercentile(residual, 99):.3f}（仅报告，不假设两者必相等）")
    missing_videos = [key for key in ds.video_keys if not ds.video_path(idx, key).is_file()]
    if missing_videos:
        messages.append("⚠ 缺少视频流：" + ", ".join(missing_videos))
    if not any("ee_pose" in key.lower() or "eef" in key.lower() or "tcp" in key.lower() for key in frame.columns):
        metrics.append("数据未发现显式末端位姿字段；当前不能把关节空间曲线称为笛卡尔大回环")
    if not messages:
        messages.append("未触发基础结构异常提示；这不等于人工确认轨迹正确")
    body = "\n".join(["**检查提示**", *messages, "", "**数值摘要**", *[f"- {m}" for m in metrics], "", "说明：`observation.state`/`action` 沿用数据集字段名；是否为实测关节、控制目标、绝对值或 delta，需要结合生成配置确认。"])
    return body, arrays


def _feature_names(info: dict[str, Any], key: str, width: int) -> list[str]:
    names = info.get("features", {}).get(key, {}).get("names")
    if isinstance(names, list) and names and isinstance(names[0], list):
        names = names[0]
    if not isinstance(names, list) or len(names) != width:
        return [f"d{i}" for i in range(width)]
    return [str(name) for name in names]


def _plot_episode(frame: pd.DataFrame, arrays: dict[str, np.ndarray], info: dict[str, Any]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    state, action = arrays.get("state"), arrays.get("action")
    n = len(frame)
    x = np.asarray(frame["timestamp"], dtype=float) if "timestamp" in frame else np.arange(n)
    if len(x) != n or not np.all(np.isfinite(x)):
        x = np.arange(n)
    state_names = _feature_names(info, "observation.state", state.shape[1]) if state is not None else []
    action_names = _feature_names(info, "action", action.shape[1]) if action is not None else []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("左臂（6 关节 + 夹爪）", "右臂（6 关节 + 夹爪）"), vertical_spacing=0.12)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#111111"]
    for row, offset in [(1, 0), (2, 7)]:
        for j in range(7):
            col = offset + j
            if state is not None and col < state.shape[1]:
                fig.add_trace(go.Scatter(x=x, y=state[:, col], name=f"obs {state_names[col]}", legendgroup=f"obs{row}", line=dict(color=colors[j], dash="solid", width=1.4)), row=row, col=1)
            if action is not None and col < action.shape[1]:
                fig.add_trace(go.Scatter(x=x, y=action[:, col], name=f"action {action_names[col]}", legendgroup=f"act{row}", line=dict(color=colors[j], dash="dash", width=1.2)), row=row, col=1)
    fig.update_layout(height=650, margin=dict(l=35, r=20, t=65, b=35), legend=dict(orientation="h", y=-0.14), title="实线=observation.state；虚线=action（维度来自数据集，不自动推断单位/语义）")
    fig.update_xaxes(title_text="timestamp (s) / frame", row=2, col=1)
    return fig


def build_app(dataset_root: Path, annotation_path: Path, jump_threshold: float):
    import gradio as gr

    ds = LeRobotDataset(dataset_root)
    ds.load_annotations(annotation_path)
    frames = VideoFrames()
    valid_indices = ds.existing_data_indices()
    valid_index_set = set(valid_indices)
    choices = []
    for item in ds.episodes:
        idx = int(item["episode_index"])
        has_data = idx in valid_index_set
        label = ds.annotations.get(idx, {}).get("label", "unchecked")
        declared = item.get("length", "?")
        choices.append((f"#{idx:04d} · {declared} frames · {'data' if has_data else 'MISSING DATA'} · {label}", idx))
    first_idx = valid_indices[0] if valid_indices else int(ds.episodes[0]["episode_index"])

    declared_total = int(ds.info.get("total_episodes", len(ds.episodes)))
    missing_data = len(ds.episodes) - len(valid_indices)
    missing_video = sum(
        1
        for item in ds.episodes
        for key in ds.video_keys
        if not ds.has_video(int(item["episode_index"]), key)
    )
    startup = (
        f"**数据集** `{ds.root}`  ·  metadata {len(ds.episodes)}/{declared_total} episodes  ·  "
        f"可读 parquet {len(valid_indices)}/{len(ds.episodes)}  ·  缺失视频文件 {missing_video}  ·  "
        f"版本 `{ds.info.get('codebase_version', 'unknown')}`  ·  FPS `{ds.info.get('fps', 'unknown')}`\n\n"
        "若 parquet/video 缺失，先不要把当前目录当作完整训练集；网页会将缺失项明确列出来。"
    )
    summary_rows = ds.completeness_rows()

    # Gradio's Video component exposes autoplay but not a muted constructor
    # argument. The browser-side hook mutes the three streams (so autoplay is
    # permitted), applies 20x playback, and aligns the wrist views to the
    # head-camera clock once when each stream becomes ready. It runs before the Python callback; the observer then
    # catches each video source being replaced by the callback response.
    autoplay_js = r"""(idx) => {
      const install = () => {
        if (window.__sftAudit20xInstalled) return;
        window.__sftAudit20xInstalled = true;
        const syncVideos = () => {
          const videos = Array.from(document.querySelectorAll(
            '.sft-audit-autoplay video'
          ));
          if (!videos.length) return;
          const master = videos[0];
          const applySettings = (video) => {
            video.muted = true;
            video.defaultMuted = true;
            video.volume = 0;
            video.autoplay = true;
            video.playsInline = true;
            const previousRate = Number(video.dataset.sftAuditRate);
            const candidates = previousRate
              ? [previousRate, 20, 16, 12, 10, 8, 5, 2, 1]
              : [20, 16, 12, 10, 8, 5, 2, 1];
            for (const rate of [...new Set(candidates)]) {
              try {
                video.playbackRate = rate;
                video.dataset.sftAuditRate = String(rate);
                try { video.defaultPlaybackRate = rate; } catch (_) {}
                return rate;
              } catch (_) {}
            }
            return 1;
          };
          const start = (video) => {
            applySettings(video);
            const promise = video.play();
            if (promise && promise.catch) promise.catch(() => {});
          };
          videos.forEach((video) => {
            applySettings(video);
            video.preload = 'auto';
            if (video.dataset.sftAudit20xBound === '1') return;
            video.dataset.sftAudit20xBound = '1';
            video.addEventListener('loadedmetadata', () => {
              applySettings(video);
              if (video !== master && master.readyState >= 1 && !master.ended) {
                try { video.currentTime = master.currentTime; } catch (_) {}
              }
              start(video);
            });
            video.addEventListener('canplay', () => start(video));
            video.addEventListener('ratechange', () => {
              const targetRate = Number(video.dataset.sftAuditRate) || 16;
              if (video.playbackRate !== targetRate) applySettings(video);
            });
          });
          videos.forEach((video) => {
            if (video.readyState >= 2 && video.paused) start(video);
          });
        };
        new MutationObserver(syncVideos).observe(document.body, {
          subtree: true,
          childList: true,
          attributes: true,
          attributeFilter: ['src']
        });
        syncVideos();
      };
      install();
      return idx;
    }"""

    def load_episode(idx: int):
        idx = int(idx)
        try:
            data = ds.dataframe(idx)
        except Exception as exc:
            return (None, None, None, None, None, None, None,
                    f"### Episode {idx}\n\n读取失败：`{type(exc).__name__}: {exc}`",
                    gr.update(minimum=0, maximum=0, value=0),
                    ds.annotations.get(idx, {}).get("label", "unchecked"),
                    ds.annotations.get(idx, {}).get("note", ""))
        description, arrays = _diagnostics(ds, idx, data, jump_threshold)
        fig = _plot_episode(data, arrays, ds.info)
        length = len(data)
        t = int(data["frame_index"].iloc[0]) if "frame_index" in data and length else 0
        images = []
        videos = []
        for key in ds.video_keys[:3]:
            path = ds.video_path(idx, key)
            fps = float(ds.info.get("features", {}).get(key, {}).get("info", {}).get("video.fps", ds.info.get("fps", 15)))
            videos.append(str(path) if ds.has_video(idx, key) and path.is_file() else None)
            images.append(frames.get(path, t, fps) if ds.has_video(idx, key) else None)
        while len(images) < 3:
            images.append(None)
        while len(videos) < 3:
            videos.append(None)
        meta = ds.episode_meta(idx)
        task_text = "; ".join(str(x) for x in meta.get("tasks", []))
        status = f"### Episode {idx} / {len(ds.episodes)-1}\n\n**任务**：{task_text or '未记录'}\n\n**帧数**：parquet {length}，metadata {meta.get('length', '未记录')}\n\n{description}"
        return (videos[0], videos[1], videos[2], images[0], images[1], images[2], fig,
                status,
                gr.update(minimum=0, maximum=max(0, length - 1), value=min(t, max(0, length - 1))),
                ds.annotations.get(idx, {}).get("label", "unchecked"),
                ds.annotations.get(idx, {}).get("note", ""))

    def load_frame(idx: int, frame_idx: int):
        idx, frame_idx = int(idx), int(frame_idx)
        out = []
        for key in ds.video_keys[:3]:
            path = ds.video_path(idx, key)
            fps = float(ds.info.get("features", {}).get(key, {}).get("info", {}).get("video.fps", ds.info.get("fps", 15)))
            out.append(frames.get(path, frame_idx, fps) if ds.has_video(idx, key) else None)
        while len(out) < 3:
            out.append(None)
        return tuple(out)

    def save_review(idx: int, label: str, note: str):
        message = ds.save_annotation(annotation_path, int(idx), label, note)
        updated_choices = []
        for item in ds.episodes:
            episode_idx = int(item["episode_index"])
            current_label = ds.annotations.get(episode_idx, {}).get("label", "unchecked")
            suffix = "data" if ds.data_path(episode_idx) is not None else "MISSING DATA"
            updated_choices.append((f"#{episode_idx:04d} · {item.get('length', '?')} frames · {suffix} · {current_label}", episode_idx))
        return message, ds.completeness_rows(), gr.update(choices=updated_choices, value=int(idx))

    def shift_episode(current_idx: int, offset: int):
        ordered = [int(item["episode_index"]) for item in ds.episodes]
        if not ordered:
            return (gr.update(),) + load_episode(int(current_idx))
        try:
            position = ordered.index(int(current_idx))
        except ValueError:
            position = 0
        target = ordered[max(0, min(len(ordered) - 1, position + int(offset)))]
        return (gr.update(value=target),) + load_episode(target)

    # Video outputs are copied into Gradio's shared temp cache. Expire those
    # generated files regularly so browsing the full dataset cannot fill /tmp.
    with gr.Blocks(title="SFT Episode Audit", delete_cache=(60, 300)) as demo:
        gr.Markdown("# SFT 示范数据逐条核查\n只读浏览原始 parquet / 视频；人工标签写入独立 sidecar，不修改数据集。")
        gr.Markdown(startup)
        with gr.Row():
            episode = gr.Dropdown(choices=choices, value=first_idx, label="Episode（按编号浏览；缺失数据会标出）", scale=4)
            frame_index = gr.Slider(minimum=0, maximum=0, step=1, value=0, label="同步帧索引", scale=3)
        with gr.Row():
            previous_button = gr.Button("← 上一条")
            next_button = gr.Button("下一条 →", variant="primary")
        gr.Markdown("### 自动回放 · 目标 20×\n切换 episode 后三路视频自动静音播放。若浏览器不支持 20×，会自动退到其支持的最高倍速；三路视频会尽量对齐。下方同步帧和滑块用于逐帧细查。")
        with gr.Row(elem_classes=["sft-audit-autoplay"]):
            video_head = gr.Video(label="cam_high / head · 目标 20×", height=260, autoplay=True, loop=False, interactive=False, include_audio=False, show_download_button=False)
            video_left = gr.Video(label="cam_left_wrist · 目标 20×", height=260, autoplay=True, loop=False, interactive=False, include_audio=False, show_download_button=False)
            video_right = gr.Video(label="cam_right_wrist · 目标 20×", height=260, autoplay=True, loop=False, interactive=False, include_audio=False, show_download_button=False)
        with gr.Row():
            image_head = gr.Image(label="cam_high / head", height=260)
            image_left = gr.Image(label="cam_left_wrist", height=260)
            image_right = gr.Image(label="cam_right_wrist", height=260)
        with gr.Row():
            chart = gr.Plot(label="关节状态与动作")
            details = gr.Markdown("选择一条 episode 开始检查。")
        with gr.Row():
            review_label = gr.Radio(["unchecked", "keep", "review", "reject"], value="unchecked", label="人工核查结论")
            note = gr.Textbox(label="备注（例如：大回环、接收臂没闭合、提前结束）", lines=2)
            save_button = gr.Button("保存核查标记", variant="primary")
        save_status = gr.Markdown()
        with gr.Accordion("450 条数据完整性清单", open=False):
            table = gr.Dataframe(
                value=summary_rows,
                headers=["episode", "metadata frames", "parquet present", "videos present", "videos missing", "review"],
                datatype=["number", "number", "bool", "number", "number", "str"],
                interactive=False,
                wrap=True,
            )
        episode_outputs = [video_head, video_left, video_right, image_head, image_left, image_right, chart, details, frame_index, review_label, note]
        episode.change(load_episode, inputs=[episode], outputs=episode_outputs, js=autoplay_js)
        frame_index.change(load_frame, inputs=[episode, frame_index], outputs=[image_head, image_left, image_right])
        save_button.click(save_review, inputs=[episode, review_label, note], outputs=[save_status, table, episode])
        navigation_outputs = [episode, *episode_outputs]
        previous_button.click(lambda idx: shift_episode(idx, -1), inputs=[episode], outputs=navigation_outputs, js=autoplay_js)
        next_button.click(lambda idx: shift_episode(idx, 1), inputs=[episode], outputs=navigation_outputs, js=autoplay_js)
        demo.load(load_episode, inputs=[episode], outputs=episode_outputs, js=autoplay_js)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot v2.1 dataset root (contains meta/info.json)")
    parser.add_argument("--annotation-path", type=Path, default=Path("./sft_audit_annotations.jsonl"), help="Sidecar for review labels; kept outside the dataset")
    parser.add_argument("--gradio-cache-dir", type=Path, default=None, help="Temporary Gradio media cache (defaults beside --annotation-path)")
    parser.add_argument("--host", default="127.0.0.1", help="Keep loopback-only when using an SSH tunnel")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--jump-threshold", type=float, default=0.5, help="Flag-only per-frame joint-change heuristic; in the dataset's stored units")
    args = parser.parse_args()
    cache_dir = args.gradio_cache_dir or args.annotation_path.expanduser().resolve().parent / "gradio-cache"
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(cache_dir)
    app = build_app(args.dataset_root, args.annotation_path, args.jump_threshold)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        allowed_paths=[str(args.dataset_root.expanduser().resolve())],
    )


if __name__ == "__main__":
    main()
