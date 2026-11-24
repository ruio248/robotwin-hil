#!/usr/bin/env python3
"""Replace LoRA-source cup K4 negatives with exact-observation SFT H10 samples."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'robotwin_hil/overlay/scripts'))
from coverage_benchmark.train_screened_task_discriminators import atomic_json
from train import digest

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def decode_selected(video: Path, selected: set[int]) -> dict[int, np.ndarray]:
    if not selected:
        return {}
    process = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-c:v', 'libdav1d', '-i', str(video),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    frame_bytes = 240 * 320 * 3
    frames = {}
    try:
        for index in range(max(selected) + 1):
            data = process.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                raise ValueError(f'Video ended at frame {index}: {video}')
            if index in selected:
                frames[index] = np.frombuffer(data, dtype=np.uint8).reshape(240, 320, 3).copy()
        process.stdout.close()
        process.terminate()
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return frames


def joint_chunk(raw):
    actions = []
    for action in raw:
        actions.append(np.concatenate([
            np.asarray(action['left_arm_joint_state']).reshape(6),
            np.asarray(action['left_ee_joint_state']).reshape(1),
            np.asarray(action['right_arm_joint_state']).reshape(6),
            np.asarray(action['right_ee_joint_state']).reshape(1),
        ]))
    out = np.asarray(actions, dtype=np.float32)
    if out.shape != (10, 14) or not np.isfinite(out).all():
        raise ValueError(f'Invalid H10 response: {out.shape}')
    return out


def atomic_npz(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18816)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=REPO / 'outputs/support_three_tasks_20261123/matched_sft_data/place_empty_cup')
    parser.add_argument('--policy-checkpoint', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=70)
    args = parser.parse_args()
    from client_server.ws import WsModelClient

    manifest_path = args.source_root / 'manifest.json'
    source = json.loads(manifest_path.read_text())
    if not source['complete'] or source['cohorts'] != {'train': 50, 'test': 20}:
        raise ValueError('Expected complete cup train50/test20 cache')
    params = args.policy_checkpoint / 'params/manifest.ocdbt'
    if not params.is_file():
        raise FileNotFoundError(params)
    entries = source['episodes'][:args.limit]
    output_task = args.output_root / 'place_empty_cup'
    output_task.mkdir(parents=True, exist_ok=True)
    client = WsModelClient(url=f'ws://127.0.0.1:{args.port}', evaluation_id='cup-sft-matched-pairs',
                           trial_id='cup-sft-pairs', action_case_id='cup-sft-pairs')
    new_entries = []
    test_frames = None
    try:
        for number, entry in enumerate(entries, 1):
            target = output_task / entry['file']
            if target.is_file():
                new_entry = dict(entry)
                new_entry['sha256'] = digest(target)
                new_entries.append(new_entry)
                print(f'REUSED {number}/{len(entries)} {entry["id"]}', flush=True)
                continue
            with np.load(args.source_root / entry['file'], allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            anchor_indices = arrays['frame_index'].astype(int)
            parquet_path = next(Path(item['path']) for item in entry['source_fingerprint']
                                if item['path'].endswith('.parquet') and '/data/' in item['path'])
            table = pq.read_table(parquet_path, columns=['observation.state', 'task_index']).to_pydict()
            episode_index = int(entry['metadata']['episode_index'])
            if entry['split'] == 'train':
                row_indices = anchor_indices
                image_indices = anchor_indices
            else:
                offset = int(entry['metadata']['video_start'])
                row_indices = offset + anchor_indices
                image_indices = row_indices
            state_from_source = np.stack([np.asarray(table['observation.state'][i], dtype=np.float32)
                                          for i in row_indices])
            if not np.array_equal(state_from_source, arrays['state']):
                raise ValueError(f'State mismatch for {entry["id"]}')
            prompts = [entry['metadata']['tasks'][str(table['task_index'][i])] for i in row_indices]
            if entry['split'] == 'test' and test_frames is None:
                all_test = [e for e in source['episodes'] if e['split'] == 'test']
                selected = set()
                for e in all_test:
                    with np.load(args.source_root / e['file']) as z:
                        selected.update(int(e['metadata']['video_start']) + int(x) for x in z['frame_index'])
                test_frames = {camera: decode_selected(next(Path(item['path']) for item in entry['source_fingerprint']
                                                            if f'observation.images.{camera}' in item['path']), selected)
                               for camera in CAMERAS}
            if entry['split'] == 'train':
                images = {camera: decode_selected(next(Path(item['path']) for item in entry['source_fingerprint']
                                                            if f'observation.images.{camera}' in item['path']),
                                                  set(image_indices)) for camera in CAMERAS}
            else:
                images = test_frames
            client.call(func_name='prepare_case', obs={'task_name': 'place_empty_cup',
                        'seed': episode_index, 'instruction': prompts[0], 'action_type': 'joint'})
            client.call(func_name='reset')
            policy = np.empty((len(anchor_indices), 32, 10, 14), dtype=np.float32)
            for row, (frame, prompt) in enumerate(zip(image_indices, prompts)):
                observation = {'state': arrays['state'][row],
                               'images': {camera: images[camera][int(frame)].transpose(2, 0, 1).copy()
                                          for camera in CAMERAS},
                               'instruction': prompt}
                client.call(func_name='update_obs', obs=observation)
                policy[row,:4] = arrays['policy_chunks'][row]
                for candidate in range(4,32):
                    policy[row, candidate] = joint_chunk(client.call(func_name='get_action'))
            arrays['policy_chunks'] = policy
            atomic_npz(target, arrays)
            new_entry = dict(entry)
            new_entry['sha256'] = digest(target)
            new_entries.append(new_entry)
            print(f'SFT K32 {number}/{len(entries)} {entry["id"]}: {len(anchor_indices)} anchors', flush=True)
            progress = {'complete': False, 'task': 'place_empty_cup', 'episodes': len(new_entries),
                        'source_manifest_sha256': digest(manifest_path)}
            atomic_json(output_task / 'progress.json', progress)
        result = dict(source)
        result['episodes'] = new_entries
        result['complete'] = len(new_entries) == 70
        result['config'] = dict(source['config'])
        result['config'].update(policy_checkpoint=str(args.policy_checkpoint.resolve()),
                                policy_checkpoint_manifest_sha256=digest(params),
                                candidate_semantics='K32 preserving matched K4 prefix, adding 28 exact-observation SFT H10 chunks',
                                source_manifest_sha256=digest(manifest_path))
        atomic_json(output_task / 'manifest.json', result)
    finally:
        client.close()


if __name__ == '__main__':
    main()
