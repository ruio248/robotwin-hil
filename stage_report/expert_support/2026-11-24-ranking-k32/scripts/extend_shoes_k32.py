#!/usr/bin/env python3
"""Resample matching SFT K32 candidates for click_alarmclock and place_dual_shoes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from resample_cup_sft import CAMERAS, atomic_npz, decode_selected, joint_chunk
from train import digest
sys.path.insert(0, str(REPO / 'robotwin_hil/overlay/scripts'))
from coverage_benchmark.train_screened_task_discriminators import atomic_json


def load_click(entry, source_root):
    folder = source_root / entry['files']['pairs.npz']
    with np.load(folder) as z:
        arrays = {key: z[key] for key in z.files}
    meta = json.loads((source_root / entry['files']['metadata.json']).read_text())
    observations = []
    for index, state in enumerate(arrays['state']):
        observations.append({'state': state, 'images': {
            'cam_high': arrays['images_head'][index].transpose(2, 0, 1).copy(),
            'cam_left_wrist': arrays['images_left'][index].transpose(2, 0, 1).copy(),
            'cam_right_wrist': arrays['images_right'][index].transpose(2, 0, 1).copy(),
        }, 'instruction': meta['instruction']})
    return arrays, observations


def load_shoes(entry, source_root, train_source, test_source, shared_train_images=None):
    with np.load(source_root / entry['file']) as z:
        arrays = {key: z[key] for key in z.files}
    indices = arrays['frame_index'].astype(int)
    if entry['split'] == 'train':
        parquet = train_source / 'data/chunk-000/file-000.parquet'
        rows = int(entry['metadata']['video_start']) + indices
        images = shared_train_images
        prompts = entry['metadata']['tasks']
        table = pq.read_table(parquet, columns=['observation.state', 'task_index']).to_pydict()
        prompt_values = [prompts[str(table['task_index'][i])] for i in rows]
    else:
        folder = test_source / f'scene_{entry["scene_seed"]}'
        parquet = folder / 'frames.parquet'
        rows = indices
        images = {camera: decode_selected(folder / f'{camera}.mp4', set(int(i) for i in indices))
                  for camera in CAMERAS}
        prompt_values = [entry['instruction']] * len(indices)
        table = pq.read_table(parquet, columns=['observation.state']).to_pydict()
    source_states = np.stack([np.asarray(table['observation.state'][int(i)], dtype=np.float32)
                              for i in rows])
    if not np.array_equal(source_states, arrays['state']):
        raise ValueError(f'Shoes state mismatch: {entry["id"]}')
    observations = []
    for row, frame in enumerate(rows):
        observations.append({'state': arrays['state'][row],
                             'images': {camera: images[camera][int(frame)].transpose(2, 0, 1).copy()
                                        for camera in CAMERAS},
                             'instruction': prompt_values[row]})
    return arrays, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('click_alarmclock', 'place_dual_shoes'), required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--policy-checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=70)
    args = parser.parse_args()
    from client_server.ws import WsModelClient
    source_root = REPO / 'outputs/support_three_tasks_20261123/matched_sft_data' / args.task
    source_path = source_root / 'manifest.json'
    source = json.loads(source_path.read_text())
    if not source['complete'] or source['cohorts'] != {'train': 50, 'test': 20}:
        raise ValueError('Expected complete train50/test20 source')
    params = args.policy_checkpoint / 'params/manifest.ocdbt'
    if not params.is_file():
        raise FileNotFoundError(params)
    output = args.output_root / args.task
    output.mkdir(parents=True, exist_ok=True)
    entries = source['episodes'][:args.limit]
    shared_train_images = None
    if args.task == 'place_dual_shoes':
        train_source = REPO / 'outputs/support_three_tasks_20261123/place_dual_shoes/train_source'
        test_source = REPO / 'outputs/support_three_tasks_20261123/place_dual_shoes/test20_frames'
        all_train = [e for e in source['episodes'] if e['split'] == 'train']
        selected = set()
        for entry in all_train:
            with np.load(source_root / entry['file']) as z:
                selected.update(int(entry['metadata']['video_start']) + int(i) for i in z['frame_index'])
        shared_train_images = {camera: decode_selected(train_source / f'videos/observation.images.{camera}/chunk-000/file-000.mp4', selected)
                               for camera in CAMERAS}
    client = WsModelClient(url=f'ws://127.0.0.1:{args.port}', evaluation_id=f'{args.task}-sft-pairs',
                           trial_id='matched-sft-pairs', action_case_id='matched-sft-pairs')
    records = []
    try:
        for count, entry in enumerate(entries, 1):
            if args.task == 'click_alarmclock':
                rel = entry['files']['policy_candidates_k4.npz']
                target = output / rel
                if target.is_file():
                    records.append(entry)
                    print(f'REUSED {count}/{len(entries)} {entry["scene_seed"]}', flush=True)
                    continue
                arrays, observations = load_click(entry, source_root)
            else:
                target = output / entry['file']
                if target.is_file():
                    records.append(dict(entry, sha256=digest(target)))
                    print(f'REUSED {count}/{len(entries)} {entry["id"]}', flush=True)
                    continue
                arrays, observations = load_shoes(entry, source_root, train_source, test_source, shared_train_images)
            client.call(func_name='prepare_case', obs={'task_name': args.task,
                        'seed': int(entry.get('scene_seed', entry.get('metadata', {}).get('episode_index', 0))),
                        'instruction': observations[0]['instruction'], 'action_type': 'joint'})
            client.call(func_name='reset')
            horizon = 50 if args.task == 'click_alarmclock' else 10
            policy = np.empty((len(observations), 32, horizon, 14), dtype=np.float32)
            for row, observation in enumerate(observations):
                client.call(func_name='update_obs', obs=observation)
                policy[row, :4] = arrays['policy_chunks'][row]
                for offset in range(4,32,4):
                    raw = client.call(func_name='get_action_candidates', obs=4)
                    if len(raw) != 4:
                        raise ValueError('Wrong microbatch')
                    for j in range(4):
                        policy[row,offset+j] = joint_chunk(raw[j])
            if args.task == 'click_alarmclock':
                atomic_npz(target, {'policy_chunks': policy})
                for key in ('pairs.npz', 'features_r3m18.npz', 'metadata.json'):
                    source_file = source_root / entry['files'][key]
                    target_file = output / entry['files'][key]
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    if not target_file.exists():
                        target_file.symlink_to(source_file.resolve())
                records.append(entry)
            else:
                arrays['policy_chunks'] = policy
                atomic_npz(target, arrays)
                records.append(dict(entry, sha256=digest(target)))
            print(f'SFT K32 {count}/{len(entries)} {entry.get("scene_seed", entry.get("id"))}: {len(observations)} anchors', flush=True)
            atomic_json(output / 'progress.json', {'task': args.task, 'episodes': len(records),
                        'source_manifest_sha256': digest(source_path)})
        manifest = dict(source)
        manifest['episodes'] = records
        manifest['complete'] = len(records) == 70
        if args.task == 'click_alarmclock':
            manifest['path_base'] = str(output.resolve())
        manifest['matched_sft_policy_checkpoint'] = str(args.policy_checkpoint.resolve())
        manifest['matched_sft_checkpoint_manifest_sha256'] = digest(params)
        manifest['candidate_semantics'] = 'K32 preserves matched K4 prefix and adds 28 same-observation SFT candidates'
        atomic_json(output / 'manifest.json', manifest)
    finally:
        client.close()


if __name__ == '__main__':
    main()
