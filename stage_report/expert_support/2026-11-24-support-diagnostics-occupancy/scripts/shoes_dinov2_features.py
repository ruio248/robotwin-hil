"""Export a frozen encoder (genpo env), then encode the exact matched-SFT anchors (robotwin_hil env)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2] / 'outputs/support_three_tasks_20261123'
ASSETS = ROOT / 'assets/dinov2-small'
DEST = ROOT / 'shoes_dinov2_10k'
TASK = 'place_dual_shoes'


def export():
    from transformers import Dinov2Model

    class CLS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Dinov2Model.from_pretrained(ASSETS).eval().requires_grad_(False)

        def forward(self, pixels):
            return self.encoder(pixels).last_hidden_state[:, 0]

    model = CLS().cuda().eval()
    with torch.no_grad():
        sample = torch.randn(3, 3, 224, 224, device='cuda')
        traced = torch.jit.trace(model, sample, check_trace=False)
        for n in (1, 3, 24):
            pixels = torch.randn(n, 3, 224, 224, device='cuda')
            torch.testing.assert_close(traced(pixels), model(pixels), atol=1e-4, rtol=1e-4)
        traced.save(str(DEST / 'encoder.pt'))


def extract():
    from resample_click_shoes_sft import CAMERAS, decode_selected, load_shoes
    from tail_value.prepare import image_tensor
    from train import digest

    source = ROOT / 'matched_sft_data' / TASK
    entries = json.loads((source / 'manifest.json').read_text())['episodes']
    packed = ROOT / TASK / 'train_source'
    selected = set()
    for entry in entries:
        if entry['split'] == 'train':
            with np.load(source / entry['file']) as z:
                selected.update(int(entry['metadata']['video_start']) + int(i) for i in z['frame_index'])
    images = {c:decode_selected(packed / f'videos/observation.images.{c}/chunk-000/file-000.mp4', selected) for c in CAMERAS}
    encoder = torch.jit.load(str(DEST / 'encoder.pt')).cuda().eval()
    for entry in entries:
        arrays, observations = load_shoes(entry, source, packed, ROOT / TASK / 'test20_frames', images)
        features = []
        with torch.inference_mode():
            for i in range(0, len(observations), 8):
                pixels = torch.stack([image_tensor(o['images'][c].transpose(1,2,0))
                                      for o in observations[i:i+8] for c in CAMERAS])
                features.append(encoder(pixels.cuda()).reshape(-1,3,384).cpu().numpy())
        value = np.concatenate(features)
        assert len(value) == len(arrays['state']) and np.isfinite(value).all()
        np.savez(DEST / (entry['id'] + '.npz'), features=value, source=digest(source / entry['file']))
    print('Encoded', len(entries), 'episodes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['export', 'extract'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    DEST.mkdir(parents=True, exist_ok=True)
    (export if args.phase == 'export' else extract)()
