'''
MIT License

Copyright (c) 2022 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import h5py
import json
import numpy as np
import torch
from PIL import Image
from datasets import normalize_rgb
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from networks import EmbeddingNet, ReadoutNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--split')
    parser.add_argument('--obj_file')
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--focal', type=float, default=221.7025)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    parser.add_argument('--model_checkpoint')
    parser.add_argument('--dir_head_checkpoint')
    parser.add_argument('--dist_head_checkpoint')
    parser.add_argument('--ee', action='store_true')
    # Vis
    parser.add_argument('--seq_id', type=int, default=2)
    parser.add_argument('--frame_id', type=int, default=0)
    parser.add_argument('--view', type=int, default=0)
    args = parser.parse_args()

    n_frames = json.load(open(f'{args.data_dir}/{args.split}_nframes.json'))
    sequences = list(n_frames.keys())
    h5 = h5py.File(f'{args.data_dir}/{args.split}.h5', 'r')

    data = h5[sequences[args.seq_id]]
    if args.frame_id > data[f'rgb{args.view}'].shape[0]:
        print(
            'Frame', args.frame_id, 'out of range. Length of current sequence is',
            data[f'rgb{args.view}'].shape[0]
        )
    rgb = Image.open(BytesIO(data[f'rgb{args.view}'][args.frame_id])).convert('RGB')
    orig_w, orig_h = rgb.size
    img = normalize_rgb(rgb.resize((args.img_w, args.img_h))).unsqueeze(0).cuda()

    colors = data['colors'][()].decode().split(',')
    with h5py.File(f'{args.data_dir}/{args.obj_file}') as obj_h5:
        obj_patches = [
            Image.open(BytesIO(obj_h5[color][0])) for color in colors
        ]
        patch_tensors = torch.stack(
            [normalize_rgb(p) for p in obj_patches]
        ).unsqueeze(0).cuda()

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, len(colors),
        args.width, args.layers, args.heads
    )
    if args.ee:
        dir_head = ReadoutNet(args.width, args.d_hidden, 3, 0)
        dist_head = ReadoutNet(args.width, args.d_hidden, 1, 0)
    else:
        dir_head = ReadoutNet(args.width, args.d_hidden, 0, 3)
        dist_head = ReadoutNet(args.width, args.d_hidden, 0, 1)
    model.load_state_dict(
        torch.load(args.model_checkpoint, map_location='cpu')['model']
    )
    dir_head.load_state_dict(
        torch.load(args.dir_head_checkpoint, map_location='cpu')
    )
    dist_head.load_state_dict(
        torch.load(args.dist_head_checkpoint, map_location='cpu')
    )
    model = model.cuda().eval()
    dir_head = dir_head.cuda().eval()
    dist_head = dist_head.cuda().eval()

    with torch.no_grad():
        emb, attn = model(img, patch_tensors)
        dir = dir_head(emb).cpu().numpy()[0].reshape(3, -1).T
        dist = dist_head(emb).cpu().numpy()[0]

    E = data[f'view_matrix{args.view}'][0]
    K = np.eye(3)
    K[0, 0] = K[1, 1] = args.focal
    K[0, 2] = (orig_w - 1) / 2
    K[1, 2] = (orig_h - 1) / 2

    plt.imshow(rgb)
    plt.axis('off')
    ax = plt.gca()
    if args.ee:
        ee_pos = data['ee_pose'][args.frame_id][:3, 3]
        p = np.insert(ee_pos, 3, 1)
        p = E @ p[:, None]
        p = K @ p[:3]
        p = p[:2, 0] / p[2, 0]
        p[0] = 256 - p[0]
        ee_img_pos = p
        for i, c in enumerate(colors):
            d = ee_pos + dir[i] * dist[i]
            d = np.insert(d, 3, 1)
            d = E @ d[:, None]
            d = K @ d[:3]
            d = d[:2, 0] / d[2, 0]
            d[0] = 256 - d[0]
            arrow = FancyArrowPatch(ee_img_pos, d, color=c, arrowstyle='->', linewidth=2, mutation_scale=20)
            ax.add_patch(arrow)
    else:
        pos = []
        img_pos = []
        for i in range(len(colors)):
            pos.append(data[f'object{i:02d}_pose'][args.frame_id][:3, 3])
            p = np.insert(pos[-1], 3, 1)
            p = E @ p[:, None]
            p = K @ p[:3]
            p = p[:2, 0] / p[2, 0]
            p[0] = 256 - p[0]
            img_pos.append(p)
        for i, c in enumerate(np.roll(colors, -1)):
            d = pos[i] + dir[i] * dist[i]
            d = np.insert(d, 3, 1)
            d = E @ d[:, None]
            d = K @ d[:3]
            d = d[:2, 0] / d[2, 0]
            d[0] = 256 - d[0]
            arrow = FancyArrowPatch(img_pos[i], d, color=c, arrowstyle='->', linewidth=2, mutation_scale=20)
            ax.add_patch(arrow)
    plt.tight_layout()
    plt.show()
