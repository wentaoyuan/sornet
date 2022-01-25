import argparse
import h5py
import json
import numpy as np
import torch
from PIL import Image
from datasets import normalize_rgb
from io import BytesIO
from matplotlib import pyplot as plt
from networks import EmbeddingNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--split')
    parser.add_argument('--obj_file')
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--view', type=int, default=0)
    parser.add_argument('--colors', nargs='+')
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--checkpoint')
    # Vis
    parser.add_argument('--seq_id', type=int, default=2)
    parser.add_argument('--frame_id', type=int, default=0)
    parser.add_argument('--layer_to_vis', type=int, default=3)
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
    rgb = Image.open(BytesIO(data[f'rgb{args.view}'][args.frame_id]))
    rgb = rgb.convert('RGB').resize((args.img_w, args.img_h))
    img = normalize_rgb(rgb).unsqueeze(0).cuda()

    colors = args.colors
    if colors is None:
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
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda().eval()

    with torch.no_grad():
        emb, attn_weights = model(img, patch_tensors)

    colormap = {
        'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1],
        'yellow': [0.5, 0.5, 0], 'cyan':[0, 0.5, 0.5], 'pink': [0.4, 0.2, 0.4]
    }

    n_obj = len(colors)
    attn_weights = attn_weights[0, args.layer_to_vis, -n_obj:].cpu()

    plt.figure(figsize=(4 * len(colors), 5))
    plt.subplot2grid((8, n_obj + 1), (2, 0), rowspan=6)
    plt.imshow(rgb)
    plt.axis('off')
    for i, attn in enumerate(attn_weights):
        mask = np.zeros((args.img_h, args.img_w, 3))
        for p, a in enumerate(attn[:-n_obj]):
            r = p // (args.img_w // args.patch_size)
            c = p % (args.img_w // args.patch_size)
            mask[
                r*args.patch_size:(r+1)*args.patch_size,
                c*args.patch_size:(c+1)*args.patch_size
            ] += a.item() * np.array(colormap[colors[i]]) * 10
        out = np.array(rgb) / 255.0 + mask
        out /= out.max()
        plt.subplot2grid((8, n_obj+1), (0, i+1), rowspan=2)
        plt.imshow(obj_patches[i])
        plt.axis('off')
        plt.subplot2grid((8, n_obj+1), (2, i+1), rowspan=6)
        plt.imshow(np.clip(out, 0, 1))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
