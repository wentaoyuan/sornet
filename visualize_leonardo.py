import argparse
import h5py
import json
import torch
from PIL import Image
from datasets import normalize_rgb, build_predicates
from io import BytesIO
from matplotlib import pyplot as plt
from networks import EmbeddingNet, ReadoutNet
from train_leonardo import plot

unary_pred = [
    'on_surface(%s, left)', 'on_surface(%s, right)', 'on_surface(%s, far)',
    'on_surface(%s, center)', 'has_obj(robot, %s)', 'top_is_clear(%s)',
    'in_approach_region(robot, %s)'
]
binary_pred = ['stacked(%s, %s)', 'aligned_with(%s, %s)']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--split')
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--view', type=int, default=0)
    parser.add_argument('--n_objects', type=int, default=4)
    parser.add_argument('--obj_file')
    parser.add_argument('--colors', nargs='+')
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    parser.add_argument('--checkpoint')
    # Vis
    parser.add_argument('--seq_id', type=int, default=2)
    parser.add_argument('--frame_id', type=int, default=0)
    args = parser.parse_args()

    n_frames = json.load(open(f'{args.data_dir}/{args.split}_nframes.json'))
    sequences = list(n_frames.keys())
    h5 = h5py.File(f'{args.data_dir}/{args.split}.h5', 'r')
    all_predicates = h5['predicates'][()].decode().split('|')
    pred_ids = {pred: i for i, pred in enumerate(all_predicates)}
    objects = [f'object{i:02d}' for i in range(args.n_objects)]
    predicates = build_predicates(objects, unary_pred, binary_pred)
    pred_ids = [pred_ids[pred] for pred in predicates]

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

    target = data['logical'][args.frame_id][pred_ids]
    target = torch.from_numpy(target).unsqueeze(0)

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, len(colors),
        args.width, args.layers, args.heads
    )
    head = ReadoutNet(
        args.width, args.d_hidden, len(unary_pred), len(binary_pred)
    )
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.cuda().eval()
    head = head.cuda().eval()

    with torch.no_grad():
        emb, attn = model(img, patch_tensors)
        logits = head(emb)

    data = (img, patch_tensors, None, target)
    fig = plot(predicates, 1, data, logits)
    plt.show()
