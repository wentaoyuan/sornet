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

from datasets import LeonardoDataset, build_predicates
from networks import EmbeddingNet, ReadoutNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import torch

unary_pred = [
    'on_surface(%s, left)', 'on_surface(%s, right)', 'on_surface(%s, far)',
    'on_surface(%s, center)', 'has_obj(robot, %s)', 'top_is_clear(%s)',
    'in_approach_region(robot, %s)'
]
binary_pred = ['stacked(%s, %s)', 'aligned_with(%s, %s)']


def calc_accuracy(pred, target):
    return (pred == target).sum(0) / target.shape[0] * 100


def calc_accuracy_allmatch(pred, target, keys, names):
    acc = {}
    acc['all'] = ((pred != target).sum(axis=1) == 0).sum() / pred.shape[0] * 100
    for key in keys:
        mask = [key in name for name in names]
        if sum(mask) > 0:
            correct = ((pred[:, mask] != target[:, mask]).sum(axis=1) == 0).sum()
            acc[key] = correct / pred.shape[0] * 100
        else:
            acc[key] = 0
    return acc


def calc_f1(pred, target):
    majority_is_one = target.shape[0] - target.sum(axis=0) < target.sum(axis=0)
    pred[:, majority_is_one] = ~pred[:, majority_is_one]
    target[:, majority_is_one] = ~target[:, majority_is_one]
    tp = (pred & target).sum(axis=0)
    fp = (pred & ~target).sum(axis=0)
    fn = (~pred & target).sum(axis=0)
    precision = tp / (tp + fp) * 100
    recall = tp / (tp + fn) * 100
    f1 = 2 * precision * recall / (precision + recall)
    f1[np.isnan(f1)] = 0
    return f1


def split_avg(data, keys, names):
    avg = {'all': np.mean(data)}
    for key in keys:
        mask = [key in name for name in names]
        if sum(mask) > 0:
            avg[key] = np.mean(data[mask])
        else:
            avg[key] = 0
    return avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--split')
    parser.add_argument('--obj_file')
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--n_views', type=int, default=1)
    parser.add_argument('--n_objects', type=int, default=4)
    parser.add_argument('--objects', nargs='+')
    parser.add_argument('--colors', nargs='+')
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--gripper', action='store_true')
    parser.add_argument('--d_hidden', type=int, default=512)
    # Evaluation
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_worker', type=int, default=0)
    args = parser.parse_args()

    if args.objects is None:
        objects = [f'object{i:02d}' for i in range(args.n_objects)]
    else:
        objects = args.objects
    pred_names = build_predicates(objects, unary_pred, binary_pred)

    loaders = []
    for v in range(args.n_views):
        data = LeonardoDataset(
            args.data_dir, args.split, pred_names, args.obj_file, args.colors,
            randpatch=False, view=v, randview=False, gripper=args.gripper
        )
        loaders.append(DataLoader(
            data, args.batch_size, pin_memory=True, num_workers=args.n_worker
        ))

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, len(objects),
        args.width, args.layers, args.heads
    )
    out_dim = args.width + 1 if args.gripper else args.width
    head = ReadoutNet(out_dim, args.d_hidden, len(unary_pred), len(binary_pred))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    head.load_state_dict(checkpoint['head'])
    model = model.cuda().eval()
    head = head.cuda().eval()

    predictions = []
    targets = []
    loaders.insert(0, tqdm(range(len(loaders[0]))))
    for data in zip(*loaders):
        data = data[1:]
        batch_size = data[0][0].shape[0]
        logits = 0
        for img, obj_patches, gripper, target in data:
            with torch.no_grad():
                img = img.cuda()
                obj_patches = obj_patches.cuda()
                emb, attn = model(img, obj_patches)
                if args.gripper:
                    gripper = gripper.cuda()
                    emb = torch.cat([
                        emb, gripper[:, None, None].expand(-1, len(objects), -1)
                    ], dim=-1)
                logits += head(emb)
        predictions.append((logits > 0).cpu().numpy())
        targets.append(target.bool().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    prefixes = [
        'on_surface', 'has_obj', 'top_is_clear',
        'in_approach_region', 'stacked', 'aligned_with'
    ]
    accuracy = split_avg(calc_accuracy(predictions, targets), prefixes, pred_names)
    accuracy_all = calc_accuracy_allmatch(predictions, targets, prefixes, pred_names)
    f1 = split_avg(calc_f1(predictions, targets), prefixes, pred_names)
    print('Accuracy')
    for key in accuracy:
        print(key, accuracy[key])
    print()
    print('All match accuracy')
    for key in accuracy_all:
        print(key, accuracy_all[key])
    print()
    print('F1 score')
    for key in f1:
        print(key, f1[key])
