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

from datasets import RegressionDataset
from networks import EmbeddingNet, ReadoutNet
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--split')
    parser.add_argument('--obj_file')
    parser.add_argument('--ee', action='store_true')
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--n_objects', type=int, default=4)
    parser.add_argument('--n_views', type=int, default=3)
    parser.add_argument('--objects', nargs='+')
    parser.add_argument('--colors', nargs='+')
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    # Evaluation
    parser.add_argument('--model_checkpoint')
    parser.add_argument('--head_checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_worker', type=int, default=2)
    args = parser.parse_args()

    loaders = []
    for i in range(args.n_views):
        data = RegressionDataset(
            args.data_dir, args.split, args.obj_file, args.colors,
            randpatch=False, view=args.n_views, randview=True,
            ee=args.ee, dist=args.dist
        )
        loaders.append(DataLoader(
            data, args.batch_size, pin_memory=True, num_workers=args.n_worker
        ))

    if args.ee:
        unary = 1 if args.dist else 3
        binary = 0
    else:
        unary = 0
        binary = 1 if args.dist else 3

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, args.n_objects,
        args.width, args.layers, args.heads
    )
    head = ReadoutNet(args.width, args.d_hidden, unary, binary)

    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    checkpoint = torch.load(args.head_checkpoint, map_location='cpu')
    head.load_state_dict(checkpoint)
    model = model.cuda().eval()
    head = head.cuda().eval()

    predictions = []
    targets = []
    loaders.insert(0, tqdm(range(len(loaders[0]))))
    for data in zip(*loaders):
        data = data[1:]
        batch_size = data[0][0].shape[0]
        logits = 0
        for img, obj_patches, target in data:
            with torch.no_grad():
                img = img.cuda()
                obj_patches = obj_patches.cuda()
                emb, attn = model(img, obj_patches)
                logits += head(emb)
        logits = logits.reshape(logits.shape[0], target.shape[2], -1).transpose(1, 2)
        predictions.append(logits.cpu().numpy() / args.n_views)
        targets.append(target.numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    dist = np.sqrt(((targets - predictions) ** 2).sum(axis=-1))
    print('Euclidean error', dist.mean())
