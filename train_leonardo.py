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

from datasets import LeonardoDataset, build_predicates, denormalize_rgb
from functools import partial
from matplotlib import pyplot as plt
from networks import EmbeddingNet, ReadoutNet
from tensorboardX import SummaryWriter
from train_utils import train_one_epoch, eval_one_epoch, train_ddp
import argparse
import json
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

unary_pred = [
    'on_surface(%s, left)', 'on_surface(%s, right)', 'on_surface(%s, far)',
    'on_surface(%s, center)', 'has_obj(robot, %s)', 'top_is_clear(%s)',
    'in_approach_region(robot, %s)'
]
binary_pred = ['stacked(%s, %s)', 'aligned_with(%s, %s)']


def step(use_gripper, data, model, head):
    img, obj_patches, gripper, target = data
    emb, attn = model(img, obj_patches)
    if use_gripper:
        emb = torch.cat(
            [emb, gripper[:, None, None].expand(-1, emb.shape[1], -1)], dim=-1
        )
    logits = head(emb)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    return logits, loss


def calc_acc(data, logits, pred_types):
    img, obj_patches, gripper, target = data
    pred = (logits.detach() > 0).int()
    acc = (pred == target.int()).sum(dim=0) / logits.shape[0] * 100
    return acc


def log(
        writer, global_step, split, epoch, idx, total,
        batch_time, data_time, avg_loss, avg_acc, pred_types=None
    ):
    print(
        f'Epoch {(epoch+1):02d} {split.capitalize()} {idx:04d}/{total:04d} '
        f'Batch time {batch_time:.3f} Data time {data_time:.3f} '
        f'Loss {avg_loss.item():.4f} Accuracy {avg_acc.mean().item():.2f}'
    )
    acc = [a.mean() for a in avg_acc.split(list(pred_types.values()))]
    writer.add_scalar(f'{split}/loss', avg_loss, global_step)
    writer.add_scalar(f'{split}/accuracy', avg_acc.mean().item(), global_step)
    for a, name in zip(acc, pred_types.keys()):
        writer.add_scalar(f'{split}/accuracy_{name}', a.item(), global_step)


def plot(predicates, n_plot, data, logits):
    img, obj_patches, gripper, target = data
    patch_size = obj_patches.shape[-1]
    img_with_obj = []
    for i, o in zip(img[:n_plot], obj_patches[:n_plot]):
        obj_panel = np.full((patch_size+4, i.shape[2], 3), 255, dtype=np.uint8)
        for j in range(obj_patches.shape[1]):
            obj_panel[4:, j*2*patch_size:(j*2+1)*patch_size] = \
                np.array(denormalize_rgb(o[j]))
        img_with_obj.append(
            np.concatenate([np.array(denormalize_rgb(i)), obj_panel], axis=0)
        )
    pred = (logits.detach() > 0).int()

    fig = plt.figure(figsize=(n_plot * 4, 9))
    for i in range(n_plot):
        plt.subplot(2, n_plot, i + 1)
        plt.imshow(img_with_obj[i])
        plt.axis('off')
        plt.subplot(2, n_plot, n_plot + i + 1)
        j = 0
        for name, p, t in zip(predicates, pred[i], target[i]):
            if t or p:
                if t:
                    c = 'k' if p else 'r'
                else:
                    c = 'b'
                plt.text(
                    0.5, 0.9 - j * 0.08, name, color=c,
                    fontsize=10, ha='center', va='center'
                )
                j += 1
        plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    return fig


def train(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group('nccl', rank=rank, world_size=args.n_gpu)
    torch.cuda.set_device(rank)

    pred_types = {}
    for pred in unary_pred:
        prefix = pred.split('(')[0]
        if prefix in pred_types:
            pred_types[prefix] += args.n_objects
        else:
            pred_types[prefix] = args.n_objects
    for pred in binary_pred:
        prefix = pred.split('(')[0]
        pred_types[prefix] = args.n_objects * (args.n_objects - 1)

    objects = [f'object{i:02d}' for i in range(args.n_objects)]
    predicates = build_predicates(objects, unary_pred, binary_pred)

    train_data = LeonardoDataset(
        args.data_dir, 'train', predicates, 'train_objects.h5',
        randpatch=True, view=args.n_views, randview=True, gripper=args.gripper
    )

    valid_data = LeonardoDataset(
        args.data_dir, 'valid', predicates, 'train_objects.h5',
        randpatch=False, view=args.n_views, randview=True, gripper=args.gripper
    )

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, args.n_objects,
        args.width, args.layers, args.heads
    )
    out_dim = args.width + 1 if args.gripper else args.width
    head = ReadoutNet(out_dim, args.d_hidden, len(unary_pred), len(binary_pred))
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), args.lr
    )

    init_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        head.load_state_dict(checkpoint['head'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    writer = None
    if rank == 0:
        writer = SummaryWriter(args.log_dir)
        json.dump(args.__dict__, open(f'{args.log_dir}/args.txt', 'w'), indent=4)

    step_fn = partial(step, args.gripper)
    plot_fn = partial(plot, predicates, args.n_plot)
    train_one = partial(
        train_one_epoch, pred_types, step_fn, calc_acc, log,
        args.print_freq, plot_fn, args.plot_freq, writer
    )
    eval_one = partial(
        eval_one_epoch, pred_types, step_fn, calc_acc, log, plot_fn, writer
    )
    train_ddp(
        train_data, valid_data, args.batch_size, args.n_worker, model, head,
        optimizer, init_epoch, args.n_epoch, train_one, eval_one,
        args.eval_freq, args.save_freq, args.log_dir
    )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)
    parser.add_argument('--n_objects', type=int, default=4)
    parser.add_argument('--n_views', type=int, default=3)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    parser.add_argument('--gripper', action='store_true')
    # Training
    parser.add_argument('--log_dir')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_worker', type=int, default=2)
    parser.add_argument('--port', default='12345')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=40)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--plot_freq', type=int, default=500)
    parser.add_argument('--n_plot', type=int, default=4)
    parser.add_argument('--eval_freq', type=int, default=2)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--resume')
    args = parser.parse_args()

    mp.spawn(
        train,
        args=(args,),
        nprocs=args.n_gpu,
        join=True
    )
