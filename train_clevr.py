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

from datasets import CLEVRDataset
from functools import partial
from networks import EmbeddingNet, ReadoutNet
from tensorboardX import SummaryWriter
from train_utils import train_one_epoch, eval_one_epoch, train_ddp
import argparse
import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pred_types = {'left': 90, 'right': 90, 'front': 90, 'behind': 90}


def step(data, model, head):
    img, obj_patches, target, mask = data
    emb, attn = model(img, obj_patches)
    logits = head(emb)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none'
    )
    loss = (loss * mask).sum() / mask.sum()
    return logits, loss


def calc_acc(data, logits, pred_types):
    img, obj_patches, target, mask = data
    pred = (logits.detach() > 0).int()
    pred = pred.reshape(logits.shape[0], len(pred_types), -1)
    mask = mask.reshape(logits.shape[0], len(pred_types), -1)
    target = target.int().reshape(logits.shape[0], len(pred_types), -1)
    acc = ((pred == target) * mask).sum(dim=-1) / mask.sum(dim=-1)
    acc = acc.nansum(dim=0) / (acc.shape[0] - acc.isnan().sum(dim=0)) * 100
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
    writer.add_scalar(f'{split}/loss', avg_loss, global_step)
    writer.add_scalar(f'{split}/accuracy', avg_acc.mean().item(), global_step)
    for a, name in zip(avg_acc, pred_types.keys()):
        writer.add_scalar(f'{split}/accuracy_{name}', a.item(), global_step)


def train(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group('nccl', rank=rank, world_size=args.n_gpu)
    torch.cuda.set_device(rank)

    train_data = CLEVRDataset(
        f'{args.data_dir}/trainA.h5',
        f'{args.data_dir}/objects.h5',
        args.max_nobj, rand_patch=True
    )

    valid_data = CLEVRDataset(
        f'{args.data_dir}/valA.h5',
        f'{args.data_dir}/objects.h5',
        args.max_nobj, rand_patch=False
    )

    model = EmbeddingNet(
        (args.img_w, args.img_h), args.patch_size, args.max_nobj,
        args.width, args.layers, args.heads
    )
    head = ReadoutNet(args.width, args.d_hidden, 0, len(pred_types))
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

    train_one = partial(
        train_one_epoch, pred_types, step, calc_acc,
        log, args.print_freq, None, 0, writer
    )
    eval_one = partial(
        eval_one_epoch, pred_types, step, calc_acc, log, None, writer
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
    parser.add_argument('--max_nobj', type=int, default=10)
    parser.add_argument('--img_h', type=int, default=320)
    parser.add_argument('--img_w', type=int, default=480)
    parser.add_argument('--n_worker', type=int, default=2)
    # Model
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--d_hidden', type=int, default=512)
    # Training
    parser.add_argument('--log_dir')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--port', default='12345')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=40)
    parser.add_argument('--print_freq', type=int, default=50)
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
