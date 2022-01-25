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


def step(data, model, head):
    img, obj_patches, target = data
    emb, attn = model(img, obj_patches)
    logits = head(emb)
    target = target.transpose(1, 2).reshape(logits.shape[0], -1)
    loss = ((target - logits) ** 2).mean()
    return logits, loss


def calc_dist(data, logits, pred_types):
    img, obj_patches, target = data
    logits = logits.reshape(logits.shape[0], target.shape[2], -1).transpose(1, 2)
    dist = ((target - logits) ** 2).sum(dim=-1).sqrt().mean()
    return dist


def log(
        writer, global_step, split, epoch, idx, total,
        batch_time, data_time, avg_loss, avg_dist, pred_types
    ):
    print(
        f'Epoch {(epoch+1):02d} {split.capitalize()} {idx:04d}/{total:04d} '
        f'Batch time {batch_time:.3f} Data time {data_time:.3f} '
        f'Loss {avg_loss.item():.4f} Dist {avg_dist.item():.2f}'
    )
    writer.add_scalar(f'{split}/loss', avg_loss, global_step)
    writer.add_scalar(f'{split}/dist', avg_dist.item(), global_step)


def train(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group('nccl', rank=rank, world_size=args.n_gpu)
    torch.cuda.set_device(rank)

    train_data = RegressionDataset(
        args.data_dir, 'train', 'train_objects.h5',
        randpatch=True, view=args.n_views, randview=True,
        ee=args.ee, dist=args.dist
    )
    valid_data = RegressionDataset(
        args.data_dir, 'valid', 'train_objects.h5',
        randpatch=False, view=args.n_views, randview=True,
        ee=args.ee, dist=args.dist
    )

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
    optimizer = torch.optim.Adam(head.parameters(), args.lr)

    writer = None
    if rank == 0:
        writer = SummaryWriter(args.log_dir)
        json.dump(args.__dict__, open(f'{args.log_dir}/args.txt', 'w'), indent=4)

    train_one = partial(
        train_one_epoch, None, step, calc_dist, log,
        args.print_freq, None, 0, writer
    )
    eval_one = partial(
        eval_one_epoch, None, step, calc_dist, log, None, writer
    )
    train_ddp(
        train_data, valid_data, args.batch_size, args.n_worker, model, head,
        optimizer, 0, args.n_epoch, train_one, eval_one,
        args.eval_freq, args.save_freq, args.log_dir
    )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_dir')
    parser.add_argument('--ee', action='store_true')
    parser.add_argument('--dist', action='store_true')
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
    # Training
    parser.add_argument('--model_checkpoint')
    parser.add_argument('--log_dir')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_worker', type=int, default=2)
    parser.add_argument('--port', default='12345')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=5)
    args = parser.parse_args()

    mp.spawn(
        train,
        args=(args,),
        nprocs=args.n_gpu,
        join=True
    )
