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

from time import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist


def train_one_epoch(
        pred_types, step_fn, metric_fn, log_fn, log_freq, plot_fn,
        plot_freq, writer, epoch, loader, model, head, optimizer
    ):
    rank = dist.get_rank()
    ws = dist.get_world_size()
    batch_time, data_time, total_loss, total_metric = 0, 0, 0, 0

    model.train()
    start = time()
    for batch_idx, data in enumerate(loader):
        data = [d.cuda() for d in data]
        data_time += time() - start

        logits, loss = step_fn(data, model, head)
        metric = metric_fn(data, logits, pred_types)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_metric += metric

        batch_time += time() - start
        start = time()

        if (batch_idx + 1) % log_freq == 0:
            dist.reduce(total_loss, 0)
            dist.reduce(total_metric, 0)
            if rank == 0:
                global_step = epoch * len(loader) + batch_idx + 1
                log_fn(
                    writer, global_step, 'train', epoch, batch_idx + 1,
                    len(loader), batch_time / log_freq, data_time / log_freq,
                    total_loss / ws /log_freq, total_metric / ws / log_freq,
                    pred_types
                )
            batch_time, data_time, total_loss, total_metric = 0, 0, 0, 0

        if rank == 0 and plot_fn is not None and (batch_idx + 1) % plot_freq == 0:
            fig = plot_fn(data, logits)
            writer.add_figure('train', fig, global_step)


def eval_one_epoch(
        pred_types, step_fn, metric_fn, log_fn, plot_fn, writer,
        epoch, loader, model, head, global_step
    ):
    rank = dist.get_rank()
    batch_time, data_time, total_loss, total_metric, total = 0, 0, 0, 0, 0

    model.eval()
    start = time()
    for data in loader:
        data = [d.cuda() for d in data]
        data_time += time() - start

        with torch.no_grad():
            logits, loss = step_fn(data, model, head)
            metric = metric_fn(data, logits, pred_types)

        total_loss += loss * logits.shape[0]
        total_metric += metric * logits.shape[0]
        total += logits.shape[0]

        batch_time += time() - start
        start = time()

    total = torch.tensor(total).cuda()
    dist.reduce(total_loss, 0)
    dist.reduce(total_metric, 0)
    dist.reduce(total, 0)

    if rank == 0:
        log_fn(
            writer, global_step, 'valid', epoch, len(loader),
            len(loader), batch_time / len(loader), data_time / len(loader),
            total_loss / total, total_metric / total, pred_types
        )

        if plot_fn is not None:
            fig = plot_fn(data, logits)
            writer.add_figure('valid', fig, global_step)


def train_ddp(
        train_data, valid_data, batch_size, n_worker, model, head, optimizer,
        init_epoch, n_epoch, train_one, eval_one, eval_freq, save_freq, log_dir,
        head_only=False
    ):
    train_sampler = DistributedSampler(train_data, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_data,
        batch_size,
        sampler=train_sampler,
        num_workers=n_worker,
        pin_memory=True
    )
    valid_sampler = DistributedSampler(valid_data, shuffle=False, drop_last=False)
    valid_loader = DataLoader(
        valid_data,
        batch_size,
        sampler=valid_sampler,
        num_workers=n_worker,
        pin_memory=True
    )

    rank = dist.get_rank()
    model = DDP(model.cuda(), device_ids=[rank])
    head = DDP(head.cuda(), device_ids=[rank])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for epoch in range(init_epoch, n_epoch):
        train_sampler.set_epoch(epoch)
        train_one(epoch, train_loader, model, head, optimizer)

        if (epoch + 1) % eval_freq == 0:
            global_step = (epoch + 1) * len(train_loader)
            valid_sampler.set_epoch(epoch)
            eval_one(epoch, valid_loader, model, head, global_step)

        if (epoch + 1) % save_freq == 0 and rank == 0:
            if head_only:
                checkpoint = head.module.state_dict()
            else:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': model.module.state_dict(),
                    'head': head.module.state_dict(),
                    'optimizer' : optimizer.state_dict()
                }
            torch.save(checkpoint, f'{log_dir}/epoch_{(epoch + 1):d}.pth')
