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

from collections import OrderedDict
import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[:x.shape[0], :x.shape[0]].to(x.device)
        return self.attn(x, x, x, attn_mask=attn_mask)

    def forward(self, x: tuple):
        x, weights = x
        attn, attn_weights = self.attention(self.ln_1(x))
        if weights is None:
            weights = attn_weights.unsqueeze(1)
        else:
            weights = torch.cat([weights, attn_weights.unsqueeze(1)], dim=1)
        x = x + attn
        x = x + self.mlp(self.ln_2(x))
        return x, weights


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks((x, None))


class EmbeddingNet(nn.Module):
    def __init__(
            self, input_dim: tuple, patch_size: int, n_objects: int,
            width: int, layers: int, heads: int
        ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width, kernel_size=patch_size,
            stride=patch_size, bias=False
        )

        scale = width ** -0.5
        n_patches = (input_dim[0] // patch_size) * (input_dim[1] // patch_size)
        self.positional_embedding = nn.Parameter(scale * torch.randn(n_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        seq_len = n_patches + n_objects
        attn_mask = torch.zeros(seq_len, seq_len)
        attn_mask[:, -n_objects:] = -float("inf")
        attn_mask.fill_diagonal_(0)

        self.transformer = Transformer(width, layers, heads, attn_mask)
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, objs: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        batch_size, n_obj = objs.shape[:2]
        objs = objs.reshape(-1, *objs.shape[2:])
        objs = self.conv1(objs)
        objs = objs.reshape(batch_size, n_obj, -1) # shape = [*, n_obj, width]

        x = x + self.positional_embedding[1:]
        objs = objs + self.positional_embedding[:1]
        x = torch.cat([x, objs], dim=1)  # shape = [*, grid ** 2 + n_obj, width]
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, weights = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, -objs.shape[1]:, :])

        return x, weights


class ReadoutNet(nn.Module):
    def __init__(self, d_input, d_hidden, n_unary, n_binary):
        super().__init__()
        self.n_unary = n_unary
        self.n_binary = n_binary
        self.d_hidden = d_hidden
        for i in range(n_unary):
            setattr(self, f'unary{i}', self.get_head(d_input, d_hidden, 1))
        for i in range(n_binary):
            setattr(self, f'binary{i}', self.get_head(d_input, d_hidden, 2))

    def get_head(self, d_input, d_hidden, n_args):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, 1)
            )
        else:
            head = nn.Linear(d_input * n_args, d_hidden)
        return head

    def forward(self, x: torch.Tensor):
        n_obj = x.shape[1]
        y = [getattr(self, f'unary{i}')(x) for i in range(self.n_unary)]
        x1 = x.repeat(1, n_obj - 1, 1)
        x2 = torch.cat([x.roll(-i, dims=1) for i in range(1, n_obj)], dim=1)
        x = torch.cat([x1, x2], dim=-1)
        y += [getattr(self, f'binary{i}')(x) for i in range(self.n_binary)]
        y = torch.cat(y, dim=1).squeeze(-1)
        return y
