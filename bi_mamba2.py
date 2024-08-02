"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

Device = torch.device


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                    self.pad_vocab_size_multiple
                    - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2(nn.Module):
    def __init__(self, d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 vocab_size: int = 50277,
                 pad_vocab_size_multiple: int = 16, ):
        super().__init__()
        args = Mamba2Config(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, vocab_size, pad_vocab_size_multiple)
        self.args = args
        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, ))
        self.A_log = nn.Parameter(torch.empty(args.nheads, ))
        self.D = nn.Parameter(torch.empty(args.nheads, ))
        self.norm = RMSNorm(args.d_inner, )
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, )

    def forward(self, u: Tensor, h=None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        if h:
            return self.step(u, h)

        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=x.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: Tensor, h: InferenceCache):
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)

class BaseBiMaba2(nn.Module):
    def __init__(self, cin, cout, mamba_dim, **mamba2_args):
        super().__init__()
        assert mamba_dim % 64 == 0, "cmid 必须是64的倍数"
        self.fc_in = nn.Linear(cin, mamba_dim, bias=False)  # 调整通道数到cmid
        self.mamba2_for = Mamba2(mamba_dim, **mamba2_args)  # 正向
        self.mamba2_back = Mamba2(mamba_dim, **mamba2_args)  # 负向
        self.fc_out = nn.Linear(mamba_dim, cout, bias=False)  # 调整通道数到cout


class BiMamba2_1d(BaseBiMaba2):
    def __init__(self, cin, cmid, cout, **mamba2_args):
        super().__init__(cin, cmid, cout, **mamba2_args)

    def forward(self, x):
        l = x.shape[2]
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # 将 l , pad到4的倍数, [b, c64,l4]
        x = rearrange(x, 'b c l-> b l c')  # 转成 1d 信号 [b, d4*w4*h4, c64]
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = rearrange(x, 'b l c -> b c l')  # 转成 2d 图片[b, l64, c64]
        x = x[:, :, :l]  # 截取原图大小
        return x


class BiMamba2_2d(BaseBiMaba2):
    def __init__(self, cin,  cout,mamba_dim, **mamba2_args):
        super().__init__(cin, cout, mamba_dim, **mamba2_args)

    def forward(self, x):
        h, w = x.shape[2:]
        x = F.pad(x, (0, (8 - x.shape[3] % 8) % 8,
                      0, (8 - x.shape[2] % 8) % 8)
                  )  # 将 h , w  pad到8的倍数, [b, c64, h8, w8]
        h8, w8 = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')  # 转成 1d 信号 [b, w8*h8, c64]
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = rearrange(x, 'b (h w) c -> b c h w', h=h8)  # 转成 2d 图片[b, w8*h8, c64]
        x = x[:, :, :h, :w]  # 截取原图大小
        return x


class BiMamba2_3d(BaseBiMaba2):
    def __init__(self, cin,  cout,mamba_dim, **mamba2_args):
        super().__init__(cin, cout, mamba_dim, **mamba2_args)



    def forward(self, x):
        d, h, w = x.shape[2:]
        x = F.pad(x, (0, (4 - x.shape[4] % 4) % 4,
                      0, (4 - x.shape[3] % 4) % 4,
                      0, (4 - x.shape[2] % 4) % 4)
                  )  # 将 d, h, w , pad到4的倍数, [b, c64,d4, h4, w4]
        d4, h4, w4 = x.shape[2:]
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # 转成 1d 信号 [b, d4*w4*h4, c64]
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = rearrange(x, 'b (d h w) c -> b c h w', d=d4, h=h4, w=w4)  # 转成 2d 图片[b, d4*w4*h4, c64]
        x = x[:, :, :d, :h, :w]  # 截取原图大小
        return x


class BiMamba2(BaseBiMaba2):
    def __init__(self, cin,  cout, mamba_dim, **mamba2_args):
        super().__init__(cin, cout, mamba_dim, **mamba2_args)


    def forward(self, x):
        size = x.shape[2:]
        x = torch.flatten(x, 2)  # b c size
        l = x.shape[2]
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # 将 l , pad到4的倍数, [b, c64,l4]
        x = rearrange(x, 'b c l-> b l c')  # 转成 1d 信号 [b, d4*w4*h4, c64]
        x = self.fc_in(x)  # 调整通道数为目标通道数
        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))
        x2 = x2.flip(1)
        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数
        x = rearrange(x, 'b l c -> b c l')  # 转成 2d 图片[b, l64, c64]
        x = x[:, :, :l]  # 截取原图大小
        x = torch.unflatten(x, 2, size)
        return x

if __name__ == '__main__':
    net = BiMamba2(64, 128, 64).cuda()
    x1 = torch.randn(1, 64, 32).cuda()
    x2 = torch.randn(1, 64, 32, 77).cuda()
    x3 = torch.randn(1, 64, 32, 77, 25).cuda()
    y1 = net(x1)
    print(y1.shape)
    y2 = net(x2)
    print(y2.shape)
    y3 = net(x3)
    print(y3.shape)
