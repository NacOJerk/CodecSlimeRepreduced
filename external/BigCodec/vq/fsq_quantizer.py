"""FSQ wrapper with the same call surface as ResidualVQ inside CodecDecoder.

Used by CodecDecoder when quantizer_type='fsq'. Returns (out, indices,
commit_loss) where commit_loss is always a zero scalar (FSQ has no learnable
codebook).

BigCodec's CodecDecoder passes (B, C, T) (channels-first) into the quantizer,
while the underlying vector_quantize_pytorch.FSQ expects (B, T, C). We detect
which layout the caller used by checking which axis matches input_dim and
transpose at the boundary, restoring the original layout on the way out.
"""
from math import prod

import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ


class FSQQuantizer(nn.Module):
    def __init__(self, input_dim: int, fsq_dim: int, levels):
        super().__init__()
        self.input_dim = input_dim
        self.levels = list(levels)
        self.codebook_size = prod(self.levels)
        self.proj_in = nn.Linear(input_dim, fsq_dim)
        self.proj_out = nn.Linear(fsq_dim, input_dim)
        self.fsq = FSQ(levels=self.levels)

    def forward(self, x: torch.Tensor):
        channels_first = x.shape[1] == self.input_dim and x.shape[-1] != self.input_dim
        if channels_first:
            x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        z = self.proj_in(x)
        z_q, indices = self.fsq(z)
        out = self.proj_out(z_q)
        if channels_first:
            out = out.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        # commit_loss is shape (1,) so sum(commit_loss) in lightning_module
        # matches ResidualVQ's per-quantizer list/tensor convention.
        commit_loss = x.new_zeros(1)
        return out, indices, commit_loss
