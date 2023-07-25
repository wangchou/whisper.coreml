from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from timeit import default_timer as timer

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

# https://github.com/apple/coremltools/issues/1900
def speedup_conversion_workaround(x: Tensor, n_state: int):
    # (1, 1500, 384) -> (1, 1501, 384) -> (1, 1500, 384)
    return torch.cat([x, torch.empty(1, 1, n_state)], dim=1).split(1500, dim=1)[0]

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.dim_per_head = (n_state// n_head)
        self.qk_scale = (self.dim_per_head) ** -0.5
        self.n_state = n_state

        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x: Tensor):
        q = self.query(x)
        k = self.key(x) * self.qk_scale
        v = self.value(x)

        # (1, 1500, 384) -> (1, 384, 1, 1500)
        q = q.transpose(1, 2).unsqueeze(2)
        k = k.unsqueeze(2)
        v = v.transpose(1, 2).unsqueeze(2)

        # multi-head
        mh_q = q.split(self.dim_per_head, dim=1)
        mh_k = k.split(self.dim_per_head, dim=3)
        mh_v = v.split(self.dim_per_head, dim=1)

        mh_wv = []
        for h in range(self.n_head):
            w = torch.einsum('bchq,bkhc->bkhq', mh_q[h], mh_k[h]).softmax(dim=1)
            mh_wv.append(torch.einsum('bkhq,bchk->bchq', w, mh_v[h]))

        # (1, 384, 1, 1500) -> (1, 1500, 384)
        wv = torch.cat(mh_wv, dim=1).squeeze(2).transpose(1,2)

        return self.out(wv)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state, eps=1e-7)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state, eps=1e-7)
        self.n_state = n_state

    def forward(self, x: Tensor):
        x = speedup_conversion_workaround(x, self.n_state)
        x = x + self.attn(self.attn_ln(x))
        x = speedup_conversion_workaround(x, self.n_state)
        x = x + self.mlp(self.mlp_ln(x))
        return x

class AudioEncoder(nn.Module):
    def __init__(
            self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, use_coreml: bool, modelName
    ):
        super().__init__()
        if not use_coreml:
            self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
            self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
            )
            self.ln_post = nn.LayerNorm(n_state, eps=1e-7)
        self.coremlEncoder = None
        self.n_state = n_state
        self.n_layer = n_layer
        self.from_block_idx = 0
        self.use_coreml = use_coreml
        self.modelName = modelName

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        ############################
        if self.use_coreml:
            self.coreml.loadEncoder()
            return self.coreml.encoderPredict(x)
        ############################

        self.from_block_idx = 0
        for i in range(0, self.n_layer, 12):
           self.from_block_idx = i
           x = self.block12(x)

        return x

    # divided sub-model for speed up ANECompilerService
    def block12(self, x: Tensor):
        if self.from_block_idx == 0:
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            x = (x + self.positional_embedding)

        for i in range(self.from_block_idx, min(self.from_block_idx + 12, self.n_layer)):
            x = self.blocks[i](x)

        if self.from_block_idx + 12 >= self.n_layer:
            x = self.ln_post(x)

        return x
