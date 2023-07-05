from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .coreml import CoremlEncoder
from timeit import default_timer as timer

#--- copied from ml-ane-transformers and ANE-Optimized-Whisper-OpenAI

from ane_transformers.reference.layer_norm import LayerNormANE

# Note: Original implementation of distilbert uses an epsilon value of 1e-12
# which is not friendly with the float16 precision that ANE uses by default
EPS = 1e-7

# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix +
               'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix +
                                                                  'weight']
    return state_dict


class LayerNormANE(LayerNormANE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(
            correct_for_bias_scale_order_inversion)

def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """ Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    for k in state_dict:
        is_linear = all(substr in k for substr in ['attn.', '.weight']) or all(substr in k for substr in ['mlp.', '.weight'])
        if is_linear:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]
#--- copied ended

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.dim_per_head = (n_state// n_head)
        self.qk_scale = (self.dim_per_head) ** -0.5

        self.query = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.key = nn.Conv2d(n_state, n_state, kernel_size=1, bias=False)
        self.value = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.out = nn.Conv2d(n_state, n_state, kernel_size=1)

    def forward(self, x: Tensor):
        #bs, dim, dummy, seqlen = x.shape
        #print("x", x.shape)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        mh_q = q.split(self.dim_per_head, dim=1)
        mh_k = k.transpose(1, 3).split(self.dim_per_head, dim=3)
        mh_v = v.split(self.dim_per_head, dim=1)

        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki]) * self.qk_scale
            for qi, ki in zip(mh_q, mh_k)
        ]

        attn_weights = [aw.softmax(dim=1) for aw in attn_weights]
        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi)
            for wi, vi in zip(attn_weights, mh_v)
        ]

        attn = torch.cat(attn, dim=1)

        attn = self.out(attn)
        #print("attn", attn.shape)

        return attn

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNormANE(n_state, eps=EPS)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Conv2d(n_state, n_mlp, kernel_size=1), nn.GELU(), nn.Conv2d(n_mlp, n_state, kernel_size=1)
        )
        self.mlp_ln = LayerNormANE(n_state, eps=EPS)
        self.n_state = n_state

    def forward(self, x: Tensor):
        # a workaround for fixing coreml conversion time grows a lot on small model
        # it's related to LayerNorm and LayerCount
        #x = torch.cat([x, torch.zeros(1, 1, self.n_state)], dim=1).split(1500, dim=1)[0]
        x = x + self.attn(self.attn_ln(x))

        #x = torch.cat([x, torch.zeros(1, 1, self.n_state)], dim=1).split(1500, dim=1)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNormANE(n_state, eps=EPS)
        self.coremlEncoder = None
        self.n_state = n_state

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        ############################
        #if self.coremlEncoder == None:
        #    self.coremlEncoder = CoremlEncoder(self.n_state)
        #return self.coremlEncoder.predictWith(x)
        ###########################3

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding)

        x = x.transpose(1, 2).unsqueeze(2)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)

        x = x.squeeze(2).transpose(1,2)
        #print(x.shape)
        return x
