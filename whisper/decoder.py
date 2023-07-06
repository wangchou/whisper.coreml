import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function
from .coreml import CoremlDecoder, CoremlDecoder256
from timeit import default_timer as timer

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.qk_scale = (n_state // n_head) ** -0.5

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        qk_mask: Optional[Tensor] = None,
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
    ):
        q = self.query(x)

        # new part of k, without previous cache
        # zeros is for dummy output, because coreml don't accept None as return value
        new_k = torch.zeros(1)
        new_v = torch.zeros(1)
        if cache_k is None or xa is None:
            k = self.key(x if xa is None else xa.split(1)[0]) # cross_kv is the same for all beams
            v = self.value(x if xa is None else xa.split(1)[0])

            new_k = k
            new_v = v
            # only for self masked attention
            if qk_mask is not None and cache_k is not None:
                k = torch.cat([cache_k, k], dim=1)
                v = torch.cat([cache_v, v], dim=1)
        else:
            # for cross-attention
            k = cache_k
            v = cache_v

        wv, qk = self.qkv_attention(q, k, v, qk_mask)
        return self.out(wv), qk.detach(), new_k, new_v

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, qk_mask: Optional[Tensor] = None
    ):
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k * self.qk_scale
        if qk_mask is not None:
            qk = qk + qk_mask
        qk = qk

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        qk_mask: Optional[Tensor] = None,
        mk: Optional[Tensor] = None,
        mv: Optional[Tensor] = None,
        ck: Optional[Tensor] = None,
        cv: Optional[Tensor] = None,
    ):
        x_out, masked_qk, new_mk, new_mv = self.attn(self.attn_ln(x), qk_mask=qk_mask, cache_k=mk, cache_v=mv)
        x = x + x_out
        cross_qk = new_ck = new_cv = None
        if self.cross_attn:
            x_out, cross_qk, new_ck, new_cv = self.cross_attn(self.cross_attn_ln(x), xa, cache_k=ck, cache_v=cv)
            x = x + x_out
        x = x + self.mlp(self.mlp_ln(x))
        return x, cross_qk, new_mk, new_mv, new_ck, new_cv

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)
        self.n_vocab = n_vocab
        self.n_state = n_state
        self.n_layer = n_layer
        self.n_head = n_head
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.coremlDecoder = None
        self.coremlDecoder256 = None

        # max token len for first time = max_prefix_len(224) + sot_len(3)
        # not sure why... decoder227 is slower than decoder256
        self.max_n_ctx_for_1st = 256

    def forward(self, x: Tensor, xa: Tensor,
                text_offset: Tensor,
                isNewCKV: Tensor,
                masked_kv_caches: Optional[Tensor] = None,
                cross_kv_caches: Optional[Tensor] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = text_offset
        n_batch, n_ctx = x.shape
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + n_ctx]
        x = x.to(xa.dtype)

        if text_offset == 0:
            max_n_ctx = self.max_n_ctx_for_1st
            qk_mask = (torch.ones(max_n_ctx, max_n_ctx) * -np.inf).triu_(1)
            qk_mask[:, n_ctx:] = -np.inf
            ## fix shape by appending zeros to max_n_ctx
            x = torch.cat([x, torch.zeros(n_batch, max_n_ctx-n_ctx, self.n_state)], dim=1)
            x, cross_qks, new_masked_kv_caches, new_cross_kv_caches = self.forwardBlocks(x,
                                                                                         xa,
                                                                                         qk_mask,
                                                                                         masked_kv_caches,
                                                                                         cross_kv_caches,
                                                                                         isNewCKV)
            x = x[:,:n_ctx, :]
            cross_qks = cross_qks[:, :, :n_ctx, :]
            logits = (
                x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
            ).float()
        else:
            qk_mask = torch.cat([torch.zeros((1,text_offset)),
                                 torch.ones((1, 448-text_offset)) * -np.inf,
                                 torch.FloatTensor([[0]])],
                                dim=1)
            logits, cross_qks, new_masked_kv_caches, new_cross_kv_caches = self.forwardBlocks(x, xa, qk_mask, masked_kv_caches, cross_kv_caches, isNewCKV)


        return logits, cross_qks, new_masked_kv_caches, new_cross_kv_caches

    def forwardBlocks(self, x: Tensor, xa: Tensor,
                      qk_mask: Optional[Tensor] = None,
                      masked_kv_caches: Optional[Tensor] = None,
                      cross_kv_caches: Optional[Tensor] = None,
                      isNewCKV: Optional[Tensor] = None # only for coremlDecoder1
                      ):

        ############################
        # Coreml Decoder part
        if masked_kv_caches is not None and x.shape[1] == 1:
            if self.coremlDecoder == None:
                self.coremlDecoder = CoremlDecoder(self.n_layer, self.n_state, self.n_head)
            return self.coremlDecoder.predictWith(x, xa, qk_mask, masked_kv_caches, cross_kv_caches, isNewCKV)

        elif x.shape[0] == 5 and x.shape[1] == 256:
            if self.coremlDecoder256 == None:
                self.coremlDecoder256 = CoremlDecoder256(self.n_layer, self.n_state, self.n_head)
            return self.coremlDecoder256.predictWith(x, xa, qk_mask)
        ############################

        cross_qks = []
        new_masked_kv_caches = []
        new_cross_kv_caches = []
        for layer_idx, block in enumerate(self.blocks):
            # mk = masked_key_cache, ck = cross_key_cache
            mk = mv = ck = cv = None
            if masked_kv_caches is not None:
                mk = masked_kv_caches[layer_idx*2]
                mv = masked_kv_caches[layer_idx*2 + 1]

            if cross_kv_caches is not None:
                ck = cross_kv_caches[layer_idx*2]
                cv = cross_kv_caches[layer_idx*2 + 1]

            x, cross_qk, new_mk, new_mv, new_ck, new_cv = block(x, xa, qk_mask, mk, mv, ck, cv)

            cross_qks.append(cross_qk)
            new_masked_kv_caches.append(new_mk)
            new_masked_kv_caches.append(new_mv)
            new_cross_kv_caches.append(new_ck)
            new_cross_kv_caches.append(new_cv)

        cross_qks = torch.cat(cross_qks)
        new_masked_kv_caches = torch.stack(new_masked_kv_caches)
        new_cross_kv_caches = torch.stack(new_cross_kv_caches)

        x = self.ln(x)

        if qk_mask.shape[0] == 1: # decoder1
            splits = self.token_embedding.weight.split(self.n_vocab//5, dim=0)
            x = x.view(*x.shape[:2], self.n_state)
            logits = torch.cat([ x @ split.transpose(0,1) for split in splits], dim=2)

            # avoid return big unused output from coremlDecoder1
            new_cross_kv_caches = torch.zeros(1)
            cross_qks = torch.zeros(1)

            return logits, cross_qks, new_masked_kv_caches, new_cross_kv_caches
        else: # decoder256 and decoder call from add timestamp
            return x, cross_qks, new_masked_kv_caches, new_cross_kv_caches
