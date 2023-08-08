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
from timeit import default_timer as timer

def fuse_query_and_qk_scale(state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    for k in state_dict:
        if all(substr in k for substr in ['query']):
            state_dict[k] = state_dict[k] * 0.125 # qk_scale = 1/(64^0.5)

# use two-levels split to reduce edge degree in graph
# it fixes slow ANECompilerServie on medium/large model
def twoLevelSplit(caches: Optional[Tensor], level1_split_count: int):
    if caches is None:
        return None

    cache_splits = []
    for splits in caches.split(level1_split_count):
        cache_splits += splits.split(1)
    return cache_splits

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.dim_per_head = n_state // n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self._register_load_state_dict_pre_hook(fuse_query_and_qk_scale)

    def forward(
        self,
        x: Tensor,
        qk_mask: Tensor,
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        new_k = k
        new_v = v

        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)

        q = q.view(*q.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, 64).permute(0, 2, 3, 1)
        v = v.view(*v.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)

        qk = q @ k + qk_mask

        w = qk.softmax(dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        return self.out(wv), new_k, new_v

class CrossMultiHeadAttention(MultiHeadAttention):
    def forward(
        self,
        x: Tensor,
        cache_k: Tensor,
        cache_v: Tensor,
    ):
        q = self.query(x)
        k = cache_k
        v = cache_v

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k

        w = qk.softmax(dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

        return self.out(wv), qk

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.n_state = n_state
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            CrossMultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        qk_mask: Optional[Tensor] = None,
        mk: Optional[Tensor] = None,
        mv: Optional[Tensor] = None,
        ck: Optional[Tensor] = None,
        cv: Optional[Tensor] = None,
    ):
        x_out, new_mk, new_mv = self.attn(self.attn_ln(x), qk_mask=qk_mask, cache_k=mk, cache_v=mv)
        x = x + x_out
        cross_qk = new_ck = new_cv = None
        if self.cross_attn:
            x_out, cross_qk = self.cross_attn(self.cross_attn_ln(x), cache_k=ck, cache_v=cv)
            x = x + x_out

        x = x + self.mlp(self.mlp_ln(x))
        return x, cross_qk, new_mk, new_mv

class TextDecoder(nn.Module):
    def __init__(
            self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, use_coreml: bool, modelName: str
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        if not use_coreml:
            self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
                [
                    ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                    for _ in range(n_layer)
                ]
            )
        self.ln = nn.LayerNorm(n_state)
        self.n_vocab = n_vocab
        self.n_state = n_state
        self.n_layer = n_layer
        self.n_head = n_head
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.coremlDecoder = None
        self.coremlDecoder256 = None
        self.coremlCrossKV = None
        self.cross_kv_caches = None
        self.use_coreml = use_coreml
        self.modelName = modelName

        # max token len for first time = max_prefix_len(224) + sot_len(3)
        # note: not sure why... decoder227 is slower than decoder256
        self.max_n_ctx_for_1st = 256

        # copyed from Whisper.__init__, will used by word_timestamps
        all_heads = torch.zeros(
            n_layer, n_head, dtype=torch.bool
        )
        all_heads[n_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def crossKVCaches(self, xa: Tensor):
        if self.use_coreml:
            self.coreml.loadCrossKV()
            return self.coreml.crossKVPredict()

        cross_k_caches = []
        cross_v_caches = []
        for block in self.blocks:
            k = block.cross_attn.key(xa)
            k = k.view(*k.shape[:2], self.n_head, 64).permute(0, 2, 3, 1)
            cross_k_caches.append(k) #[1, 12, 64, 1500]

            v = block.cross_attn.value(xa)
            v = v.view(*v.shape[:2], self.n_head, 64).permute(0, 2, 1, 3)
            cross_v_caches.append(v) #[1, 12, 1500, 64]
        return torch.cat(cross_k_caches, dim=0), torch.cat(cross_v_caches, dim=0)

    def forward(self, x: Tensor,
                xa: Optional[Tensor],
                text_offset: Tensor,
                masked_kv_caches: Optional[Tensor] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = text_offset
        n_batch, n_ctx = x.shape

        x = self.token_embedding(x) + self.positional_embedding[offset : offset + n_ctx]

        if self.use_coreml:
            self.coreml.bs = x.shape[0]

        if text_offset == 0: # decoder256
            if xa is not None:
                self.cross_k_caches, self.cross_v_caches = self.crossKVCaches(xa)

            max_n_ctx = self.max_n_ctx_for_1st
            qk_mask = (torch.ones(max_n_ctx, max_n_ctx) * -np.inf).triu_(1)
            qk_mask[:, n_ctx:] = -np.inf
            x = torch.cat([x, torch.zeros(n_batch, max_n_ctx-n_ctx, self.n_state)], dim=1)

            # predict beam by beam for reuse decoder256 coreml model for bs=1 and bs=5
            x_bs = x.split(1)

            for bs_idx in range(len(x_bs)):
                # cross_qk only used for word level timestamp, its bs=1
                _x, _cross_qks, _new_masked_kv_caches = self.forwardBlocks(x_bs[bs_idx],
                                                                           qk_mask,
                                                                           masked_kv_caches,
                                                                           self.cross_k_caches,
                                                                           self.cross_v_caches,
                                                                           beam_idx=bs_idx)
                if bs_idx == 0:
                    x = _x
                    new_masked_kv_caches = _new_masked_kv_caches
                    cross_qks = _cross_qks
                else:
                    x = torch.cat([x, _x], dim=0)
                    if not self.use_coreml:
                        new_masked_kv_caches = torch.cat([new_masked_kv_caches, _new_masked_kv_caches], dim=1)

            x = x.split(n_ctx, dim=1)[0]
            cross_qks = cross_qks.split(n_ctx, dim=1)[0]
            logits = (
                x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
            ).float()
        else: # decoder1
            qk_mask = torch.cat([torch.zeros((1,text_offset)),
                                 torch.ones((1, 448-text_offset)) * -np.inf,
                                 torch.FloatTensor([[0]])],
                                 dim=1)
            # nn.Linear speedup trick
            if x.shape[0] == 1 and x.shape[1] == 1:
                qk_mask = torch.cat([qk_mask, torch.FloatTensor([[1]]) * -np.inf], dim=1)

            logits, new_masked_kv_caches = self.forwardBlocks(x,
                                                              qk_mask,
                                                              masked_kv_caches,
                                                              self.cross_k_caches,
                                                              self.cross_v_caches,
                                                              text_offset,
                                                              )
            cross_qks = None

        return logits, cross_qks, new_masked_kv_caches

    def forwardBlocks(self,
                      x: Tensor,
                      qk_mask: Optional[Tensor] = None,
                      masked_kv_caches: Optional[Tensor] = None,
                      cross_k_caches: Optional[Tensor] = None,
                      cross_v_caches: Optional[Tensor] = None,
                      text_offset: Optional[int] = None, # for objc part of coremlDecoder1
                      beam_idx: Optional[int] = None,    # for objc part of coremlDecoder256
                      ):

        # Coreml Decoder part
        if self.use_coreml:
            if masked_kv_caches is not None and x.shape[1] == 1:
                self.coreml.loadDecoder1()
                return self.coreml.decoder1Predict(x, qk_mask, text_offset)
            else:
                self.coreml.n_alignment_head = self.alignment_heads.to_sparse().indices().shape[1]
                self.coreml.loadDecoder256()
                return self.coreml.decoder256Predict(x, qk_mask, beam_idx)

        if x.shape[0] == 1 and x.shape[1] == 1:
            # nn.Linear speed up trick
            # mlp([1,1,768]) is 25% slower than mlp([1, 2~100, 768]) on ANE
            # I don't know why... note: this also makes whisper on cpu 10.5s -> 14.3s
            x = torch.cat([x, torch.zeros((1, 1, self.n_state))], dim=1)

        cross_head_weights = []
        new_masked_kv_caches = []

        mkv_caches = twoLevelSplit(masked_kv_caches, 8)
        ck_caches = twoLevelSplit(cross_k_caches, 4)
        cv_caches = twoLevelSplit(cross_v_caches, 4)

        for layer_idx, block in enumerate(self.blocks):
            mk = mv = ck = cv = None
            if masked_kv_caches is not None:
                mk = mkv_caches[layer_idx*2].squeeze(0)
                mv = mkv_caches[layer_idx*2 + 1].squeeze(0)

            if cross_k_caches is not None:
                ck = ck_caches[layer_idx]
                cv = cv_caches[layer_idx]

            x, cross_qk, new_mk, new_mv= block(x, qk_mask, mk, mv, ck, cv)

            for head_idx in range(self.n_head):
                if self.alignment_heads[layer_idx][head_idx]:
                    cross_head_weights.append(cross_qk[0][head_idx])

            new_masked_kv_caches.append(new_mk)
            new_masked_kv_caches.append(new_mv)

        cross_head_weights = torch.stack(cross_head_weights)
        new_masked_kv_caches = torch.stack(new_masked_kv_caches)

        x = self.ln(x)

        if qk_mask.shape[0] == 1: # decoder1
            splits = self.token_embedding.weight.split(12288, dim=0)
            logits = torch.cat([x @ split.transpose(0,1) for split in splits], dim=2)

            if x.shape[0] == 1:
                # end Linear speed up trick
                logits = logits.split(1, dim=1)[0]
                new_masked_kv_caches = new_masked_kv_caches.split(1, dim=2)[0]

            return logits, new_masked_kv_caches
        else: # decoder256
            return x, cross_head_weights, new_masked_kv_caches

