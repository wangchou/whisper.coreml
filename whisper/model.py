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

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
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
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)

            new_k = k
            new_v = v
            # only for self masked attention
            if qk_mask is not None and cache_k is not None:
                k = torch.cat([cache_k, k], dim=1)
                v = torch.cat([cache_v, v], dim=1)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
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

        qk = q.float() @ k.float() * self.qk_scale
        if qk_mask is not None:
            qk = qk + qk_mask
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w.float() @ v.float()).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

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
        cross_qk = None
        if self.cross_attn:
            x_out, cross_qk, new_ck, new_cv = self.cross_attn(self.cross_attn_ln(x), xa, cache_k=ck, cache_v=cv)
            x = x + x_out
        x = x + self.mlp(self.mlp_ln(x))
        return x, cross_qk, new_mk, new_mv, new_ck, new_cv

################################################
# Coreml Encoder part
from ctypes import cdll, c_int, c_float, c_char_p, c_void_p, POINTER
import ctypes
import torch

class CoremlEncoder():
    def __init__(self, n_state: int):
        self.n_state = n_state
        self.encoderObj = None
        self.mlmodel_handle = None

    def loadModel(self):
        if self.mlmodel_handle == None:
            modelSize = self.getModelSize(self.n_state)
            self.encoderObj = cdll.LoadLibrary(f'./coreml/{modelSize}/encoderWrapper.so')
            self.encoderObj.loadModel.argtypes = [c_char_p]
            self.encoderObj.loadModel.restype = c_void_p
            c_string = bytes(f'./coreml/{modelSize}/CoremlEncoder.mlmodelc', 'ascii')
            self.mlmodel_handle = self.encoderObj.loadModel(c_string)

    def predictWith(self, melSegment):
        if self.mlmodel_handle == None:
            self.loadModel()
        self.encoderObj.predictWith.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]
        self.encoderObj.predictWith.restypes = None

        # force memory continuous, this is very important
        melSegment = melSegment.contiguous()
        melSegmentDataPtr = ctypes.cast(melSegment.data_ptr(), POINTER(c_float))

        # alloc output buffer
        output_floats = torch.ones((1, 1500, self.n_state), dtype=torch.float32).contiguous()
        output_floats_ptr = ctypes.cast(output_floats.data_ptr(), POINTER(c_float))
        self.encoderObj.predictWith(self.mlmodel_handle, melSegmentDataPtr, output_floats_ptr)
        return output_floats

    def closeModel(self):
        if self.mlmodel_handle != None:
            self.encoderObj.closeModel.argtypes = [c_void_p]
            self.encoderObj.closeModel.restypes = None
            self.encoderObj.closeModel(self.mlmodel_handle)

    def getModelSize(self, n_state: int):
        if n_state == 384:
            return "tiny"
        elif n_state == 512:
            return "base"
        elif n_state == 768:
            return "small"
        elif n_state == 1024:
            return "medium"
        elif n_state == 1280:
            return "large"
        else:
            return "unknown_model_size"
########################################
class CoremlDecoder():
    def __init__(self, n_layer: int, n_state: int):
        self.n_layer = n_layer
        self.n_state = n_state
        self.decoderObj = None
        self.mlmodel_handle = None

    def loadModel(self):
        if self.mlmodel_handle == None:
            modelSize = self.getModelSize(self.n_state)
            self.decoderObj = cdll.LoadLibrary(f'./coreml/{modelSize}/decoderWrapper.so')
            self.decoderObj.loadModel.argtypes = [c_char_p, c_int, c_int]
            self.decoderObj.loadModel.restype = c_void_p
            self.decoderObj.loadModel.restype = c_void_p
            c_string = bytes(f'./coreml/{modelSize}/CoremlDecoder.mlmodelc', 'ascii')
            self.mlmodel_handle = self.decoderObj.loadModel(c_string, self.n_layer, self.n_state)

            bs = 5 # beam_size
            n_head = 6 # tiny=6, base=8, small=12, medium=16, large=20
            n_state = self.n_state
            n_layer = self.n_layer
            dtype1=torch.float32
            # prepare output buffers
            self.out_x = torch.ones((bs, 1, n_state), dtype=dtype1).contiguous()
            self.out_cross_qks = torch.ones((n_layer * bs, n_head, 1, 1500), dtype=dtype1).contiguous()
            self.new_masked_kv_caches = torch.ones((n_layer * 2, bs, 1, n_state), dtype=dtype1).contiguous()
            self.new_cross_kv_caches = torch.ones(1, dtype=dtype1).contiguous() # this is dummy output
            self.outXPtr = ctypes.cast(self.out_x.data_ptr(), POINTER(c_float))
            self.outCQKPtr = ctypes.cast(self.out_cross_qks.data_ptr(), POINTER(c_float))
            self.outMKVPtr = ctypes.cast(self.new_masked_kv_caches.data_ptr(), POINTER(c_float))
            self.outCKVPtr = ctypes.cast(self.new_cross_kv_caches.data_ptr(), POINTER(c_float))

    def predictWith(self, x, xa, masked_kv_caches, cross_kv_caches):
        if self.mlmodel_handle == None:
            self.loadModel()
        self.decoderObj.predictWith.argtypes = [c_void_p,
                                                POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
                                                c_int, c_int, c_int,
                                                POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
        self.decoderObj.predictWith.restypes = None

        # prepare inputs
        x = x.contiguous()
        xPtr = ctypes.cast(x.data_ptr(), POINTER(c_float))
        xa = xa.contiguous()
        xaPtr = ctypes.cast(xa.data_ptr(), POINTER(c_float))
        masked_kv_caches = masked_kv_caches.contiguous()
        mkvPtr = ctypes.cast(masked_kv_caches.data_ptr(), POINTER(c_float))
        cross_kv_caches = cross_kv_caches.contiguous()
        ckvPtr = ctypes.cast(cross_kv_caches.data_ptr(), POINTER(c_float))

        # predict
        text_offset = masked_kv_caches.shape[2] if masked_kv_caches is not None else 0
        startT = timer()
        self.decoderObj.predictWith(self.mlmodel_handle,
                                    xPtr, xaPtr, mkvPtr, ckvPtr,
                                    self.n_layer, self.n_state, text_offset,
                                    self.outXPtr, self.outCQKPtr, self.outMKVPtr, self.outCKVPtr)
        print(f"\tpredictWit took {timer() - startT:.3f}")

        return self.out_x, self.out_cross_qks, self.new_masked_kv_caches, self.new_cross_kv_caches

    def closeModel(self):
        if self.mlmodel_handle != None:
            self.decoderObj.closeModel.argtypes = [c_void_p]
            self.decoderObj.closeModel.restypes = None
            self.decoderObj.closeModel(self.mlmodel_handle)

    def getModelSize(self, n_state: int):
        if n_state == 384:
            return "tiny"
        elif n_state == 512:
            return "base"
        elif n_state == 768:
            return "small"
        elif n_state == 1024:
            return "medium"
        elif n_state == 1280:
            return "large"
        else:
            return "unknown_model_size"
########################################

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.coremlEncoder = None
        self.n_state = n_state

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        ############################
        # Coreml Encoder part
        #encoder_startT = timer()
        if self.coremlEncoder == None:
            self.coremlEncoder = CoremlEncoder(self.n_state)
        x = self.coremlEncoder.predictWith(x)
        #print(f"\tendcoder took {timer() - encoder_startT}")
        return x
        ###########################3

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

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
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.coremlDecoder = None

    def forward(self, x: Tensor, xa: Tensor,
                text_offset: Tensor,
                masked_kv_caches: Optional[Tensor] = None,
                cross_kv_caches: Optional[Tensor] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = text_offset#masked_kv_caches.shape[2] if masked_kv_caches is not None else 0
        n_batch, n_ctx = x.shape
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + n_ctx]
        x = x.to(xa.dtype)

        n_ctx = x.shape[1]
        # general slice is not support in ane => generate it
        #qk_mask = self.mask[:n_ctx, :n_ctx]
        if text_offset == 0:
            qk_mask = (torch.ones(n_ctx, n_ctx) * -np.inf).triu_(1)
            qk_mask = torch.cat([torch.ones(n_ctx, 448) * -np.inf, qk_mask], dim=1)
        else:
            qk_mask = torch.cat([torch.zeros((1,text_offset)),
                                 torch.ones((1, 448-text_offset)) * -np.inf,
                                 torch.FloatTensor([[0]])],
                                dim=1)

        x, cross_qks, new_masked_kv_caches, new_cross_kv_caches = self.forwardBlocks(x, xa, qk_mask, masked_kv_caches, cross_kv_caches)

        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits, cross_qks, new_masked_kv_caches, new_cross_kv_caches

    def forwardBlocks(self, x: Tensor, xa: Tensor,
                      qk_mask: Optional[Tensor] = None,
                      masked_kv_caches: Optional[Tensor] = None,
                      cross_kv_caches: Optional[Tensor] = None):

        ############################
        # Coreml Decoder part
        #if masked_kv_caches is not None and x.shape[1] == 1:
        #    #startT = timer()
        #    if self.coremlDecoder == None:
        #        self.coremlDecoder = CoremlDecoder(self.n_layer, self.n_state)
        #    x, cross_qks, new_masked_kv_caches, new_cross_kv_caches = self.coremlDecoder.predictWith(x, xa, masked_kv_caches, cross_kv_caches)
        #    #print(f"\tcoreml decoder took {timer() - startT:.3f}")
        #    return x, cross_qks, new_masked_kv_caches, new_cross_kv_caches
        ###########################3

        cross_qks = []
        new_masked_kv_caches = []
        new_cross_kv_caches = []
        for layer_idx, block in enumerate(self.blocks):
            # mk = masked_key_cache, ck=cross_key_cache
            if masked_kv_caches is not None:
                mk = masked_kv_caches[layer_idx*2]
                mv = masked_kv_caches[layer_idx*2 + 1]
            else:
                mk = None
                mv = None
            if cross_kv_caches is not None:
                ck = cross_kv_caches[layer_idx*2]
                cv = cross_kv_caches[layer_idx*2 + 1]
            else:
                ck = None
                cv = None

            x, cross_qk, new_mk, new_mv, new_ck, new_cv = block(x, xa, qk_mask, mk, mv, ck, cv)

            cross_qks.append(cross_qk)
            new_masked_kv_caches.append(new_mk)
            new_masked_kv_caches.append(new_mv)
            new_cross_kv_caches.append(new_ck)
            new_cross_kv_caches.append(new_cv)

        cross_qks = torch.cat(cross_qks)
        new_masked_kv_caches = torch.stack(new_masked_kv_caches)
        if cross_kv_caches is None:
            new_cross_kv_caches = torch.stack(new_cross_kv_caches)
        else:
            # this speed up coreml a lot by avoiding big matrix of [8, 5, 1500, 384]
            # coreml do not support return None?
            # we probably will apply similar approach for only returns new part of new_masked_kv_cache
            new_cross_kv_caches = torch.zeros(1)

        x = self.ln(x)

        # print(f"x {x.shape}, cross_qks {cross_qks.shape}, new_masked_kv_caches {new_masked_kv_caches.shape}, new_cross_kv_caches {new_cross_kv_caches.shape}")
        return x, cross_qks, new_masked_kv_caches, new_cross_kv_caches

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)
        self.n_layer = dims.n_text_layer
        self.n_state = dims.n_text_state

        bs = 5
        self.text_offset = 0
        self.masked_kv_caches = None#torch.zeros((2*self.n_layer, 5, 448, self.n_state))
        self.cross_kv_caches = None

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

# unuse funcs called??
#    def embed_audio(self, mel: torch.Tensor):
#        return self.encoder(mel)
#
#    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
#        output, cross_qks, new_masked_kv_caches, new_cross_kv_caches = self.decoder(tokens, self.encoder(mel),
#                                                                                    self.masked_kv_caches,
#                                                                                    self.cross_kv_caches,
#                                                                                    )
#        self.masked_kv_caches = new_masked_kv_caches
#        self.cross_kv_caches = new_cross_kv_caches
#        return output, cross_qks

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if self.text_offset == 0:
            self.masked_kv_caches = torch.zeros((2*self.n_layer, 1, 448, self.n_state))
        output, cross_qks, new_masked_kv_caches, new_cross_kv_caches = self.decoder(tokens,
                                                                                    self.encoder(mel),
                                                                                    self.text_offset,
                                                                                    self.masked_kv_caches,
                                                                                    self.cross_kv_caches)
        # this only called once in each add_word_timestamps,
        # => no next call
        # => no need for update self.xxx_cache
        return output, cross_qks

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
