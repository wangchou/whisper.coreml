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

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        text_offset: Optional[int] = None,
        layer_kv_cache: Optional[Tensor] = None,
    ):
        q = self.query(x)

        if text_offset == 0 or xa is None:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)

            # update layer_kv_cache by k&v
            is_cross_attn = k.shape[1] > 448 #n_text_ctx = 448
            # original code in hook
            #if module not in cache or is_cross_attn:
            #    # save as-is, for the first token or cross attention
            #    cache[module] = output
            #else:
            #    cache[module] = torch.cat([cache[module], output], dim=1).detach()
            # translated code
            if is_cross_attn:
                layer_kv_cache[0] = k.detach()
                layer_kv_cache[1] = v.detach()
            elif text_offset == 0:
                for b in range(0, k.shape[0]):
                    layer_kv_cache[0][b][:k.shape[1]] = k[b]
                    layer_kv_cache[1][b][:k.shape[1]] = v[b]
            elif text_offset != None:
                #k torch.Size([5, 3, 384])
                #layer_kv_cache torch.Size([2, 5, 448, 384])
                #range(0, 3)
                to_text_offset = text_offset+k.shape[1]
                cache_k = layer_kv_cache[0, :, :text_offset]
                cache_v = layer_kv_cache[1, :, :text_offset]
                k = torch.cat([cache_k, k], dim=1)
                v = torch.cat([cache_v, v], dim=1)
                for b in range(0, k.shape[0]):
                    layer_kv_cache[0][b][:to_text_offset] = k[b]
                    layer_kv_cache[1][b][:to_text_offset] = v[b]
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = layer_kv_cache[0]
            v = layer_kv_cache[1]

        #if layer_kv_cache is not None:
        #    print(f"k.shape ={k.shape}, v.shape={v.shape}, layer_kv_cache.shape={layer_kv_cache.shape}")
        # masked attn
        # k = v = (batch_size, 0~448, n_state=384) n_text_ctx = 448
        #
        # cross attn
        # k = v = (batch_size, 1500, n_state)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk.detach(), layer_kv_cache

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

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
        text_offset: int,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        masked_kv_cache: Optional[Tensor] = None,
        cross_kv_cache: Optional[Tensor] = None,
    ):
        x_out, masked_qk, new_masked_kv_cache = self.attn(self.attn_ln(x), mask=mask, text_offset=text_offset, layer_kv_cache=masked_kv_cache)
        x = x + x_out
        cross_qk = None
        if self.cross_attn:
            x_out, cross_qk, new_cross_kv_cache = self.cross_attn(self.cross_attn_ln(x), xa, text_offset=text_offset, layer_kv_cache=cross_kv_cache)
            x = x + x_out
        x = x + self.mlp(self.mlp_ln(x))
        return x, cross_qk, new_masked_kv_cache, new_cross_kv_cache

################################################
# Coreml Encoder part
from ctypes import cdll, c_float, c_char_p, c_void_p, POINTER
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
            self.encoderObj = cdll.LoadLibrary(f'./coreml/{modelSize}/objcWrapper.so')
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
        if self.coremlEncoder == None:
            self.coremlEncoder = CoremlEncoder(self.n_state)
        x = self.coremlEncoder.predictWith(x)
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

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)


    def forward(self, x: Tensor, xa: Tensor,
                text_offset, masked_kv_caches, cross_kv_caches):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = text_offset #next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        layer_idx = 0
        cross_qks = []
        new_masked_kv_caches = []
        new_cross_kv_caches = []
        for block in self.blocks:
            masked_kv_cache = masked_kv_caches[layer_idx]
            cross_kv_cache = cross_kv_caches[layer_idx]
            x, cross_qk, new_masked_kv_cache, new_cross_kv_cache = block(x, text_offset, xa, mask=self.mask, masked_kv_cache=masked_kv_cache.detach(), cross_kv_cache = cross_kv_cache.detach())

            cross_qks.append(cross_qk)
            new_masked_kv_caches.append(new_masked_kv_cache)
            new_cross_kv_caches.append(new_cross_kv_cache)

            layer_idx += 1

        cross_qks = torch.cat(cross_qks)
        new_masked_kv_caches = torch.stack(new_masked_kv_caches)
        new_cross_kv_caches = torch.stack(new_cross_kv_caches)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits, cross_qks, new_masked_kv_caches, new_cross_kv_caches

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

        ###########################
        # Make decoder passing kv_caches
        # layer_count, 2(k&v), batch_size, n_ctx = 448, n_state=384(tiny)
        n_layer = self.dims.n_text_layer
        n_ctx = self.dims.n_text_ctx
        n_state = self.dims.n_text_state
        beam_size = 5
        masked_kv_caches = torch.empty(n_layer, 2, beam_size, n_ctx, n_state)
        self.register_buffer("masked_kv_caches", masked_kv_caches, persistent=False)

        # layer_count, 2(k&v), batch_size, n_audio_ctx, n_state=384(tiny)
        cross_kv_caches = torch.empty(n_layer, 2, beam_size, 1500, n_state)
        self.register_buffer("cross_kv_caches", cross_kv_caches, persistent=False)

        self.text_offset = 0

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        output, cross_qks, new_masked_kv_caches, new_cross_kv_caches = self.decoder(tokens, self.encoder(mel),
                                                                                    self.text_offset,
                                                                                    self.masked_kv_caches,
                                                                                    self.cross_kv_caches,
                                                                                    )
        self.masked_kv_caches = new_masked_kv_caches
        self.cross_kv_caches = new_cross_kv_caches
#        self.text_offset += output.shape[1]
        return output, cross_qks

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        output, cross_qks, new_masked_kv_caches, new_cross_kv_caches = self.decoder(tokens, self.encoder(mel),
                                                                                    self.text_offset,
                                                                                    self.masked_kv_caches,
                                                                                    self.cross_kv_caches,
                                                                                    )
        self.masked_kv_caches = new_masked_kv_caches
        self.cross_kv_caches = new_cross_kv_caches
#        self.text_offset += output.shape[1]
        return output, cross_qks

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            is_cross_attn = output.shape[1] > self.dims.n_text_ctx
            if module not in cache or is_cross_attn:
                # save as-is, for the first token or cross attention
                cache[module] = output
#            else:
#                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
