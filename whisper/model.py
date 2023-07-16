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
from .coreml import CoremlEncoder, CoremlDecoder, CoremlDecoder256
from .encoder import AudioEncoder
from .decoder import TextDecoder
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

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, use_coreml: bool, modelName):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            use_coreml,
            modelName,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            use_coreml,
            modelName,
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
        self.isNewCKV = True

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
        #startT = timer()
        if self.text_offset == 0:
            self.masked_kv_caches = None
        output, cross_qks, new_masked_kv_caches = self.decoder(tokens,
                                                               self.encoder(mel),
                                                               self.text_offset,
                                                               self.isNewCKV,
                                                               self.masked_kv_caches)
        # this only called once in each add_word_timestamps,
        # => no next call
        # => no need for update self.xxx_cache
        #print(f"Model.forward tooks {timer()-startT:.4f}")
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
