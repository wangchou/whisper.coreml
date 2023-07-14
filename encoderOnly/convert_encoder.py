#### from encoder.py
from whisper.model import Whisper, ModelDimensions
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

class AudioEncoderCoreml(nn.Module):
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
        self.ln_post = nn.LayerNorm(n_state, eps=1e-7)
        self.coremlEncoder = None
        self.n_state = n_state
        self.n_layer = n_layer
        self.from_block_idx = 0

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        self.from_block_idx = 0
        for i in range(0, self.n_layer, 4):
           self.from_block_idx = i
           x = self.block4(x)

        return x

    # divided sub-model for speed up ANECompilerService
    def block4(self, x: Tensor):
        if self.from_block_idx == 0:
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            x = (x + self.positional_embedding)

        for i in range(self.from_block_idx, min(self.from_block_idx + 4, self.n_layer)):
            x = self.blocks[i](x)

        if self.from_block_idx + 4 >= self.n_layer:
            x = self.ln_post(x)

        return x

class WhisperCoreml(Whisper):
    def __init__(self, dims: ModelDimensions):
        super().__init__(dims)

        setattr(self, 'encoder', AudioEncoderCoreml(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        ))
###################

import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
from timeit import default_timer as timer
import numpy as np
import os
import sys

print("-------------")
print("🐳 Encoder 🐳")
print("-------------")

modelName = sys.argv[1] if len(sys.argv) > 1 else "small"
model = whisper.load_model(modelName).cpu()
modelSize = modelName.split(".")[0]
n_state = { 'tiny': 384, 'base': 512, 'small': 768, 'medium': 1024, 'large': 1280}[modelSize]
n_layer = { 'tiny': 4, 'base': 6, 'small': 12, 'medium': 24, 'large': 32}[modelSize]

whisperCoreml = WhisperCoreml(model.dims).eval()
whisperCoreml.load_state_dict(model.state_dict())
encoder = whisperCoreml.encoder
encoder.eval()

total_conversion_time = 0
total_prediction_time = 0
skip_model_load = True
def convertBlock4(encoder, from_block_idx, skip_model_load: bool):
    global total_conversion_time
    global total_prediction_time
    print(f"- {modelName} encoder Block {from_block_idx}..<{min(from_block_idx+4, n_layer)} -")

    #
    # Torch Trace
    #
    if from_block_idx == 0:
        x = torch.ones((1, 80, 3000))
    else:
        x = torch.ones((1, 1500, n_state))

    encoder.from_block_idx = from_block_idx
    traced_encoder = torch.jit.trace_module(encoder,
                                            {'block4': (x)})

    # ct.convert only look forward func
    traced_encoder.forward = traced_encoder.block4

    #
    # coremltools convert
    #
    pipeline = ct.PassPipeline.CLEANUP
    pipeline.insert_pass(-1, "common::add_fp16_cast") # fp16 for ane
    #pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": "layer_norm,softmax"})
    pipeline.remove_passes({
        # fix complex graph caused by speedup_conversion_workaround
        "common::const_deduplication",
    })

    startT = timer()
    encoder = ct.convert(
        traced_encoder,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="out_x", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16, # make fp16 input and output available
        pass_pipeline=pipeline,
        skip_model_load=skip_model_load,
    )

    conversion_time = timer() - startT
    total_conversion_time += conversion_time
    print(" ")
    print(f"conversion time: {conversion_time:.3f}s")


    folder_path = f"coreml/{modelName}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    encoder.save(f"{folder_path}/CoremlEncoder{from_block_idx}.mlpackage")

    if not skip_model_load:
        #
        # prediction
        #
        torch_output = traced_encoder.forward(x)
        #print("torch model output:", torch_output)

        x = x.cpu().detach().numpy()

        durations = []
        for i in range(5):
            startT = timer()
            coreml_output = torch.from_numpy(
                list(encoder.predict({'x': x}).values())[0]
            )
            durations.append(timer() - startT)
        prediction_time = np.median(durations)
        total_prediction_time += prediction_time
        print(f"prediction time: {prediction_time:.3f}s")

        #print(f"coreml {modelName} block{i} model output:", coreml_output)
        diff = torch.abs(torch_output - coreml_output).detach()
        print("diff avg,max:", torch.mean(diff), torch.max(diff))

skip_model_load = True
for block_idx in range(0, n_layer, 4):
    convertBlock4(encoder, block_idx, skip_model_load)

print("---------------------")
print(f"{modelName} encoder total conversion time: {total_conversion_time:.3f}s")
if not skip_model_load:
    print(f"{modelName} encoder total prediction_time time: {total_prediction_time:.3f}s")
print("")
# note
# conversion time on Macbook M1 Air 16GB
# tiny:        7s
# small:      36s (coremltools: 0s + ANECompilerService: 36s), predict: 115ms
# medium:    101s (12s + 89s), 344ms
# large:     178s (24s + 154s, use 6GB memory), 628ms
