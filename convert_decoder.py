import whisper
import torch
import coremltools as ct
import os
import sys
import numpy as np
from timeit import default_timer as timer

print("--------------")
print("🐳 Decoder1 🐳")
print("--------------")

# model setting
modelName = sys.argv[1] if len(sys.argv) > 1 else "small"
bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1 # beam search size

model = whisper.load_model(modelName).cpu()
modelSize = modelName.split(".")[0]
n_state = { 'tiny': 384, 'base': 512, 'small': 768, 'medium': 1024, 'large': 1280, 'turbo': 1280}[modelSize]
n_text_layer = { 'tiny': 4, 'base': 6, 'small': 12, 'medium': 24, 'large': 32, 'turbo': 4}[modelSize]
n_head = n_state//64

decoder = model.decoder
decoder.eval()

inType=np.float16
outType=np.float16

# input data for trace
x = torch.ones((bs, 1, n_state))
if bs == 1:
    qk_mask = torch.zeros((1,450))
else:
    qk_mask = torch.zeros((1,449))

masked_kv_caches = torch.ones((n_text_layer * 2, bs, 448, n_state))
cross_k_caches = torch.ones((n_text_layer, n_head, 64, 1500))
cross_v_caches = torch.ones((n_text_layer, n_head, 1500, 64))

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
    traced_decoder = torch.jit.trace_module(decoder,
                                        {'forwardBlocks': {"x":x,
                                                           "qk_mask": qk_mask,
                                                           "masked_kv_caches": masked_kv_caches,
                                                           "cross_k_caches": cross_k_caches,
                                                           "cross_v_caches": cross_v_caches,
                                                           }
                                        },
                                        example_inputs_is_kwarg=True)
# ct.convert only look forward func
traced_decoder.forward = traced_decoder.forwardBlocks

# input types for convert
input1 = ct.TensorType("x", x.shape, dtype=inType)
input2 = ct.TensorType("qk_mask", qk_mask.shape, dtype=inType)
input3 = ct.TensorType("masked_kv_caches", masked_kv_caches.shape, dtype=inType)
input4 = ct.TensorType("cross_k_caches", cross_k_caches.shape, dtype=inType)
input5 = ct.TensorType("cross_v_caches", cross_v_caches.shape, dtype=inType)
inputs = [input1, input2, input3, input4, input5]

outputs = [ct.TensorType("out_x", dtype=outType),
           ct.TensorType("out_new_masked_kv_caches", dtype=outType)]

startT = timer()
decoder = ct.convert(
    traced_decoder,
    convert_to="mlprogram",
    inputs=inputs,
    outputs=outputs,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16, # make fp16 input and output available
    #skip_model_load=True,
)
print(f"{modelName} bs={bs} decoder1 conversion time: {timer()-startT:.3f}s")

folder_path = f"coreml/{modelName}"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
decoder.save(f"{folder_path}/Decoder.mlpackage")
