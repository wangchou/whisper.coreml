import whisper
import torch
import coremltools as ct
import os
import sys
import numpy as np
from timeit import default_timer as timer

print("--------------------------")
print("🐳 mel->cross_kv_caches 🐳")
print("--------------------------")

# model setting
modelName = sys.argv[1] if len(sys.argv) > 1 else "small"
model = whisper.load_model(modelName).cpu()
modelSize = modelName.split(".")[0]
n_state = { 'tiny': 384, 'base': 512, 'small': 768, 'medium': 1024, 'large': 1280}[modelSize]
n_layer = { 'tiny': 4, 'base': 6, 'small': 12, 'medium': 24, 'large': 32}[modelSize]

decoder = model.decoder
decoder.eval()

inType=np.float16
# coreml has some issue when output type = fp16 when using ane or gpu
# https://github.com/apple/coremltools/issues/1893
outType=np.float16

bs = 1 # beam_size

# max token len for first time = max_prefix_len(224) + sot_len(3)
max_n_ctx = decoder.max_n_ctx_for_1st
xa = torch.ones((1, 1500, n_state))

traced_decoder = torch.jit.trace_module(decoder,
                                        {'crossKVCaches': {"xa":xa}
                                        },
                                        example_inputs_is_kwarg=True)
# ct.convert only look forward func
traced_decoder.forward = traced_decoder.crossKVCaches

# input types for convert
input1 = ct.TensorType("xa", xa.shape, dtype=inType)
inputs = [input1]

outputs = [ct.TensorType("out_cross_k_caches", dtype=outType),
           ct.TensorType("out_cross_v_caches", dtype=outType),]

startT = timer()
decoder = ct.convert(
    traced_decoder,
    convert_to="mlprogram",
    inputs=inputs,
    outputs=outputs,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS16, # make fp16 input and output available
    skip_model_load=True,
)
print(f"{modelName} crossKVCaches conversion time: {timer()-startT:.3f}s")

folder_path = f"coreml/{modelName}"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
decoder.save(f"{folder_path}/CoremlCrossKV.mlpackage")

## test accuracy
#torch_output = traced_decoder.forward(x, xa, qk_mask)[0]
#print("torch model output:", torch_output[:,0,:2], torch_output[:,max_n_ctx-1,n_state-1])
#print(f"torch predicted {timer()-startT:.3f}")

# coremltools has some issue on np.float16 output
# it will took so long to convert fp16 to fp32 for python
#coreml_output = torch.from_numpy(
#        decoder.predict({'x': x,
#                         'xa': xa,
#                         'qk_mask': qk_mask})['out_x']
#)

#print(f"coreml predicted {timer()-startT:.3f}")
#print(f"coreml {modelName} model output:", coreml_output[:,0,:2], coreml_output[:,max_n_ctx-1,n_state-1])
#diff = torch.abs(torch_output - coreml_output).detach()
#print("diff avg,max:", torch.mean(diff), torch.max(diff))
