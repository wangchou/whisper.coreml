import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import os
import numpy as np

# model setting
modelSize = "tiny"
model = whisper.load_model(modelSize).cpu()
n_state = { 'tiny': 384, 'base': 512, 'small': 768, 'medium': 1024, 'large': 1280}[modelSize]
n_layer = { 'tiny': 4, 'base': 6, 'small': 12, 'medium': 24, 'large': 32}[modelSize]
text_offset = 10

# trace model by torch.jit
decoder = model.decoder
decoder.eval()

# float32 -> float16 to avoid casting kv_cache in each call
dtype1=torch.float16
dtype2=np.float16

bs = 5 # beam_size

# input data for trace
x = torch.ones((bs, 1, n_state), dtype=dtype1)
xa = torch.ones((bs, 1500, n_state), dtype=dtype1)
masked_kv_caches = torch.ones((n_layer * 2, bs, text_offset, n_state), dtype=dtype1)
cross_kv_caches = torch.ones((n_layer * 2, bs, 1500, n_state), dtype=dtype1)

traced_decoder = torch.jit.trace_module(decoder, {'forwardBlocks': (x, xa, masked_kv_caches, cross_kv_caches)})
# ct.convert only look forward func
traced_decoder.forward = traced_decoder.forwardBlocks

# input types for convert
range0to448 = ct.RangeDim(lower_bound=1, upper_bound=448, default=1)
#input1 = ct.TensorType("x", ct.Shape((bs, range0to448, n_state)), dtype=dtype2)
input1 = ct.TensorType("x", x.shape, dtype=dtype2)
input2 = ct.TensorType("xa", xa.shape, dtype=dtype2)
input3 = ct.TensorType("masked_kv_caches", ct.Shape((n_layer*2, bs, range0to448, n_state)), dtype=dtype2)
input4 = ct.TensorType("cross_kv_caches", cross_kv_caches.shape, dtype=dtype2)
inputs = [input1, input2, input3, input4]

outputs = [ct.TensorType("logits"),
           ct.TensorType("cross_qk"),
           ct.TensorType("new_masked_kv_caches"),
           ct.TensorType("new_cross_kv_caches"),
           ]

decoder = ct.convert(
    traced_decoder,
    convert_to="mlprogram",
    inputs=inputs,
#    outputs=outputs,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16, # make fp16 input and output available
)

# set some input to optional
# https://github.com/apple/coremltools/issues/388
#spec = decoder.get_spec()
#spec.description.input[2].type.isOptional = True
#spec.description.input[3].type.isOptional = True
#decoder = ct.models.MLModel(spec, weights_dir=decoder.weights_dir)

folder_path = f"coreml/{modelSize}"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
decoder.save(f"{folder_path}/Decoder.mlpackage")
#decoder_block_fp16 = quantization_utils.quantize_weights(decoder_block, nbits=16)
#decoder_block_fp16.save(f"{folder_path}/DecoderBlock.mlmodel")

## test accuracy
#torch_output = traced_decoder.forward(x, xa, text_offset, qk_mask, masked_kv_caches, cross_kv_caches)[0]
#print("torch model output:", torch_output[0][0][:5])
#coreml_output = torch.from_numpy(
#        decoder_block.predict({'x': x,
#                               'text_offset': text_offset,
#                               'xa': xa,
#                               'qk_mask': qk_mask,
#                               'mk': mk,
#                               'mv': mv,
#                               'ck': ck,
#                               'cv': cv})['x_output']
#)
#print(f"coreml {modelSize} model output:", coreml_output[0][0][:5])
#diff = torch.abs(torch_output - coreml_output).detach()
#print("diff avg,max:", torch.mean(diff), torch.max(diff))
#
