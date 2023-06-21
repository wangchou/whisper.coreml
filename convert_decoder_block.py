import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import os
import numpy as np

# model setting
modelSize = "tiny"
model = whisper.load_model(modelSize).cpu()
n_state = 384 # tiny=384, base=512, small=768, medium=1024, large=1280

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
qk_mask = torch.ones((1, 1), dtype=dtype1)
mk = torch.ones((bs, 1, n_state), dtype=dtype1)
mv = torch.ones((bs, 1, n_state), dtype=dtype1)
ck = torch.ones((bs, 1500, n_state), dtype=dtype1)
cv = torch.ones((bs, 1500, n_state), dtype=dtype1)

traced_decoder_block = torch.jit.trace(decoder.blocks[0], (x, xa, qk_mask, mk, mv, ck, cv))

# input types for convert
range0to448 = ct.RangeDim(lower_bound=0, upper_bound=448, default=1)
input1 = ct.TensorType("x", ct.Shape((bs, range0to448, n_state)), dtype=dtype2)
input2 = ct.TensorType("xa", xa.shape, dtype=dtype2)
input3 = ct.TensorType("qk_mask", qk_mask.shape, dtype=dtype2)
input4 = ct.TensorType("mk", ct.Shape((bs, range0to448, n_state)), dtype=dtype2)
input5 = ct.TensorType("mv", ct.Shape((bs, range0to448, n_state)), dtype=dtype2)
input6 = ct.TensorType("ck", ck.shape, dtype=dtype2)
input7 = ct.TensorType("cv", cv.shape, dtype=dtype2)
inputs = [input1, input2, input3, input4, input5, input6, input7]

outputs = [ct.TensorType("x_output"),
           ct.TensorType("cross_qk"),
           ct.TensorType("new_mk"),
           ct.TensorType("new_mv"),
           ct.TensorType("new_ck"),
           ct.TensorType("new_cv"),
           ]

decoder_block = ct.convert(
    traced_decoder_block,
    convert_to="mlprogram",
    inputs=inputs,
    outputs=outputs,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16, # make fp16 input and output available
)

folder_path = f"coreml/{modelSize}"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
decoder_block.save(f"{folder_path}/DecoderBlock.mlpackage")
#decoder_block_fp16 = quantization_utils.quantize_weights(decoder_block, nbits=16)
#decoder_block_fp16.save(f"{folder_path}/DecoderBlock.mlmodel")

# test accuracy
torch_output = traced_decoder_block.forward(x, xa, qk_mask, mk, mv, ck, cv)[0]
print("torch model output:", torch_output[0][0][:5])
coreml_output = torch.from_numpy(
        decoder_block.predict({'x': x,
                               'xa': xa,
                               'qk_mask': qk_mask,
                               'mk': mk,
                               'mv': mv,
                               'ck': ck,
                               'cv': cv})['x_output']
)
print(f"coreml {modelSize} model output:", coreml_output[0][0][:5])
diff = torch.abs(torch_output - coreml_output).detach()
print("diff avg,max:", torch.mean(diff), torch.max(diff))

