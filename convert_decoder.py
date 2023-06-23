import whisper
import torch
import coremltools as ct
import os
import numpy as np

# model setting
modelSize = "tiny"
model = whisper.load_model(modelSize).cpu()
n_state = { 'tiny': 384, 'base': 512, 'small': 768, 'medium': 1024, 'large': 1280}[modelSize]
n_layer = { 'tiny': 4, 'base': 6, 'small': 12, 'medium': 24, 'large': 32}[modelSize]

decoder = model.decoder
decoder.eval()

dtype1=torch.float16
dtype2=np.float16

bs = 5 # beam_size
text_offset = 10

# input data for trace
x = torch.ones((bs, 1, n_state), dtype=dtype1)
xa = torch.ones((bs, 1500, n_state), dtype=dtype1)
masked_kv_caches = torch.ones((n_layer * 2, bs, text_offset, n_state), dtype=dtype1)
cross_kv_caches = torch.ones((n_layer * 2, bs, 1500, n_state), dtype=dtype1)

traced_decoder = torch.jit.trace_module(decoder,
                                        {'forwardBlocks': (x, xa, masked_kv_caches, cross_kv_caches)})
# ct.convert only look forward func
traced_decoder.forward = traced_decoder.forwardBlocks

# input types for convert
range1to448 = ct.RangeDim(lower_bound=1, upper_bound=448, default=10)
#enumeratedShapes = ct.EnumeratedShapes(shapes=[[8, 5, 1, 384],
#                                               [8, 5, 10, 384]],
#                                       default=[8, 5, 10, 384])
input1 = ct.TensorType("x", x.shape, dtype=dtype2)
input2 = ct.TensorType("xa", xa.shape, dtype=dtype2)
#input3 = ct.TensorType("masked_kv_caches", enumeratedShapes, dtype=dtype2)
input3 = ct.TensorType("masked_kv_caches", ct.Shape((n_layer*2, bs, range1to448, n_state)), dtype=dtype2)
input4 = ct.TensorType("cross_kv_caches", cross_kv_caches.shape, dtype=dtype2)
inputs = [input1, input2, input3, input4]

outputs = [ct.TensorType("out_x", dtype=dtype2),
           ct.TensorType("out_cross_qks", dtype=dtype2),
           ct.TensorType("out_new_masked_kv_caches", dtype=dtype2),
           ct.TensorType("out_new_cross_kv_caches", dtype=dtype2)]

decoder = ct.convert(
    traced_decoder,
    convert_to="mlprogram",
    inputs=inputs,
    outputs=outputs,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16, # make fp16 input and output available
)

folder_path = f"coreml/{modelSize}"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
decoder.save(f"{folder_path}/CoremlDecoder.mlpackage")

## test accuracy
torch_output = traced_decoder.forward(x, xa,masked_kv_caches, cross_kv_caches)[0]
print("torch model output:", torch_output[0][0][:2], torch_output[bs-1][0][n_state-1] )

coreml_output = torch.from_numpy(
        decoder.predict({'x': x,
                         'xa': xa,
                         'masked_kv_caches': masked_kv_caches,
                         'cross_kv_caches':cross_kv_caches})['out_x']
)
print(f"coreml {modelSize} model output:", coreml_output[0][0][:2], coreml_output[bs-1][0][n_state-1])
diff = torch.abs(torch_output - coreml_output).detach()
print("diff avg,max:", torch.mean(diff), torch.max(diff))

