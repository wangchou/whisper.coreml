import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
from timeit import default_timer as timer
import numpy as np
import os

modelSize = "base"
model = whisper.load_model(modelSize).cpu()
n_state = { 'tiny': 384, 'base': 512, 'small': 768, 'medium': 1024, 'large': 1280}[modelSize]
n_layer = { 'tiny': 4, 'base': 6, 'small': 12, 'medium': 24, 'large': 32}[modelSize]

encoder = model.encoder
encoder.eval()

total_conversion_time = 0
total_prediction_time = 0
def convertBlock4(encoder, from_block_idx):
    global total_conversion_time
    global total_prediction_time
    print("")
    print(f"- {modelSize} model Block {from_block_idx}..<{min(from_block_idx+4, n_layer)} -")

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
    )

    conversion_time = timer() - startT
    total_conversion_time += conversion_time
    print(" ")
    print(f"conversion time: {conversion_time:.3f}s")


    folder_path = f"coreml/{modelSize}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    encoder.save(f"{folder_path}/CoremlEncoder{from_block_idx}.mlpackage")

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

    #print(f"coreml {modelSize} block{i} model output:", coreml_output)
    diff = torch.abs(torch_output - coreml_output).detach()
    print("diff avg,max:", torch.mean(diff), torch.max(diff))

for block_idx in range(0, n_layer, 4):
    convertBlock4(encoder, block_idx)

print("---------------------")
print(f"{modelSize} total conversion time: {total_conversion_time:.3f}s")
print(f"{modelSize} total prediction_time time: {total_prediction_time:.3f}s")

# note
# conversion time on Macbook M1 Air 16GB
# tiny:        7s
# small:      36s (coremltools: 0s + ANECompilerService: 36s), predict: 115ms
# medium:    101s (12s + 89s), 344ms
# large:     178s (24s + 154s, use 6GB memory), 628ms
