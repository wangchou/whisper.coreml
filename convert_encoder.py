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

encoder = model.encoder
encoder.eval()

total_conversion_time = 0
total_prediction_time = 0
skip_model_load = True
def convertBlock12(encoder, from_block_idx, skip_model_load: bool):
    global total_conversion_time
    global total_prediction_time
    print(f"- {modelName} encoder Block {from_block_idx}..<{min(from_block_idx+12, n_layer)} -")

    #
    # Torch Trace
    #
    if from_block_idx == 0:
        x = torch.ones((1, 80, 3000))
    else:
        x = torch.ones((1, 1500, n_state))

    encoder.from_block_idx = from_block_idx
    traced_encoder = torch.jit.trace_module(encoder,
                                            {'block12': (x)})

    # ct.convert only look forward func
    traced_encoder.forward = traced_encoder.block12

    #
    # coremltools convert
    #
    pipeline = ct.PassPipeline.CLEANUP
    pipeline.insert_pass(-1, "common::add_fp16_cast") # fp16 for ane
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
    print(f"conversion time: {conversion_time:.3f}s")
    print(" ")

    folder_path = f"coreml/{modelName}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    encoder.save(f"{folder_path}/Encoder{from_block_idx}.mlpackage")

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
for block_idx in range(0, n_layer, 12):
    convertBlock12(encoder, block_idx, skip_model_load)

print("---------------------")
print(f"{modelName} encoder total conversion time: {total_conversion_time:.3f}s")
if not skip_model_load:
    print(f"{modelName} encoder total prediction_time time: {total_prediction_time:.3f}s")
print("")
