import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import os
from timeit import default_timer as timer

# model setting
modelSize = "medium"
model = whisper.load_model(modelSize).cpu()

# trace model by torch.jit
encoder = model.encoder
encoder.eval()

melSegment = torch.ones((1, 80, 3000))
traced_encoder = torch.jit.trace(encoder, melSegment)

# convert to coreml model
startT = timer()
pipeline = ct.PassPipeline()
pipeline.remove_passes({
    # this resolve complex graph caused by speedup_conversion_workaround func
    # in deep network
    "common::const_deduplication",
})
encoder = ct.convert(
    traced_encoder,
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="melSegment", shape=melSegment.shape)],
    outputs=[ct.TensorType(name="output")],
    compute_units=ct.ComputeUnit.ALL,
    pass_pipeline=pipeline,
)
print("---")
print(f"{modelSize} coreml conversion took {timer()-startT:.3f}")
print("---")

folder_path = f"coreml/{modelSize}"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
encoder.save(f"{folder_path}/CoremlEncoder.mlpackage")

# test accuracy
torch_output = traced_encoder.forward(melSegment)
print("torch model output:", torch_output)
melSegment = melSegment.cpu().detach().numpy()

for i in range(1,4):
    startT = timer()
    coreml_output = torch.from_numpy(
        list(encoder.predict({'melSegment': melSegment}).values())[0]
    )
    print(f"coreml prediction {i} took {timer()-startT:.3f}")


print(f"coreml {modelSize} model output:", coreml_output)
diff = torch.abs(torch_output - coreml_output).detach()
print("diff avg,max:", torch.mean(diff), torch.max(diff))

# note
# conversion time on Macbook M1 Air 16GB
# tiny:        7s
# small:      51s (coremltools: 17s + ANECompilerService: 34s), predict: 111ms
# medium:   4m03s (1m36s + 2m27s), 334ms
# large:    9m51s (4m05s + 5m46s, use 9GB memory), 626ms
