import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import os
from timeit import default_timer as timer

modelSize = "medium"
model = whisper.load_model(modelSize).cpu()

encoder = model.encoder
encoder.eval()

melSegment = torch.ones((1, 80, 3000))
traced_encoder = torch.jit.trace(encoder, melSegment)

pipeline = ct.PassPipeline.CLEANUP
pipeline.insert_pass(11, "common::add_fp16_cast") # fp16 for ane
pipeline.remove_passes({
    # fix complex graph caused by speedup_conversion_workaround
    "common::const_deduplication",
})

startT = timer()
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

torch_output = traced_encoder.forward(melSegment)
print("torch model output:", torch_output)

melSegment = melSegment.cpu().detach().numpy()

for i in range(5):
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
# small:      45s (coremltools: 10s + ANECompilerService: 35s), predict: 112ms
# medium:   3m25s (58s + 2m27s), 335ms
# large:    8m00s (2m25s + 5m35s, use 9GB memory), 620ms
