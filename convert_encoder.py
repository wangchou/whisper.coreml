import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import os

# model setting
modelSize = "tiny"
model = whisper.load_model(modelSize).cpu()

# trace model by torch.jit
encoder = model.encoder
encoder.eval()

melSegment = torch.ones((1, 80, 3000))
traced_encoder = torch.jit.trace(encoder, melSegment)

# convert to coreml model
encoder = ct.convert(
    traced_encoder,
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="melSegment", shape=melSegment.shape)],
    outputs=[ct.TensorType(name="output")],
    compute_units=ct.ComputeUnit.ALL,
)

folder_path = f"coreml/{modelSize}"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)
encoder.save(f"{folder_path}/CoremlEncoder.mlpackage")

# test accuracy
torch_output = traced_encoder.forward(melSegment)
print("torch model output:", torch_output)
melSegment = melSegment.cpu().detach().numpy()
coreml_output = torch.from_numpy(
  list(encoder.predict({'melSegment': melSegment}).values())[0]
)
print(f"coreml {modelSize} model output:", coreml_output)
diff = torch.abs(torch_output - coreml_output).detach()
print("diff avg,max:", torch.mean(diff), torch.max(diff))

# note
# convertion time on Macbook M1 Air 16GB
# tiny:       28s
# small:   5 mins
# medium: 40 mins (29GB)
# large:  crashed, use 60+GB memory after 23mins
