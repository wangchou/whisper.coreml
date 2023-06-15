import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# model setting
modelSize="tiny"
model = whisper.load_model(modelSize).cpu()
n_state = 384 # tiny=384, base=512, small=768, medium=1024, large=1280

# trace model by torch.jit
encoder = model.encoder
encoder.eval()

melSegment = torch.ones((1, 80, 3000))
traced_encoder = torch.jit.trace(encoder, melSegment)

# convert to coreml model
encoder = ct.convert(
    traced_encoder,
    inputs=[ct.TensorType(name="melSegment", shape=melSegment.shape)],
    outputs=[ct.TensorType(name="output")],
    compute_units=ct.ComputeUnit.ALL,
)

encoder_fp16 = quantization_utils.quantize_weights(encoder, nbits=16)
encoder_fp16.save(f"coreml/encoder_{modelSize}_fp16.mlmodel")

# test accuracy
torch_output = traced_encoder.forward(melSegment)
print("torch model output:", torch_output)
melSegment = melSegment.cpu().detach().numpy()
coreml_output = torch.from_numpy(
  list(encoder_fp16.predict({'melSegment': melSegment}).values())[0]
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
