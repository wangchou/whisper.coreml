import whisper
import torch
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import os
from timeit import default_timer as timer

# model setting
modelSize = "tiny"
model = whisper.load_model(modelSize).cpu()

# trace model by torch.jit
encoder = model.encoder
encoder.eval()

melSegment = torch.ones((1, 80, 3000))
traced_encoder = torch.jit.trace(encoder, melSegment)

# convert to coreml model
startT = timer()
encoder = ct.convert(
    traced_encoder,
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="melSegment", shape=melSegment.shape)],
    outputs=[ct.TensorType(name="output")],
    compute_units=ct.ComputeUnit.ALL,
)
print("---")
print(f"coreml conversion took {timer()-startT:.3f}")
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
# convertion time on Macbook M1 Air 16GB
# tiny:       10s
# small:    1m10s
# medium:   7m15s
# large:


#pipeline = ct.PassPipeline()
#pipeline.remove_passes({
##    "common::lower_complex_dialect_ops",
##    "common::update_output_dtypes",
##    "common::cast_optimization",
##    "common::noop_elimination",
#    # quantization pass 1: canonicalize zero point
##    # always start quantization passes with canonicalizations
#    "common::nullify_redundant_quantization_zero_point",
##    # quantization pass 2: remove redundancy
#    # remove redundancy after canonicalization but before anything else
#    "common::dequantize_quantize_pair_elimination",
##    # the main quantization passes
#    "common::distributive_quantized_binary_op_scale_normalization",
#    # the last quantization pass: replace const dequantize with constexpr
#    # after all quantization passes, since constexpr will not be further optimized
#    # before const elimination, otherwise const dequantize would get bloated
#    "common::dequantize_to_constexpr",
##    "common::const_elimination",
#    "common::sanitize_input_output_names",
#    "common::divide_to_multiply",
#    "common::add_conv_transpose_output_shape",
##    "common::const_elimination",
#    "common::const_deduplication",  # after all consts have been settled
#    "common::loop_invariant_elimination",
#    "common::remove_symbolic_reshape",
#    "common::noop_elimination",
#    "common::fuse_matmul_weight_bias",
#    "common::fuse_linear_bias",
#    "common::fuse_gelu_tanh_approximation",
#    "common::fuse_gelu_exact",
#    "common::fuse_leaky_relu",
#    "common::rank0_expand_dims_swap",
#    "common::compose_conv1d",  # compose conv1d before any other conv passes
#    "common::use_reflection_padding",
#    "common::merge_consecutive_paddings",
#    # Should come after use_reflection_padding, which will introduce new padding layers
#    "common::fuse_pad_conv",  # Should come after merge_consecutive_paddings
#    "common::image_input_preprocess",
#    "common::replace_stack_reshape",
#    # should come before detect_concat_interleave since it may add concat
#    "common::reduce_transposes",
#    "common::fuse_conv_scale",
#    "common::fuse_conv_bias",
#    "common::fuse_onehot_matmul_to_gather",
#    "common::fuse_layernorm_or_instancenorm",
#    # should come after reduce_transposes, to detect instance_norm
#    "common::fuse_elementwise_to_batchnorm",  # should come after fuse_layernorm_or_instancenorm
#    "common::fuse_reduce_mean",  # should come after fuse_layernorm_or_instancenorm
#    "common::fuse_conv_batchnorm",  # should come after fuse_elementwise_to_batchnorm
#    "common::fuse_conv_scale",
#    # Re-run the fuse conv scale pass after the conv and batch_norm are fused
#    "common::fuse_conv_bias",
#    # Re-run the fuse conv bias pass after the conv and batch_norm are fused
#    "common::fuse_conv_batchnorm",
#    # In some cases, we need to run conv / batch_norm fusion again after the fuse_conv_scale and fuse_conv_bias passes
#    "common::detect_concat_interleave",
#    "common::concat_to_pixel_shuffle",
#    # should come after detect_concat_interleave and after replace_stack_reshape
#    "common::fuse_prelu",
#    # reduce_transpose pass should run before and after this pass (the one after will be run during the cleanup passes stage)
#    "common::prelu_to_lrelu",
#    "common::merge_consecutive_relus",
#    "common::merge_consecutive_reshapes",
#    "common::merge_consecutive_transposes",
#    # "expand_high_rank_reshape_and_transpose" must come after "common::merge_consecutive_transposes"
#    "common::expand_high_rank_reshape_and_transpose",
#    "common::reduce_transposes",
#    # "remove_redundant_ops" pass should be applied towards the end, once other graph passes have done their optimizations.
#    # For instance, it should come after passes such as "reduce_transpose" that can introduce redundant transposes
#    # in the network (while reducing the total number of transposes), and after passes such as "fuse_layernorm_or_instancenorm"
#    # which detects patterns that involve redundant ops ("sub") etc.
#    "common::remove_redundant_ops",
##    "common::add_fp16_cast",  # Will be removed if compute precision is not FP16.
##    "common::dead_code_elimination",  # always end with dce
#})
