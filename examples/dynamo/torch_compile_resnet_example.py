"""
.. _torch_compile_resnet:

Compiling ResNet using the Torch-TensorRT `torch.compile` Backend
==========================================================

This interactive script is intended as a sample of the Torch-TensorRT workflow with `torch.compile` on a ResNet model."""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt
import torchvision.models as models

# %%

# Initialize model with half precision and sample inputs
model = models.resnet18(pretrained=True).half().eval().to("cuda")
inputs = [torch.randn((1, 3, 224, 224)).to("cuda").half()]

# %%
# Optional Input Arguments to `torch_tensorrt.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Enabled precision for TensorRT optimization
enabled_precisions = {torch.half}

# Whether to print verbose logs
debug = True

# Workspace size for TensorRT
workspace_size = 20 << 30

# Maximum number of TRT Engines
# (Lower value allows more graph segmentation)
min_block_size = 7

# Operations to Run in Torch, regardless of converter support
torch_executed_ops = {}

# %%
# Compilation with `torch_tensorrt.compile`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Build and compile the model with torch.compile, using Torch-TensorRT backend
optimized_model = torch_tensorrt.compile(
    model,
    ir="torch_compile",
    inputs=inputs,
    enabled_precisions=enabled_precisions,
    debug=debug,
    workspace_size=workspace_size,
    min_block_size=min_block_size,
    torch_executed_ops=torch_executed_ops,
    use_python_runtime=True,
)

# %%
# Equivalently, we could have run the above via the torch.compile frontend, as so:
# `optimized_model = torch.compile(model, backend="torch_tensorrt", options={"enabled_precisions": enabled_precisions, ...}); optimized_model(*inputs)`

# %%
# Inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Does not cause recompilation (same batch size as input)
new_inputs = [torch.randn((1, 3, 224, 224)).half().to("cuda")]
new_outputs = optimized_model(*new_inputs)

# %%

# Does cause recompilation (new batch size)
new_batch_size_inputs = [torch.randn((8, 3, 224, 224)).half().to("cuda")]
new_batch_size_outputs = optimized_model(*new_batch_size_inputs)

# %%
# Cleanup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Finally, we use Torch utilities to clean up the workspace
torch._dynamo.reset()

# %%
# Cuda Driver Error Note
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Occasionally, upon exiting the Python runtime after Dynamo compilation with `torch_tensorrt`,
# one may encounter a Cuda Driver Error. This issue is related to https://github.com/NVIDIA/TensorRT/issues/2052
# and can be resolved by wrapping the compilation/inference in a function and using a scoped call, as in::
#
#       if __name__ == '__main__':
#           compile_engine_and_infer()
