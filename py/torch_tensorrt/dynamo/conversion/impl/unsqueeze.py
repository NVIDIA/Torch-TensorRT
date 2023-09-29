from typing import Optional, cast

from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_positive_dim,
    get_trt_tensor,
)
from torch_tensorrt.fx.converters.converter_utils import set_layer_name
from torch_tensorrt.fx.types import Shape, TRTTensor
from torch_tensorrt.fx.utils import get_dynamic_dims


def unsqueeze(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_t: TRTTensor,
    dim: Shape,
) -> TRTTensor:
    input_val = get_trt_tensor(ctx, input_t, f"{name}_input_t")
    if not isinstance(input_val, TRTTensor):
        raise RuntimeError(
            f"unsqueeze received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    dim = cast(int, dim)

    input_shape_size = (
        len(input_val.shape) + 1
        if ctx.net.has_implicit_batch_dimension
        else len(input_val.shape)
    )
    dim = get_positive_dim(dim, input_shape_size + 1)

    if ctx.net.has_implicit_batch_dimension:
        assert dim != 0
        dim -= 1

    assert (
        len(get_dynamic_dims(input_val.shape)) <= 1
    ), "Currently we don't support unsqueeze with more than one dynamic dims."
    layer = ctx.net.add_shuffle(input_val)
    layer.reshape_dims = (
        tuple(input_val.shape)[:dim] + (1,) + tuple(input_val.shape)[dim:]
    )
    set_layer_name(layer, target, name, source_ir)
    return layer.get_output(0)
