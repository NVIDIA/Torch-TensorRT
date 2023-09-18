import functools
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import tensorrt as trt
import torch
from torch import SymBool, SymFloat, SymInt
from torch.fx.node import Argument, Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion.converter_registry import (
    ConverterImplSignature,
    ConverterRegistry,
)
from torch_tensorrt.fx.converters.converter_utils import (
    Frameworks,
    get_axes_for_reduce_op,
    to_numpy,
    unified_dtype_converter,
)
from torch_tensorrt.fx.types import TRTDataType, TRTNetwork, TRTTensor

_LOGGER: logging.Logger = logging.getLogger(__name__)


def get_node_name(node: torch.fx.Node) -> str:
    # nn_module_stack preserves the call stack of pytorch nn.modules
    # The call stack contains a detailed name of the module
    # which shows exactly where the module is located in the
    # network architecture.
    stack_item = node.meta.get("nn_module_stack", None)
    # The current node is the last item in the stack
    mod_stack = stack_item.popitem() if stack_item else ""
    node_name = str(node)
    if mod_stack:
        mod_name = str(mod_stack[0]).replace("___", "/")
        # Clean up the module name
        mod_name = re.sub("^.*__self", "", mod_name)
        mod_name = re.sub(r"_(\d+)$", r"/\g<1>", mod_name)
        node_name = mod_name + "/" + node_name
    else:
        # Try an alternative way to get the module info
        # like the node.meta['source_fn'] attr
        pass

    _LOGGER.debug(f"Node meta name {node_name}")
    return node_name


def is_only_operator_on_placeholder(node: torch.fx.Node) -> bool:
    """Detects whether a call_function node is the only operator on a placeholder"""
    # Returns true if the node operates on a placeholder and is a direct output
    return (
        node.op == "call_function"
        and any(
            arg.op == "placeholder"
            for arg in node.args
            if isinstance(arg, torch.fx.Node)
        )
        and any(user.op == "output" for user in list(node.users.keys()))
    )


def dynamic_unsupported(node: torch.fx.Node) -> bool:
    """Validates that a node has no dynamic args, kwargs, or outputs"""
    return _dynamic_unsupported(node=node)


def dynamic_unsupported_with_args(
    arg_positions_to_check: Optional[List[int]] = None,
) -> Callable[[torch.fx.Node], bool]:
    """Returns a validator that a node has no dynamic args at specific positions"""
    return functools.partial(
        _dynamic_unsupported, arg_positions_to_check=arg_positions_to_check
    )


def _dynamic_unsupported(
    node: torch.fx.Node, arg_positions_to_check: Optional[List[int]] = None
) -> bool:
    # Validate that none of the inputs to the node have Dynamic shapes
    assert isinstance(
        node, torch.fx.Node
    ), "Inputs to validator functions must be FX Nodes"

    def _is_subnode_dynamic(subnode: torch.fx.Node) -> bool:
        """Checks if a node itself has Dynamic properties"""
        return getattr(
            subnode.meta["val"], "_has_symbolic_sizes_strides", False
        ) or isinstance(subnode.meta["val"], (SymFloat, SymInt, SymBool))

    # Check node value itself
    if arg_positions_to_check is None and _is_subnode_dynamic(node):
        return False

    # Check node arguments individually
    if arg_positions_to_check is None and any(
        _is_subnode_dynamic(arg) for arg in node.args if isinstance(arg, torch.fx.Node)
    ):
        return False
    # Check specific arg positions if the caller has specified positions to check
    elif arg_positions_to_check is not None and any(
        _is_subnode_dynamic(node.args[i])
        for i in arg_positions_to_check
        if isinstance(node.args[i], torch.fx.Node)
    ):
        return False

    # Check node keyword arguments individually
    if arg_positions_to_check is None and any(
        _is_subnode_dynamic(kwarg)
        for kwarg in node.kwargs.values()
        if isinstance(kwarg, torch.fx.Node)
    ):
        return False

    return True


def cast_trt_tensor(
    network: TRTNetwork,
    input_val: TRTTensor,
    dtype: TRTDataType,
    name: str,
    target: Target = "",
    source_ir: Optional[SourceIR] = None,
) -> TRTTensor:
    """Given a TRT Tensor, convert that Tensor to the specified dtype

    Adds an Identity layer to the network which performs the conversion
    if the input's dtype is different from the cast type. Otherwise returns
    input unchanged

    Args:
        network (TRTNetwork): A TensorRT network
        input_val (TRTTensor): A TRT Tensor to cast to a new data type
        dtype (TRTDataType, torch.dtype, np.dtype): The data type to cast the input Tensor to
        name (str): Name of the calling layer
        target (Target): Target of calling node
        source_ir (SourceIR): SourceIR of calling converter
    Returns:
        A TensorRT ITensor which has been casted to the specified dtype
    """
    trt_dtype = unified_dtype_converter(dtype, Frameworks.TRT)

    if input_val.dtype != trt_dtype:
        source_ir = source_ir if source_ir is not None else SourceIR.UNKNOWN
        target_str = ConverterRegistry.qualified_name_or_str(target)
        target_name = f"{source_ir}_ops{('.' + target_str) if target_str else ''}"

        identity_layer = network.add_identity(input_val)
        identity_layer.set_output_type(0, trt_dtype)
        identity_layer.name = f"Cast ITensor {input_val.name} from {input_val.dtype} to {trt_dtype} - [{target_name}]-[{name}]"
        return identity_layer.get_output(0)
    else:
        return input_val


def cast_int_int_div_trt_tensor(
    network: TRTNetwork,
    lhs_val: TRTTensor,
    rhs_val: TRTTensor,
    name: str,
) -> List[TRTTensor]:
    """
    Given two `int` data type TRT Tensor to div operation, cast the TRT Tensor to float type
    Args:
        network (TRTNetwork): A TensorRT network
        lhs_val (TRTTensor): A TRT Tensor numerator
        rhs_val (TRTTensor): A TRT Tensor numerator
        name (str): Name of calling layer
    Returns:
        A list of lhs_val and rhs_val casted to the approriate datatype
    """
    if lhs_val.dtype == trt.int32 and rhs_val.dtype == trt.int32:
        lhs_val = cast_trt_tensor(network, lhs_val, trt.float32, name)
        rhs_val = cast_trt_tensor(network, rhs_val, trt.float32, name)
    return [lhs_val, rhs_val]


def broadcastable(
    a: TRTTensor,
    b: TRTTensor,
) -> bool:
    "Check if two tensors are broadcastable according to torch rules"
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    # check from the trailing
    diff = len(a_shape) - len(b_shape)

    # Validate tensors have same rank and shape
    if diff == 0 and all(a_shape[i] == b_shape[i] for i in range(len(a_shape))):
        return True

    # Left-pad the shorter dimension with ones
    if diff > 0:
        b_shape = (1,) * abs(diff) + b_shape
    else:
        a_shape = (1,) * abs(diff) + a_shape

    # Validate one of the following conditions for broadcastability per-dimension
    # 1. Equal number of dimensions or 2. Dimension has shape 1
    for i in range(len(a_shape)):
        if not (a_shape[i] == b_shape[i] or a_shape[i] == 1 or b_shape[i] == 1):
            return False
    return True


get_axes_for_reduce_op = functools.partial(
    get_axes_for_reduce_op, has_implicit_batch_dimension=False
)


def extend_attr_to_tuple(
    val: Any,
    num_elem: int,
) -> Tuple[Any, ...]:
    """
    If `val` is not a tuple or a list, then we make a tuple of size `num_elem` by
    replicating `val` `num_elem` times.

    Args:
        val (Any): Value that we want to process.

    Returns:
        A tuple.
    """
    if not isinstance(val, (tuple, list)):
        val = (val,) * num_elem
    elif len(val) == 1:
        val = (val[0],) * num_elem

    if isinstance(val, list):
        val = tuple(val)

    if isinstance(val, tuple):
        return val
    else:
        raise AssertionError(f"Object {val} could not be extended to tuple")


def cast_int_or_float_to_bool(
    network: TRTNetwork, name: str, tensor: TRTTensor
) -> TRTTensor:
    if tensor.dtype != trt.bool:
        return cast_trt_tensor(network, tensor, trt.bool, name)

    return tensor


def create_constant(
    network: TRTNetwork,
    value: Union[int, float, bool, np.ndarray, torch.Tensor],
    name: str,
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType]],
) -> TRTTensor:
    """
    Add a TensorRT constant layer whose value is `value` to `network`.
    Args:
        network (TRTNetwork): A TensorRT network to which we want to add
            a constant layer.
        value (Union[int, float, bool, np.ndarray, torch.Tensor]): A literal value, Numpy array,
            or a PyTorch tensor that will be used as value of the added TensorRT Constant layer.
        name (str): Name of the added TensorRT Constant layer.
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If a dtype is given, we will convert the type of the given `value` to this dtype.
    Returns:
        A TensorRT ITensor that represents the given value.
    """
    constant = network.add_constant(
        (1,) if isinstance(value, (int, float, bool)) else value.shape,
        to_numpy(value, dtype).copy(),
    )
    constant.name = name
    return constant.get_output(0)


def get_trt_tensor(
    network: TRTNetwork,
    input_val: Any,
    name: str,
    dtype: Optional[Union[torch.dtype, np.dtype, TRTDataType]] = None,
) -> TRTTensor:
    """
    Given a value of random type, we try to convert it to a TensorRT ITensor.
    An runtime error is raised if we're not able to do that.
    Args:
        network (TRTNetwork): A TensorRT network. If we want to
            add a TensorRT Constant layer, we will add it to this network.
        input_val (Any): An value that we want to convert to a TensorRT ITensor.
        name (str): The name of the created TensorRT Constant layer if there's
            one.
        dtype (Optional[Union[torch.dtype, np.dtype, TRTDataType]]):
            If dtype is provided, the given value will be converted to this dtype.
    Returns:
        A TensorRT ITensor that represents the given value.
    """
    # If the input is 64-bit, cast it to 32-bit for TRT freezing
    if isinstance(input_val, torch.Tensor):
        if input_val.dtype == torch.int64:
            input_val = input_val.to(torch.int32)
        elif input_val.dtype == torch.float64:
            input_val = input_val.to(torch.float32)
    elif isinstance(input_val, np.ndarray):
        if input_val.dtype == np.int64:
            input_val = input_val.astype(np.int32)
        elif input_val.dtype == np.float64:
            input_val = input_val.astype(np.float32)

    if isinstance(input_val, (torch.Tensor, np.ndarray, int, float, bool)):
        return create_constant(network, input_val, name, dtype)
    elif isinstance(input_val, TRTTensor):
        return input_val
    else:
        raise AssertionError(f"Cannot convert {input_val} to TRT constant")


@overload
def get_positive_dim(dim: int, dim_size: int) -> int:
    ...


@overload
def get_positive_dim(dim: Sequence[int], dim_size: int) -> Tuple[int, ...]:
    ...


def get_positive_dim(
    dim: Union[int, Sequence[int]], dim_size: int
) -> Union[int, Tuple[int, ...]]:
    """
    Given an integer number or tuple that represents dimension(s) in the array,
    transform it to a positive integer dim if it's negative. Otherwise, do
    nothing.

    Args:
        dim (Union[int, Sequence[int]]): A integer or Sequence of integers that represent dimension(s) in an array.
        dim_size (int): The size of the dimension in the array.

    Returns:
        A positive integer or tuple of integers that represent the same dimension as the given dim.
    """

    def positive_dim(d: int) -> int:
        if d < 0:
            return d % dim_size
        return d

    return (
        positive_dim(dim)
        if isinstance(dim, int)
        else tuple(positive_dim(d) for d in dim)
    )


def enforce_tensor_types(
    type_dictionary: Dict[Union[int, str], Tuple[Union[TRTTensor, np.ndarray], ...]],
    promote: bool = True,
) -> Callable[[ConverterImplSignature], ConverterImplSignature]:
    """Decorator to enforce tensor types for input arguments to converters

    Keys in the type dictionary must be integers if they refer to a positional
    argument in args, or strings if they refer to a keyword argument in kwargs

    Values must be tuples of data types denoting the approved data types for a given position
    The approved types are TRTTensor, np.ndarray, and torch.Tensor.

    Note: torch.Tensor cannot be present without np.ndarray

    The promote argument controls whether tensors will be promoted if they are of the
    incorrect format
    """
    assert all(
        isinstance(key, (int, str)) for key in type_dictionary
    ), "Invalid key for type enforcement"
    assert all(
        (
            isinstance(val, tuple)
            and not (torch.Tensor in val and np.ndarray not in val)
            and all((dtype in (TRTTensor, np.ndarray, torch.Tensor)) for dtype in val)
        )
        for val in type_dictionary.values()
    ), "Invalid value(s) specified in type enforcement"

    def wrapper(func: ConverterImplSignature) -> ConverterImplSignature:
        @functools.wraps(func)
        def convert_with_type_enforcement(
            network: TRTNetwork,
            target: Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
            name: str,
        ) -> Union[TRTTensor, Sequence[TRTTensor]]:
            new_args = args
            new_kwargs = {**kwargs}
            new_value = None

            # Go through type dictionary and promote types accordingly
            for index, approved_dtypes in type_dictionary.items():
                # Referencing an arg
                if isinstance(index, int):
                    candidate = args[index]
                # Referencing a kwarg
                elif isinstance(index, str):
                    candidate = kwargs[index]

                # If the candidate Tensor is already an approved type, do nothing
                if isinstance(candidate, approved_dtypes):
                    continue
                # If the candidate Tensor is not an approved type, but promotion is disabled, error
                elif not promote:
                    raise AssertionError(
                        f"Detected argument at index {index} had type {type(candidate)} "
                        f"which is not one of the approved types {approved_dtypes}"
                    )
                # Numpy arrays are preferred in general - if approved, promote to Numpy first
                elif np.ndarray in approved_dtypes and not isinstance(
                    candidate, TRTTensor
                ):
                    new_value = to_numpy(candidate)
                # As a fallback, freeze tensors to IConstantLayers if they cannot be handled as Numpy arrays
                elif TRTTensor in approved_dtypes:
                    _LOGGER.debug(
                        f"Freezing tensor {name}_constant_{index} to TRT IConstantLayer"
                    )
                    new_value = get_trt_tensor(
                        network, candidate, name + f"_constant_{index}"
                    )
                else:
                    raise AssertionError(
                        f"Argument at index {index} was not able to be converted to one of "
                        f"the following types: {approved_dtypes}"
                    )

                # Reassemble args or kwargs if the value was modified
                if isinstance(index, int):
                    new_args = new_args[:index] + (new_value,) + new_args[index + 1 :]
                elif isinstance(index, str):
                    new_kwargs[index] = new_value

            return func(network, target, new_args, new_kwargs, name)

        return convert_with_type_enforcement

    return wrapper
