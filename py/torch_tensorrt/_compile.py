from typing import List, Dict, Any
import torch_tensorrt.ts

from torch_tensorrt import logging
import torch
import torch.fx
from enum import Enum

import torch_tensorrt.fx
from torch_tensorrt.fx.utils import LowerPrecision


class _IRType(Enum):
    """Enum to set the minimum required logging level to print a message to stdout"""

    ts = 0
    fx = 1
    dynamo = 2
    torch_compile = 3


class _ModuleType(Enum):
    """Enum to set the minimum required logging level to print a message to stdout"""

    nn = 0
    ts = 1
    fx = 2


def _parse_module_type(module: Any) -> _ModuleType:
    if any(
        isinstance(module, t)
        for t in [torch.jit.ScriptModule, torch.jit.ScriptFunction]
    ):
        return _ModuleType.ts
    elif isinstance(module, torch.fx.GraphModule):
        return _ModuleType.fx
    elif isinstance(module, torch.nn.Module):
        return _ModuleType.nn
    else:
        raise RuntimeError("Module is an unknown format")


def _get_target_ir(module_type: _ModuleType, ir: str) -> _IRType:
    module_is_tsable = any([module_type == t for t in [_ModuleType.nn, _ModuleType.ts]])
    module_is_fxable = any([module_type == t for t in [_ModuleType.nn, _ModuleType.fx]])

    ir_targets_torchscript = any([ir == opt for opt in ["torchscript", "ts"]])
    ir_targets_fx = ir == "fx"
    ir_targets_dynamo = ir == "dynamo"
    ir_targets_torch_compile = ir == "torch_compile"

    if module_is_tsable and ir_targets_torchscript:
        return _IRType.ts
    elif module_is_fxable and ir_targets_fx:
        return _IRType.fx
    elif module_is_fxable and ir_targets_dynamo:
        return _IRType.dynamo
    elif module_is_fxable and ir_targets_torch_compile:
        return _IRType.torch_compile
    else:
        if ir == "default":
            # Options are listed in order of preference
            if module_is_fxable:
                logging.log(
                    logging.Level.Info, "ir was set to default, using dynamo as ir"
                )
                return _IRType.dynamo
            elif module_is_tsable:
                logging.log(
                    logging.Level.Warning,
                    "Input graph is a Torchscript module but the ir provided is default (dynamo). Please set ir=torchscript to suppress the warning. Compiling the module with ir=torchscript",
                )
                return _IRType.ts
            else:
                raise ValueError("Module was provided in an unsupported format")
        else:
            raise ValueError("Unknown ir was requested")


def compile(
    module: Any,
    ir="default",
    inputs=[],
    enabled_precisions=set([torch.float]),
    **kwargs,
):
    """Compile a PyTorch module for NVIDIA GPUs using TensorRT

    Takes a existing PyTorch module and a set of settings to configure the compiler
    and using the path specified in ``ir`` lower and compile the module to TensorRT
    returning a PyTorch Module back

    Converts specifically the forward method of a Module

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)

    Returns:
        torch.nn.Module: Compiled Module, when run it will execute via TensorRT
    """
    module_type = _parse_module_type(module)
    target_ir = _get_target_ir(module_type, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if module_type == _ModuleType.nn:
            logging.log(
                logging.Level.Info,
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript",
            )
            ts_mod = torch.jit.script(module)
        return torch_tensorrt.ts.compile(
            ts_mod, inputs=inputs, enabled_precisions=enabled_precisions, **kwargs
        )
    elif target_ir == _IRType.fx:
        if (
            torch.float16 in enabled_precisions
            or torch_tensorrt.dtype.half in enabled_precisions
        ):
            lower_precision = LowerPrecision.FP16
        elif (
            torch.float32 in enabled_precisions
            or torch_tensorrt.dtype.float in enabled_precisions
        ):
            lower_precision = LowerPrecision.FP32
        else:
            raise ValueError(f"Precision {enabled_precisions} not supported on FX")

        return torch_tensorrt.fx.compile(
            module,
            inputs,
            lower_precision=lower_precision,
            max_batch_size=inputs[0].size(0),
            explicit_batch_dimension=True,
            dynamic_batch=False,
            **kwargs,
        )
    elif target_ir == _IRType.dynamo:
        from torch_tensorrt import Device
        from torch_tensorrt.dynamo.utils import prepare_inputs, prepare_device
        import collections.abc

        if not isinstance(inputs, collections.abc.Sequence):
            inputs = [inputs]
        device = kwargs.get("device", Device._current_device())
        torchtrt_inputs, torch_inputs = prepare_inputs(inputs, prepare_device(device))
        module = torch_tensorrt.dynamo.trace(module, torch_inputs, **kwargs)
        return torch_tensorrt.dynamo.compile(
            module,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            **kwargs,
        )
    elif target_ir == _IRType.torch_compile:
        return torch_compile(module, enabled_precisions=enabled_precisions, **kwargs)
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")


def torch_compile(module, **kwargs):
    """
    Returns a boxed model which is the output of torch.compile.
    This does not compile the model to TRT. Execute this model on
    sample inputs to compile the model to TRT.
    """
    from torch_tensorrt.dynamo.backend import torch_tensorrt_backend

    boxed_fn = torch.compile(module, backend=torch_tensorrt_backend, options={**kwargs})

    return boxed_fn


def convert_method_to_trt_engine(
    module: Any,
    method_name: str,
    ir="default",
    inputs=[],
    enabled_precisions=set([torch.float]),
    **kwargs,
):
    """Convert a TorchScript module method to a serialized TensorRT engine

    Converts a specified method of a module to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)
    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    module_type = _parse_module_type(module)
    target_ir = _get_target_ir(module_type, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if module_type == _ModuleType.nn:
            logging.log(
                logging.Level.Info,
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript",
            )
            ts_mod = torch.jit.script(module)
        return torch_tensorrt.ts.convert_method_to_trt_engine(
            ts_mod,
            method_name,
            inputs=inputs,
            enabled_precisions=enabled_precisions,
            **kwargs,
        )
    elif target_ir == _IRType.fx:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=fx"
        )
    elif target_ir == _IRType.dynamo:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=dynamo."
        )
    elif target_ir == _IRType.torch_compile:
        raise RuntimeError(
            "convert_method_to_trt_engine call is not supported for ir=torch_compile"
        )
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")
