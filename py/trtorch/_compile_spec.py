from typing import List, Dict, Any, Set
import torch
import trtorch._C
from trtorch import _types
from trtorch.Input import Input
from trtorch.Device import Device

import warnings


def _supported_input_size_type(input_size: Any) -> bool:
    if isinstance(input_size, torch.Size):
        return True
    elif isinstance(input_size, tuple):
        return True
    elif isinstance(input_size, list):
        return True
    else:
        raise TypeError(
            "Input sizes for inputs are required to be a List, tuple or torch.Size or a Dict of three sizes (min, opt, max), found type: "
            + str(type(input_size)))


def _parse_input_ranges(input_sizes: List) -> List:

    if any(not isinstance(i, dict) and not _supported_input_size_type(i) for i in input_sizes):
        raise KeyError("An input size must either be a static size or a range of three sizes (min, opt, max) as Dict")

    parsed_input_sizes = []
    for i in input_sizes:
        if isinstance(i, dict):
            if all(k in i for k in ["min", "opt", "min"]):
                parsed_input_sizes.append(
                    Input(min_shape=i["min"], opt_shape=i["opt"], max_shape=i["max"])._to_internal())

            elif "opt" in i:
                parsed_input_sizes.append(Input(shape=i["opt"])._to_internal())

            else:
                raise KeyError(
                    "An input size must either be a static size or a range of three sizes (min, opt, max) as Dict")

        elif isinstance(i, list):
            parsed_input_sizes.append(Input(shape=i)._to_internal())

        elif isinstance(i, tuple):
            parsed_input_sizes.append(Input(shape=i)._to_internal())

        elif isinstance(i, torch.Size):
            parsed_input_sizes.append(Input(shape=i)._to_internal())

    return parsed_input_sizes


def _parse_op_precision(precision: Any) -> _types.dtype:
    if isinstance(precision, torch.dtype):
        if precision == torch.int8:
            return _types.dtype.int8
        elif precision == torch.half:
            return _types.dtype.half
        elif precision == torch.float:
            return _types.dtype.float
        else:
            raise TypeError("Provided an unsupported dtype as operating precision (support: int8, half, float), got: " +
                            str(precision))

    elif isinstance(precision, _types.DataTypes):
        return precision

    else:
        raise TypeError("Op precision type needs to be specified with a torch.dtype or a trtorch.dtype, got: " +
                        str(type(precision)))


def _parse_enabled_precisions(precisions: Any) -> Set:
    parsed_precisions = set()
    if any([isinstance(precisions, type) for type in [list, tuple, set]]):
        for p in precisions:
            parsed_precisions.add(_parse_op_precision(p))
    else:
        parsed_precisions.add(_parse_op_precision(precisions))
    return parsed_precisions


def _parse_device_type(device: Any) -> _types.DeviceType:
    if isinstance(device, torch.device):
        if device.type == 'cuda':
            return _types.DeviceType.gpu
        else:
            ValueError("Got a device type other than GPU or DLA (type: " + str(device.type) + ")")
    elif isinstance(device, _types.DeviceType):
        return device
    elif isinstance(device, str):
        if device == "gpu" or device == "GPU":
            return _types.DeviceType.gpu
        elif device == "dla" or device == "DLA":
            return _types.DeviceType.dla
        else:
            ValueError("Got a device type other than GPU or DLA (type: " + str(device) + ")")
    else:
        raise TypeError("Device specification must be of type torch.device, string or trtorch.DeviceType, but got: " +
                        str(type(device)))


def _parse_device(device_info: Any) -> trtorch._C.Device:
    if isinstance(device_info, dict):
        info = trtorch._C.Device()
        if "device_type" not in device_info:
            raise KeyError("Device type is required parameter")
        else:
            assert isinstance(device_info["device_type"], _types.DeviceType)
            info.device_type = _parse_device_type(device_info["device_type"])

        if "gpu_id" in device_info:
            assert isinstance(device_info["gpu_id"], int)
            info.gpu_id = device_info["gpu_id"]

        if "dla_core" in device_info:
            assert isinstance(device_info["dla_core"], int)
            info.dla_core = device_info["dla_core"]

        if "allow_gpu_fallback" in device_info:
            assert isinstance(device_info["allow_gpu_fallback"], bool)
            info.allow_gpu_fallback = device_info["allow_gpu_fallback"]

        return info
    elif isinstance(device_info, Device):
        return device_info._to_internal()
    elif isinstance(device_info, torch.device):
        return (Device._from_torch_device(device_info))._to_internal()
    else:
        raise ValueError(
            "Unsupported data for device specification. Expected either a dict, trtorch.Device or torch.Device")


def _parse_torch_fallback(fallback_info: Dict[str, Any]) -> trtorch._C.TorchFallback:
    info = trtorch._C.TorchFallback()
    if "enabled" not in fallback_info:
        raise KeyError("Enabled is required parameter")
    else:
        assert isinstance(fallback_info["enabled"], bool)
        info.enabled = fallback_info["enabled"]
    if "min_block_size" in fallback_info:
        assert isinstance(fallback_info["min_block_size"], int)
        info.min_block_size = fallback_info["min_block_size"]

    if "forced_fallback_ops" in fallback_info:
        assert isinstance(fallback_info["forced_fallback_ops"], list)
        info.forced_fallback_operators = fallback_info["forced_fallback_ops"]

    if "forced_fallback_modules" in fallback_info:
        assert isinstance(fallback_info["forced_fallback_modules"], list)
        info.forced_fallback_modules = fallback_info["forced_fallback_modules"]

    return info


def _parse_compile_spec(compile_spec: Dict[str, Any]) -> trtorch._C.CompileSpec:
    info = trtorch._C.CompileSpec()
    if "input_shapes" not in compile_spec and "inputs" not in compile_spec:
        raise KeyError(
            "Module input definitions are requried to compile module. Provide a list of trtorch.Input keyed to \"inputs\" in the compile spec"
        )

    if "input_shapes" in compile_spec and "inputs" in compile_spec:
        raise KeyError(
            "Found both key \"input_shapes\", and \"inputs\" in compile spec, please port forward to using only \"inputs\""
        )

    if "input_shapes" in compile_spec:
        warnings.warn(
            "Key \"input_shapes\" is deprecated in favor of \"inputs\". Support for \"input_shapes\" will be removed in TRTorch v0.5.0",
            DeprecationWarning)
        info.inputs = _parse_input_ranges(compile_spec["input_shapes"])

    if "inputs" in compile_spec:
        info.inputs = [i._to_internal() for i in compile_spec["inputs"]]

    if "op_precision" in compile_spec and "enabled_precisions" in compile_spec:
        raise KeyError(
            "Found both key \"op_precision\", and \"enabled_precisions\" in compile spec, please port forward to using only \"enabled_precisions\""
        )

    if "op_precision" in compile_spec:
        warnings.warn(
            "Key \"op_precision\" is being deprecated in favor of \"enabled_precision\" which expects a set of precisions to be enabled during compilation (FP32 will always be enabled), Support for \"op_precision\" will be removed in TRTorch v0.5.0",
            DeprecationWarning)
        info.enabled_precisions = _parse_enabled_precisions(compile_spec["op_precision"])

    if "enabled_precisions" in compile_spec:
        info.enabled_precisions = _parse_enabled_precisions(compile_spec["enabled_precisions"])
        # We want default behavior to match PyTorch, so in the case the user did not explicitly set the dtype for inputs they
        # will follow PyTorch convetions
        for i in info.inputs:
            if not i._explicit_set_dtype:
                if _types.dtype.int8 in info.enabled_precisions:
                    i.dtype = _types.dtype.float32
                elif _types.dtype.half in info.enabled_precisions:
                    i.dtype = _types.dtype.float16
                else:
                    i.dtype = _types.dtype.float32

    if "calibrator" in compile_spec:
        info.ptq_calibrator = compile_spec["calibrator"]

    if "sparse_weights" in compile_spec:
        assert isinstance(compile_spec["sparse_weights"], bool)
        info.sparse_weights = compile_spec["sparse_weights"]

    if "disable_tf32" in compile_spec:
        assert isinstance(compile_spec["disable_tf32"], bool)
        info.disable_tf32 = compile_spec["disable_tf32"]

    if "refit" in compile_spec:
        assert isinstance(compile_spec["refit"], bool)
        info.refit = compile_spec["refit"]

    if "debug" in compile_spec:
        assert isinstance(compile_spec["debug"], bool)
        info.debug = compile_spec["debug"]

    if "strict_types" in compile_spec:
        assert isinstance(compile_spec["strict_types"], bool)
        info.strict_types = compile_spec["strict_types"]

    if "device" in compile_spec:
        info.device = _parse_device(compile_spec["device"])

    if "capability" in compile_spec:
        assert isinstance(compile_spec["capability"], _types.EngineCapability)
        info.capability = compile_spec["capability"]

    if "num_min_timing_iters" in compile_spec:
        assert type(compile_spec["num_min_timing_iters"]) is int
        info.num_min_timing_iters = compile_spec["num_min_timing_iters"]

    if "num_avg_timing_iters" in compile_spec:
        assert type(compile_spec["num_avg_timing_iters"]) is int
        info.num_avg_timing_iters = compile_spec["num_avg_timing_iters"]

    if "workspace_size" in compile_spec:
        assert type(compile_spec["workspace_size"]) is int
        info.workspace_size = compile_spec["workspace_size"]

    if "max_batch_size" in compile_spec:
        assert type(compile_spec["max_batch_size"]) is int
        info.max_batch_size = compile_spec["max_batch_size"]

    if "truncate_long_and_double" in compile_spec:
        assert type(compile_spec["truncate_long_and_double"]) is bool
        info.truncate_long_and_double = compile_spec["truncate_long_and_double"]

    if "torch_fallback" in compile_spec:
        info.torch_fallback = _parse_torch_fallback(compile_spec["torch_fallback"])

    return info


def TensorRTCompileSpec(compile_spec: Dict[str, Any]) -> torch.classes.tensorrt.CompileSpec:
    """
    Utility to create a formated spec dictionary for using the PyTorch TensorRT backend

    Args:
        compile_spec (dict): Compilation settings including operating precision, target device, etc.
            One key is required which is ``input_shapes``, describing the input sizes or ranges for inputs
            to the graph as well as expect types and formats for those inputs. All other keys are optional.
            Entries for each method to be compiled.

            Note: Partial compilation of TorchScript modules is not supported through the PyTorch TensorRT backend
            If you need this feature, use trtorch.compile to compile your module. Usage of the resulting module is
            as if you were using the TensorRT integration.

            .. code-block:: py

                CompileSpec = {
                    "forward" : trtorch.TensorRTCompileSpec({
                        "inputs": [
                            trtorch.Input((1, 3, 224, 224)), # Static input shape for input #1
                            trtorch.Input(
                                min_shape=1, 3, 224, 224),
                                opt_shape=(1, 3, 512, 512),
                                max_shape=(1, 3, 1024, 1024),
                                dtype=torch.int32
                                format=torch.channel_last
                            ) # Dynamic input shape for input #2
                        ],
                        "device": {
                            "device_type": torch.device("cuda"), # Type of device to run engine on (for DLA use trtorch.DeviceType.DLA)
                            "gpu_id": 0, # Target gpu id to run engine (Use Xavier as gpu id for DLA)
                            "dla_core": 0, # (DLA only) Target dla core id to run engine
                            "allow_gpu_fallback": false, # (DLA only) Allow layers unsupported on DLA to run on GPU
                        },
                        "enabled_precisions": {torch.half}, # Operating precision set to FP16
                        "sparse_weights": Enable sparsity for convolution and fully connected layers.
                        "disable_tf32": False, # Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
                        "refit": False, # enable refit
                        "debug": False, # enable debuggable engine
                        "strict_types": False, # kernels should strictly run in operating precision
                        "capability": trtorch.EngineCapability.DEFAULT, # Restrict kernel selection to safe gpu kernels or safe dla kernels
                        "num_min_timing_iters": 2, # Number of minimization timing iterations used to select kernels
                        "num_avg_timing_iters": 1, # Number of averaging timing iterations used to select kernels
                        "workspace_size": 0, # Maximum size of workspace given to TensorRT
                        "max_batch_size": 0, # Maximum batch size (must be >= 1 to be set, 0 means not set)
                        "truncate_long_and_double": False, # Truncate long and double into int and float
                    })
                }

            Input Sizes can be specified as torch sizes, tuples or lists. Op precisions can be specified using
            torch datatypes or trtorch datatypes and you can use either torch devices or the trtorch device type enum
            to select device type.

    Returns:
        torch.classes.tensorrt.CompileSpec: List of methods and formated spec objects to be provided to ``torch._C._jit_to_tensorrt``
    """

    parsed_spec = _parse_compile_spec(compile_spec)

    backend_spec = torch.classes.tensorrt.CompileSpec()

    for i in parsed_spec.inputs:
        clone = torch.classes.tensorrt._Input()
        clone._set_min(i.min)
        clone._set_opt(i.opt)
        clone._set_max(i.max)
        clone._set_dtype(i.dtype)
        clone._set_format(i.format)
        clone._set_input_is_dynamic(i.input_is_dynamic)
        backend_spec._append_input(clone)

    d = torch.classes.tensorrt._Device()
    d._set_device_type(int(parsed_spec.device.device_type))
    d._set_gpu_id(parsed_spec.device.gpu_id)
    d._set_dla_core(parsed_spec.device.dla_core)
    d._set_allow_gpu_fallback(parsed_spec.device.allow_gpu_fallback)

    if parsed_spec.torch_fallback.enabled:
        raise RuntimeError(
            "Partial module compilation is not currently supported via the PyTorch TensorRT backend. If you need partial compilation, use trtorch.compile"
        )

    torch_fallback = torch.classes.tensorrt._TorchFallback()
    torch_fallback._set_enabled(parsed_spec.torch_fallback.enabled)
    torch_fallback._set_min_block_size(parsed_spec.torch_fallback.min_block_size)
    torch_fallback._set_forced_fallback_operators(parsed_spec.torch_fallback.forced_fallback_operators)
    torch_fallback._set_forced_fallback_modules(parsed_spec.torch_fallback.forced_fallback_modules)

    backend_spec._set_device(d)
    backend_spec._set_torch_fallback(torch_fallback)
    backend_spec._set_precisions([int(i) for i in parsed_spec.enabled_precisions])

    backend_spec._set_disable_tf32(parsed_spec.disable_tf32)
    backend_spec._set_refit(parsed_spec.refit)
    backend_spec._set_debug(parsed_spec.debug)
    backend_spec._set_refit(parsed_spec.refit)
    backend_spec._set_strict_types(parsed_spec.strict_types)
    backend_spec._set_capability(int(parsed_spec.capability))
    backend_spec._set_num_min_timing_iters(parsed_spec.num_min_timing_iters)
    backend_spec._set_num_avg_timing_iters(parsed_spec.num_avg_timing_iters)
    backend_spec._set_workspace_size(parsed_spec.workspace_size)
    backend_spec._set_max_batch_size(parsed_spec.max_batch_size)
    backend_spec._set_truncate_long_and_double(parsed_spec.truncate_long_and_double)
    backend_spec._set_ptq_calibrator(parsed_spec._get_calibrator_handle())

    return backend_spec
