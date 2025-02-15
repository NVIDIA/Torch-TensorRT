import logging
from types import FunctionType
from typing import Tuple

import tensorrt_bindings.plugin as trtp
import torch
from sympy import lambdify
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

_LOGGER: logging.Logger = logging.getLogger(__name__)


def mksym(shape_env, value, source, dynamic_dim):
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        hint=value,
        source=source,
    )


def generate_plugin(plugin_name: str):
    namespace, name = plugin_name.split("::")

    # retrieve the corresponding torch operation using the passed in string
    torch_op = getattr(getattr(torch.ops, namespace), name)

    # helper function that generates the required signature based on the torch operation
    def generate_signature(torch_op):
        schema = torch_op._schemas[""]
        tensor_args = []
        arg_list = []

        args = []
        kwargs = []

        register_func_annotation = {}
        impl_func_annotation = {}

        for arg in schema.arguments:
            arg_list.append(arg.name)

            # TODO: Torch types need to be converted to python primitive types here
            # Some other types are not handled:
            # - torch._C.ListType.ofT(<type>)
            # - torch._C.TupleType.get()
            # - torch._C.DictType.get(<key_type>, <value_type>)
            # - torch._C.OptionalType.ofT(<type>)
            # - torch._C.DeviceObjType.get()
            # - torch._C.FunctionType.get()
            # - torch._C.ClassType

            if arg.type.isSubtypeOf(torch._C.TensorType.get()):
                tensor_args.append(arg)
                register_func_annotation[arg.name] = trtp.TensorDesc
                impl_func_annotation[arg.name] = trtp.Tensor
            elif arg.type.isSubtypeOf(torch._C.FloatType.get()):
                register_func_annotation[arg.name] = float
                impl_func_annotation[arg.name] = float
            elif arg.type.isSubtypeOf(torch._C.IntType.get()):
                register_func_annotation[arg.name] = int
                impl_func_annotation[arg.name] = int
            elif arg.type.isSubtypeOf(torch._C.Booltype.get()):
                register_func_annotation[arg.name] = bool
                impl_func_annotation[arg.name] = bool
            elif arg.type.isSubtypeOf(torch._C.Stringtype.get()):
                register_func_annotation[arg.name] = str
                impl_func_annotation[arg.name] = str
            else:
                raise ValueError("arg type is not handled")

            if arg.default_value is None:
                args.append(arg.name)
            else:
                kwargs.append(f"{arg.name} = {arg.default_value}")

        input_signature = ", ".join(arg_list)
        plugin_signature = f"def add_plugin_desc({input_signature}):"
        args_input = ", ".join(args)
        kwargs_input = ", ".join(kwargs)

        plugin_impl_arg_list = arg_list
        plugin_impl_arg_list.append("outputs")
        plugin_impl_arg_list.append("stream")
        plugin_impl_input = ", ".join(plugin_impl_arg_list)
        plugin_impl_signagture = f"def add_plugin_impl({plugin_impl_input}):"

        register_func_annotation["return"] = Tuple[trtp.TensorDesc]

        impl_func_annotation["outputs"] = Tuple[trtp.Tensor]
        impl_func_annotation["stream"] = int

        return (
            args_input,
            kwargs_input,
            plugin_signature,
            plugin_impl_signagture,
            register_func_annotation,
            impl_func_annotation,
        )

    # Use the helper function to get the required signatures
    (
        args_input,
        kwargs_input,
        plugin_signature,
        plugin_impl_signature,
        register_func_annotation,
        impl_func_annotation,
    ) = generate_signature(torch_op)

    def _generic_plugin_desc(*args, **kwargs) -> Tuple[trtp.TensorDesc]:
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        syms_args = []
        for arg in args:
            sample = {f"{i}": 5 for i in range(arg.ndim)}
            syms_arg = [
                mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
                for k, v in sample.items()
            ]
            syms_args.append(syms_arg)

        with FakeTensorMode() as fake_mode:
            fake_args = []
            for syms_arg in syms_args:
                fake_arg = torch.randn(syms_arg)
                fake_args.append(fake_arg)

            output = torch_op(*fake_args, **kwargs)

        # We assume that number of dimensions are the same in torch op
        shape_calc_fns = [None] * args[0].ndim
        for i in range(args[0].ndim):
            input_node_expr = [syms_arg[i].node.expr for syms_arg in syms_args]
            shape_calc_fns[i] = lambdify(
                tuple(input_node_expr), output.shape[i].node.expr, "math"
            )

        out_desc = args[0].like()
        for i in range(out_desc.ndim):
            input_shape_expr = [arg.shape_expr[i] for arg in args]
            out_desc.shape_expr[i] = shape_calc_fns[i](*input_shape_expr)

        return (out_desc,)

    codegen_plugin = f"""
{plugin_signature}
    return _generic_plugin_desc({args_input}, {kwargs_input})
    """

    _LOGGER.warning(f"Plugin registration function: \n{codegen_plugin}")

    plugin_code = compile(codegen_plugin, "<string>", "exec")

    globals()["_generic_plugin_desc"] = _generic_plugin_desc

    plugin = FunctionType(plugin_code.co_consts[0], globals(), "plugin")

    # Function annotation is required for dynamic function to work in TensorRT.Plugin
    plugin.__annotations__ = register_func_annotation

    trtp.register(plugin_name)(plugin)

    def _generic_plugin_impl(outputs, stream, *args, **kwargs):
        in_tensors = [torch.as_tensor(i, device="cuda") for i in args]

        dest_tensors = [torch.as_tensor(o, device="cuda") for o in outputs]

        stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(stream):
            out_tensors = torch_op(*in_tensors, **kwargs)
            [d.copy_(o) for (d, o) in zip(dest_tensors, out_tensors)]

    plugin_impl_func = f"""
{plugin_impl_signature}
    _generic_plugin_impl(outputs, stream, {args_input}, {kwargs_input})
    """

    _LOGGER.warning(f"Plugin implementation function: \n{plugin_impl_func}")

    plugin_impl_code = compile(plugin_impl_func, "<string>", "exec")

    globals()["_generic_plugin_impl"] = _generic_plugin_impl

    plugin_impl = FunctionType(plugin_impl_code.co_consts[0], globals(), "plugin_impl")

    plugin_impl.__annotations__ = impl_func_annotation

    trtp.impl(plugin_name)(plugin_impl)

    return plugin
