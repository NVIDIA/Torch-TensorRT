from typing import Callable, Optional

from torch.fx.node import Node
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import generate_plugin
from torch_tensorrt.dynamo.conversion.plugins._generate_plugin_converter import (
    generate_plugin_converter,
)


def custom_op(
    op_name: str,
    capability_validator: Optional[Callable[[Node, CompilationSettings], bool]] = None,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    supports_dynamic_shapes: bool = False,
):
    generate_plugin(op_name)
    generate_plugin_converter(
        op_name, capability_validator, priority, supports_dynamic_shapes
    )
