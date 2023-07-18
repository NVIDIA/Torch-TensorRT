from packaging import version
from torch_tensorrt._util import sanitized_torch_version

if version.parse(sanitized_torch_version()) >= version.parse("2.1.dev"):
    from ._settings import *
    from .compile import compile
    from .aten_tracer import trace
