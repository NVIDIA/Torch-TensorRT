import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestRSqrtConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim_alpha", (2, 1), 2),
            ("3d_dim_alpha", (2, 1, 2), 2),
        ]
    )
    def test_rsqrt(self, _, x, alpha):
        class rsqrt(nn.Module):
            def forward(self, input):
                return torch.ops.aten.rsqrt.default(input)

        inputs = [torch.randn(x) + 1]
        self.run_test(
            rsqrt(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                "2d_dim_dtype_half",
                (1, 1),
                (2, 2),
                (4, 4),
                torch.half,
                torch.half,
            ),
            (
                "3d_dim_dtype_float",
                (1, 1, 1),
                (1, 2, 3),
                (3, 3, 3),
                torch.float,
                torch.float,
            ),
        ]
    )
    def test_dynamic_shape_rsqrt(
        self, _, min_shape, opt_shape, max_shape, type, output_type
    ):
        class rsqrt(nn.Module):
            def forward(self, input):
                return torch.ops.aten.rsqrt.default(input)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(
            rsqrt(), input_specs, output_dtypes=[output_type]
        )


if __name__ == "__main__":
    run_tests()
