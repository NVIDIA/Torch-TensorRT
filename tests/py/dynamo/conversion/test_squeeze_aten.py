import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSqueezeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim", (0), (2, 1)),
            ("3d_one_dim", (0), (2, 2, 1)),
        ]
    )
    def test_squeeze_single_dim(self, _, dim, init_size):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.ops.aten.squeeze.dim(x, dim)

        inputs = [torch.randn(*init_size)]
        self.run_test(
            Squeeze(),
            inputs,
        )

    @parameterized.expand(
        [
            ("3d_two_dim", (0, 1), (2, 1, 1)),
            ("4d_dim", (0, 1, 2), (2, 2, 1, 1)),
        ]
    )
    def test_squeeze_multi_dims(self, _, dim, init_size):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.ops.aten.squeeze.dims(x, dim)

        inputs = [torch.randn(*init_size)]
        self.run_test(
            Squeeze(),
            inputs,
        )


class TestSqueezeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "2d_dim",
                (1),
                (1, 1),
                (1, 1),
                (3, 1),
                torch.half,
                torch.half,
            ),
            (
                "3d_one_dim",
                (1),
                (1, 2, 1),
                (1, 2, 1),
                (3, 2, 1),
                torch.float,
                torch.float,
            ),
            (
                "3d_one_dim_dynamic",
                (0),
                (1, 2, 1),
                (1, 2, 1),
                (3, 2, 1),
                torch.float,
                torch.float,
            ),
        ]
    )
    def test_squeeze(self, _, dim, min_shape, opt_shape, max_shape, type, output_type):
        class Squeeze(nn.Module):
            def forward(self, x):
                return torch.ops.aten.squeeze.dim(x, dim)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Squeeze(),
            input_specs,
            output_dtypes=[output_type],
        )


if __name__ == "__main__":
    run_tests()
