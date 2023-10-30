import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestCumsumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1,), 0),
            ((2,), 0),
            ((3,), -1),
        ]
    )
    def test_cumsum_1D(self, shape, dim):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dim)

        inputs = [torch.randn(shape)]
        self.run_test(
            Cumsum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 1), 0),
            ((3, 1), 1),
            ((2, 3), -1),
            ((2, 3), -2),
        ]
    )
    def test_cumsum_2D(self, shape, dims):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Cumsum(),
            inputs,
        )

    @parameterized.expand(
        [
            ((4, 2, 3), 0),
            ((4, 2, 3), 1),
            ((1, 2, 3), 2),
            ((1, 2, 3), -1),
            ((1, 2, 3), -2),
        ]
    )
    def test_cumsum_3D(self, shape, dims):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Cumsum(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
