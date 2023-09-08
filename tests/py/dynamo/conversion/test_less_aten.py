import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLessConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_less_tensor(self, _, shape):
        class less(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return lhs_val < rhs_val

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            less(),
            inputs,
            expected_ops={torch.ops.aten.lt.Tensor},
            output_dtypes=[torch.bool],
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_less_tensor_scalar(self, _, shape, scalar):
        class less(nn.Module):
            def forward(self, lhs_val):
                return lhs_val < torch.tensor(scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            less(),
            inputs,
            expected_ops={torch.ops.aten.lt.Tensor},
            output_dtypes=[torch.bool],
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_less_scalar(self, _, shape, scalar):
        class less(nn.Module):
            def forward(self, lhs_val):
                return lhs_val < scalar

        inputs = [torch.randn(shape)]
        self.run_test(
            less(),
            inputs,
            expected_ops={torch.ops.aten.lt.Scalar},
            output_dtypes=[torch.bool],
        )


if __name__ == "__main__":
    run_tests()
