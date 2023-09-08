import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLogicalXorConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_logical_xor(self, _, shape):
        class logical_xor(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.logical_xor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            logical_xor(),
            inputs,
            expected_ops={torch.ops.aten.logical_xor.default},
        )


if __name__ == "__main__":
    run_tests()
