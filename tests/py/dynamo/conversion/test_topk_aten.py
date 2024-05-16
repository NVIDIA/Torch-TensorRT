import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestSortConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4), 1, 0, True, True),
            ((3850, 2), 3840, 0, False, True),
            # default dim:-1 largest:True, sorted:True
            ((3, 5, 12), 3),
        ]
    )
    def test_topk(self, input_shape, k, dim=-1, largest=True, sorted=True):
        class Topk(nn.Module):
            def forward(self, x):
                return torch.ops.aten.topk.default(x, k, dim, largest, sorted)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Topk(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
