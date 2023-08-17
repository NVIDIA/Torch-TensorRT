import operator
import unittest

import torch
import torch.nn as nn
from harness import DispatchTestCase
from torch.testing._internal.common_utils import run_tests


# TODO: Switch this test back to self.run_test once an implementation exists
# for a converter that returns a list, such as aten.split
@unittest.skip("Pending aten.split converter. Currently tested by E2E")
class TestGetItemConverter(DispatchTestCase):
    def test_getitem(self):
        class GetItem(nn.Module):
            def forward(self, x):
                lis = torch.split(x, 5)
                b = operator.getitem(lis, 0)
                c = operator.getitem(lis, 1)
                d = b + c
                return d

        inputs = [
            torch.randn((3, 3, 10)),
            torch.randn((3, 3, 10)),
            torch.randn((3, 3, 10)),
        ]
        self.run_test(
            GetItem(),
            inputs,
            expected_ops={operator.getitem},
            disable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
