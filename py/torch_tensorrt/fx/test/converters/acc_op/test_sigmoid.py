import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase


class TestSigmoid(AccTestCase):
    def test_sigmoid(self):
        class Sigmoid(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(Sigmoid(), inputs, expected_ops={acc_ops.sigmoid})


if __name__ == "__main__":
    run_tests()
