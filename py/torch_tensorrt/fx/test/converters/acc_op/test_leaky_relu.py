import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestLeakyReLUConverter(AccTestCase):
    def test_leaky_relu(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.leaky_relu(x, negative_slope=0.05)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.leaky_relu})

    def test_leaky_relu_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.leaky_relu(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs, expected_ops={acc_ops.leaky_relu})

    def test_leaky_relu_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return nn.functional.leaky_relu(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 3, 3), (3, 3, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(TestModule(), input_specs, expected_ops={acc_ops.leaky_relu})


if __name__ == "__main__":
    run_tests()
