import os
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_tensorrt


class TestHardwareCompatibility(TestCase):
    def test_hw_compat_enabled(self):
        class SampleModel(torch.nn.Module):
            def forward(self, x):
                return torch.softmax((x * 7) @ x.T, dim=0)

        inputs = [torch.randn(5, 7).cuda()]

        # Validate that the hardware compatibility mode has been enabled
        optimized_model_hw_compat = torch_tensorrt.compile(
            SampleModel(),
            "dynamo",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            hardware_compatible=True,
            use_python_runtime=False,
        )

        self.assertTrue(optimized_model_hw_compat._run_on_acc_0.hardware_compatible)

        cpp_repr = optimized_model_hw_compat._run_on_acc_0.engine._properties.__repr__()

        self.assertIn("Hardware Compatibility: Enabled", cpp_repr)

        # Validate that the hardware compatibility mode has been disabled
        optimized_model_not_hw_compat = torch_tensorrt.compile(
            SampleModel(),
            "dynamo",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
            hardware_compatible=False,
            use_python_runtime=False,
        )

        self.assertFalse(
            optimized_model_not_hw_compat._run_on_acc_0.hardware_compatible
        )

        cpp_repr = (
            optimized_model_not_hw_compat._run_on_acc_0.engine._properties.__repr__()
        )

        self.assertIn("Hardware Compatibility: Disabled", cpp_repr)

    @unittest.skipIf(
        torch.ops.tensorrt.ABI_VERSION() != "5",
        "Detected incorrect ABI version, please update this test case",
    )
    def test_hw_compat_3080_build(self):
        inputs = [torch.randn(5, 7).cuda()]

        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        model = torch.jit.load("../../ts/models/hw_compat.ts").cuda()
        out = model(*inputs)
        self.assertTrue(
            isinstance(out, tuple)
            and len(out) == 1
            and isinstance(out[0], torch.Tensor),
            "Invalid output detected",
        )
        os.chdir(cwd)


if __name__ == "__main__":
    run_tests()
