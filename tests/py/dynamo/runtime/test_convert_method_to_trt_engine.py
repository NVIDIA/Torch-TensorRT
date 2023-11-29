import unittest

import tensorrt as trt
import torch
import torch_tensorrt
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity


class TestConvertMethodToTrtEngine(unittest.TestCase):
    def test_convert_module(self):
        class Test(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        # Prepare the input data
        input_data_0, input_data_1 = torch.randn((2, 4)), torch.randn((2, 4))

        # Create a model
        model = Test()
        symbolic_traced_gm = torch.fx.symbolic_trace(model)

        # Convert to TensorRT engine
        trt_engine_str = torch_tensorrt.dynamo.convert_method_to_trt_engine(
            symbolic_traced_gm, "forward", inputs=[input_data_0, input_data_1]
        )

        # Deserialize the TensorRT engine
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(trt_engine_str)

        # Inference on TRT Engine
        py_trt_module = PythonTorchTensorRTModule(engine, ["a", "b"], ["output0"])
        trt_output = py_trt_module(input_data_0, input_data_1).cpu()

        # Inference on PyTorch model
        model_output = model(input_data_0, input_data_1)

        cos_sim = cosine_similarity(model_output, trt_output)
        self.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"TRT outputs don't match with the original model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
