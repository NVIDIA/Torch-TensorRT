# type: ignore
import os
import shutil
import unittest
from typing import Optional

import pytest
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from torch.testing._internal.common_utils import TestCase
from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


class MyEngineCache(BaseEngineCache):
    def __init__(
        self,
        engine_cache_dir: str,
    ) -> None:
        self.engine_cache_dir = engine_cache_dir
        if not os.path.exists(self.engine_cache_dir):
            os.makedirs(self.engine_cache_dir, exist_ok=True)

        self.hashes = {}

    def save(
        self,
        hash: str,
        blob: bytes,
        prefix: str = "blob",
    ):
        if not os.path.exists(self.engine_cache_dir):
            os.makedirs(self.engine_cache_dir, exist_ok=True)

        path = os.path.join(
            self.engine_cache_dir,
            f"{prefix}_{hash}.bin",
        )
        with open(path, "wb") as f:
            f.write(blob)

        self.hashes[hash] = 0

    def load(self, hash: str, prefix: str = "blob") -> Optional[bytes]:
        path = os.path.join(self.engine_cache_dir, f"{prefix}_{hash}.bin")
        if os.path.exists(path):
            with open(path, "rb") as f:
                blob = f.read()
            self.hashes[hash] += 1
            return blob
        return None


class TestHashFunction(TestCase):

    def test_reexport_is_equal(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        batch = torch.export.Dim("batch", min=1, max=200)

        exp_program1 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs1 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings1 = CompilationSettings(
            cache_built_engines=True, reuse_cached_engines=True
        )
        hash1 = BaseEngineCache.get_hash(exp_program1.module(), input_specs1, settings1)

        exp_program2 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs2 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings2 = CompilationSettings(
            cache_built_engines=True, reuse_cached_engines=True
        )
        hash2 = BaseEngineCache.get_hash(exp_program2.module(), input_specs2, settings2)

        self.assertEqual(hash1, hash2)

    def test_input_shape_change_is_not_equal(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        batch = torch.export.Dim("batch", min=1, max=200)

        exp_program1 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs1 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings1 = CompilationSettings(
            cache_built_engines=True, reuse_cached_engines=True
        )
        hash1 = BaseEngineCache.get_hash(exp_program1.module(), input_specs1, settings1)

        exp_program2 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs2 = (
            torch_trt.Input(
                min_shape=(1, 3, 300, 300),
                opt_shape=(100, 3, 300, 300),
                max_shape=(200, 3, 300, 300),
            ),
        )
        settings2 = CompilationSettings(
            cache_built_engines=True, reuse_cached_engines=True
        )
        hash2 = BaseEngineCache.get_hash(exp_program2.module(), input_specs2, settings2)

        self.assertNotEqual(hash1, hash2)

    def test_engine_settings_is_not_equal(self):
        pyt_model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        batch = torch.export.Dim("batch", min=1, max=200)

        exp_program1 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs1 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings1 = CompilationSettings(
            cache_built_engines=True,
            reuse_cached_engines=True,
            enabled_precisions={torch.float32},
        )
        hash1 = BaseEngineCache.get_hash(exp_program1.module(), input_specs1, settings1)

        exp_program2 = torch.export.export(
            pyt_model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )
        input_specs2 = (
            torch_trt.Input(
                min_shape=(1, 3, 224, 224),
                opt_shape=(100, 3, 224, 224),
                max_shape=(200, 3, 224, 224),
            ),
        )
        settings2 = CompilationSettings(
            cache_built_engines=True,
            reuse_cached_engines=True,
            enabled_precisions={torch.float32, torch.float16},
        )
        hash2 = BaseEngineCache.get_hash(exp_program2.module(), input_specs2, settings2)

        self.assertNotEqual(hash1, hash2)


class TestEngineCache(TestCase):

    @pytest.mark.xfail
    def test_dynamo_compile_with_default_disk_engine_cache(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")
        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        # Mark the dim0 of inputs as dynamic
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )

        engine_cache_dir = "/tmp/test_torch_dynamo_with_default_disk_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        def remove_timing_cache(path=TIMING_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)

        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((128, 3, 224, 224)).to("cuda")]
        results = []
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(3):
            # remove timing cache and reset dynamo for engine caching messurement
            remove_timing_cache()
            torch._dynamo.reset()
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            torch.cuda.synchronize()
            start.record()
            trt_gm = torch_trt.dynamo.compile(
                exp_program,
                tuple(inputs),
                use_python_runtime=True,
                enabled_precisions={torch.float},
                debug=False,
                min_block_size=1,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                engine_cache_dir=engine_cache_dir,
            )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            results.append(trt_gm(*inputs))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        assertions.assertTrue(
            times[0] > times[2],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {times[0]} ms, time taken with engine caching: {times[2]} ms",
        )

    def test_dynamo_compile_with_custom_engine_cache(self):
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/test_torch_dynamo_with_custom_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)

        example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
        # Mark the dim0 of inputs as dynamic
        batch = torch.export.Dim("batch", min=1, max=200)
        exp_program = torch.export.export(
            model, args=example_inputs, dynamic_shapes={"x": {0: batch}}
        )

        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((128, 3, 224, 224)).to("cuda")]
        results = []
        for i in range(3):
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            trt_gm = torch_trt.dynamo.compile(
                exp_program,
                tuple(inputs),
                use_python_runtime=True,
                enabled_precisions={torch.float},
                debug=False,
                min_block_size=1,
                cache_built_engines=cache_built_engines,
                reuse_cached_engines=reuse_cached_engines,
                custom_engine_cache=custom_engine_cache,
            )
            results.append(trt_gm(*inputs))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        [
            assertions.assertTrue(
                count == 1,
                f"cache was not hit exactly once for entry ({h}, hit: {count})",
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    def test_dynamo_compile_change_input_shape(self):
        """Runs compilation 3 times, the cache should miss each time"""
        model = models.resnet18(pretrained=True).eval().to("cuda")
        # Mark the dim0 of inputs as dynamic

        engine_cache_dir = "/tmp/test_torch_dynamo_with_custom_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)

        for i in range(3):
            inputs = (torch.rand((4 * (i + 1), 3, 224, 224)).to("cuda"),)
            trt_gm = torch_trt.dynamo.compile(
                torch.export.export(model, args=inputs),
                inputs=inputs,
                use_python_runtime=False,
                enabled_precisions={torch.float},
                debug=False,
                min_block_size=1,
                cache_built_engines=True,
                reuse_cached_engines=True,
            )

        [
            assertions.assertTrue(
                count == 0, f"Unintended cache hit for entry ({h}, hit: {count})"
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    @pytest.mark.xfail
    def test_torch_compile_with_default_disk_engine_cache(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/test_torch_compile_with_default_disk_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        def remove_timing_cache(path=TIMING_CACHE_PATH):
            if os.path.exists(path):
                os.remove(path)

        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]
        results = []
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(3):
            # remove timing cache and reset dynamo for engine caching measurement
            remove_timing_cache()
            torch._dynamo.reset()
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            torch.cuda.synchronize()
            start.record()
            compiled_model = torch.compile(
                model,
                backend="tensorrt",
                options={
                    "use_python_runtime": False,
                    "enabled_precisions": {torch.float},
                    "debug": False,
                    "min_block_size": 1,
                    "cache_built_engines": cache_built_engines,
                    "reuse_cached_engines": reuse_cached_engines,
                    "engine_cache_dir": engine_cache_dir,
                    "engine_cache_size": 1 << 30,  # 1GB
                },
            )
            results.append(compiled_model(*inputs))  # trigger the compilation
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        assertions.assertTrue(
            times[0] > times[2],
            msg=f"Engine caching didn't speed up the compilation. Time taken without engine caching: {times[0]} ms, time taken with engine caching: {times[2]} ms",
        )

    def test_torch_compile_with_custom_engine_cache(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")

        engine_cache_dir = "/tmp/test_torch_compile_with_custom_engine_cache"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)
        # The 1st iteration is to measure the compilation time without engine caching
        # The 2nd and 3rd iterations are to measure the compilation time with engine caching.
        # Since the 2nd iteration needs to compile and save the engine, it will be slower than the 1st iteration.
        # The 3rd iteration should be faster than the 1st iteration because it loads the cached engine.
        inputs = [torch.rand((100, 3, 224, 224)).to("cuda")]
        results = []
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(3):
            if i == 0:
                cache_built_engines = False
                reuse_cached_engines = False
            else:
                cache_built_engines = True
                reuse_cached_engines = True

            start.record()
            compiled_model = torch.compile(
                model,
                backend="tensorrt",
                options={
                    "use_python_runtime": False,
                    "enabled_precisions": {torch.float},
                    "debug": False,
                    "min_block_size": 1,
                    "cache_built_engines": cache_built_engines,
                    "reuse_cached_engines": reuse_cached_engines,
                    "custom_engine_cache": custom_engine_cache,
                },
            )
            results.append(compiled_model(*inputs))  # trigger the compilation
            end.record()
            torch.cuda.synchronize()
            torch._dynamo.reset()
            times.append(start.elapsed_time(end))

        cos_sim = cosine_similarity(results[0], results[1])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[0] doesn't match with results[1]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        cos_sim = cosine_similarity(results[1], results[2])
        assertions.assertTrue(
            cos_sim > COSINE_THRESHOLD,
            msg=f"results[1] doesn't match with results[2]. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
        )

        [
            assertions.assertTrue(
                count == 1,
                f"cache was not hit exactly once for entry ({h}, hit: {count})",
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    def test_torch_trt_compile_change_input_shape(self):
        # Custom Engine Cache
        model = models.resnet18(pretrained=True).eval().to("cuda")
        engine_cache_dir = "/tmp/test_torch_trt_compile_change_input_shape"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)
        for i in range(3):
            inputs = [torch.rand((4 * (i + 1), 3, 224, 224)).to("cuda")]
            compiled_model = torch_trt.compile(
                model,
                inputs=inputs,
                **{
                    "use_python_runtime": True,
                    "enabled_precisions": {torch.float},
                    "debug": False,
                    "min_block_size": 1,
                    "cache_built_engines": True,
                    "reuse_cached_engines": True,
                    "custom_engine_cache": custom_engine_cache,
                },
            )
            compiled_model(*inputs)
        [
            assertions.assertTrue(
                count == 0, f"Unintended cache hit for entry ({h}, hit: {count})"
            )
            for h, count in custom_engine_cache.hashes.items()
        ]

    def test_torch_compile_graph_break(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                x = x + x
                x = x + x
                x = torch.ops.aten.relu.default(x)
                x = x + x
                x = x + x
                x = torch.ops.aten.relu.default(x)
                x = x + x
                x = x + x
                return x

        model = MyModel().eval().cuda()
        engine_cache_dir = "/tmp/test_torch_compile_graph_break"
        if os.path.exists(engine_cache_dir):
            shutil.rmtree(engine_cache_dir)

        custom_engine_cache = MyEngineCache(engine_cache_dir)
        inputs = [torch.rand((3, 3, 224, 224)).to("cuda")]
        for i in range(3):
            compiled_model = torch.compile(
                model,
                backend="tensorrt",
                options={
                    "use_python_runtime": True,
                    "enabled_precisions": {torch.float},
                    "debug": False,
                    "min_block_size": 1,
                    "cache_built_engines": True,
                    "reuse_cached_engines": True,
                    "custom_engine_cache": custom_engine_cache,
                    "torch_executed_ops": {"torch.ops.aten.relu.default"},
                },
            )
            compiled_model(*inputs)

        [
            assertions.assertTrue(
                count == 2,
                f"cache was not hit exactly twice for entry ({h}, hit: {count})",
            )
            for h, count in custom_engine_cache.hashes.items()
        ]
