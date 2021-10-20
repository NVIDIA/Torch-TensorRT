# Tests

Right now there are two types of tests. Converter level tests and Module level tests.

The goal of Converter tests are to tests individual converters againsts specific subgraphs. The current tests in `core/conveters` are good examples on how to write these tests. In general every converter should have at least 1 test. More may be required if the operation has switches that change the behavior of the op.

Module tests are designed to test the compiler against common network architectures and verify the integration of converters together into a single engine.

In addition to the above, we have lowering tests (`//core/lowering`) which test the functionality of lowering passes and partitioning tests (`//core/partitioning `) which test different cases of torch fallback on test networks.

You can run the whole test suite with bazel. But be aware you may exhaust GPU memory (this may be seen as a cuDNN initialization error) running them naively, you therefore may need to limit the number of concurrent tests. Also because the inputs to tests are random it may make sense to run tests a few times.

Here are some settings that we usually test with:

```
bazel test //tests --compilation_mode=dbg --test_output=errors --jobs=4 --runs_per_test=5
```

`--runs_per_test` is optional and can be performed to check if numerical issues in outputs persist across multiple runs.

`--jobs=4` is useful and is sometimes required to prevent too many processes to use GPU memory and cause CUDA out of memory issues.

### Testing using pre-built TRTorch library

Currently, the default strategy when we run all the tests (`bazel test //tests`) is to build the testing scripts along with the full TRTorch library (`libtrtorch.so`) from scratch. This can lead to increased testing time and might not be needed incase you already have a pre-built TRTorch library that you want to link against.

In order to **not** build the entire TRTorch library and only build the test scripts, please use the following command.

```
bazel test //tests  --compilation_mode=dbg --test_output=summary --define trtorch_src=pre_built --jobs 2
```

 The flag `--define trtorch_src=pre_built` signals bazel to use pre-compiled library as an external dependency for tests. The pre-compiled library path is defined as a `local_repository` rule in root `WORKSPACE` file (`https://github.com/NVIDIA/TRTorch/blob/master/WORKSPACE`).

```
# External dependency for trtorch if you already have precompiled binaries.
# This is currently used in pytorch NGC container CI testing.
local_repository(
    name = "trtorch",
    path = "/opt/pytorch/trtorch"
)
```
