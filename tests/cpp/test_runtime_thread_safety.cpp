#include <string>
#include <thread>
#include "gtest/gtest.h"
#include "tests/util/util.h"
#include "torch/script.h"
#include "trtorch/trtorch.h"

void run_infer(
    int thread_id,
    torch::jit::Module& mod,
    torch::jit::Module& trt_mod,
    const std::vector<torch::jit::IValue> inputs,
    const std::vector<torch::jit::IValue> inputs_trt,
    std::vector<torch::jit::IValue>& out_vec,
    std::vector<torch::jit::IValue>& trt_out_vec) {
  int count = 10;
  while (count-- > 0) {
    out_vec[thread_id] = mod.forward(inputs);
    trt_out_vec[thread_id] = trt_mod.forward(inputs_trt);
  }
}

TEST(CppAPITests, RuntimeThreadSafety) {
  std::string path = "tests/modules/resnet50_traced.jit.pt";
  torch::jit::Module mod;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    mod = torch::jit::load(path);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }
  mod.eval();
  mod.to(torch::kCUDA);

  torch::Tensor in_jit = at::randint(5, {1, 3, 224, 224}, torch::kCUDA).to(torch::kFloat);
  torch::Tensor in_trt = in_jit.clone().to(torch::kFloat);

  std::vector<torch::jit::IValue> inputs_jit;
  std::vector<torch::jit::IValue> inputs_trt;
  inputs_jit.push_back(in_jit.clone());
  inputs_trt.push_back(in_trt.clone());

  std::vector<trtorch::CompileSpec::Input> input_ranges;
  for (auto in : inputs_trt) {
    input_ranges.push_back({std::vector<int64_t>{1, 3, 224, 224},
                            std::vector<int64_t>{1, 3, 224, 224},
                            std::vector<int64_t>{16, 3, 224, 224},
                            torch::kFloat});
  }
  auto compile_settings = trtorch::CompileSpec(input_ranges);

  // FP32 execution
  compile_settings.enabled_precisions = {torch::kFloat};
  compile_settings.strict_types = true;
  auto trt_mod = trtorch::CompileGraph(mod, compile_settings);
  std::cout << "trtorch::CompileGraph" << std::endl;

  int num_threads = 10;
  std::vector<torch::jit::IValue> out_vec(num_threads), trt_out_vec(num_threads);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(
        run_infer,
        i,
        std::ref(mod),
        std::ref(trt_mod),
        inputs_jit,
        inputs_trt,
        std::ref(out_vec),
        std::ref(trt_out_vec)));
  }

  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  bool flag = true;
  for (int i = 0; i < num_threads; i++) {
    bool f = trtorch::tests::util::almostEqual(out_vec[i].toTensor(), trt_out_vec[i].toTensor(), 1e-2);
    flag = flag && f;
  }
  ASSERT_TRUE(flag);
}