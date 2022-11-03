#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/runtime/CUDADevice.h"
#include "core/runtime/TRTEngine.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

using EngineID = int64_t;
const std::string ABI_VERSION = "4";
typedef enum {
  ABI_TARGET_IDX = 0,
  NAME_IDX,
  DEVICE_IDX,
  ENGINE_IDX,
  INPUT_BINDING_NAMES_IDX,
  OUTPUT_BINDING_NAMES_IDX
} SerializedInfoIndex;

c10::optional<CUDADevice> get_most_compatible_device(const CUDADevice& target_device);
std::vector<CUDADevice> find_compatible_devices(const CUDADevice& target_device);

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);

class DeviceList {
  using DeviceMap = std::unordered_map<int, CUDADevice>;
  DeviceMap device_list;

 public:
  // Scans and updates the list of available CUDA devices
  DeviceList();

 public:
  void insert(int device_id, CUDADevice cuda_device);
  CUDADevice find(int device_id);
  DeviceMap get_devices();
  std::string dump_list();
};

DeviceList get_available_device_list();
const std::unordered_map<std::string, std::string>& get_dla_supported_SMs();

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
