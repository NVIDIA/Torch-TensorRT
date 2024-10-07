#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include "ATen/core/function_schema.h"
#include "NvInfer.h"
#include "core/runtime/Platform.h"
#include "core/runtime/RTDevice.h"
#include "core/runtime/TRTEngine.h"
#include "core/util/prelude.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

using EngineID = int64_t;
const std::string ABI_VERSION = "6";
extern bool MULTI_DEVICE_SAFE_MODE;
extern bool CUDAGRAPHS_MODE;

typedef enum {
  ABI_TARGET_IDX = 0,
  NAME_IDX,
  DEVICE_IDX,
  ENGINE_IDX,
  INPUT_BINDING_NAMES_IDX,
  OUTPUT_BINDING_NAMES_IDX,
  HW_COMPATIBLE_IDX,
  SERIALIZED_METADATA_IDX,
  TARGET_PLATFORM_IDX,
  SERIALIZATION_LEN, // NEVER USED FOR DATA, USED TO DETERMINE LENGTH OF SERIALIZED INFO
} SerializedInfoIndex;

std::string base64_encode(const std::string& in);
std::string base64_decode(const std::string& in);

c10::optional<RTDevice> get_most_compatible_device(
    const RTDevice& target_device,
    const RTDevice& curr_device = RTDevice(),
    bool hardware_compatible = false);
std::vector<RTDevice> find_compatible_devices(const RTDevice& target_device, bool hardware_compatible);

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TRTEngine> compiled_engine);

c10::intrusive_ptr<TRTEngine> setup_engine(
    const std::string& abi_version,
    const std::string& name,
    const std::string& serialized_device_info,
    const std::string& serialized_engine,
    const std::string& serialized_in_binding_names,
    const std::string& serialized_out_binding_names,
    const std::string& serialized_hardware_compatible,
    const std::string& serialized_metadata,
    const std::string& serialized_target_platform);

void multi_gpu_device_check();

bool get_multi_device_safe_mode();

void set_multi_device_safe_mode(bool multi_device_safe_mode);

bool get_cudagraphs_mode();

void set_cudagraphs_mode(bool cudagraphs_mode);

class DeviceList {
  using DeviceMap = std::unordered_map<int, RTDevice>;
  DeviceMap device_list;

 public:
  // Scans and updates the list of available CUDA devices
  DeviceList();

 public:
  void insert(int device_id, RTDevice cuda_device);
  RTDevice find(int device_id);
  DeviceMap get_devices();
  std::string dump_list();
};

DeviceList get_available_device_list();
const std::unordered_map<std::string, std::string>& get_dla_supported_SMs();

void set_rt_device(RTDevice& cuda_device);
// Gets the current active GPU (DLA will not show up through this)
RTDevice get_current_device();

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
