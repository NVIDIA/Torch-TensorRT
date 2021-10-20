#include <algorithm>

#include "torch/csrc/jit/api/module.h"

#include "core/compiler.h"
#include "core/util/prelude.h"

#include "trtorch/trtorch.h"

namespace trtorch {

std::ostream& operator<<(std::ostream& os, const CompileSpec::DataType& dtype) {
  switch (dtype) {
    case CompileSpec::DataType::kChar:
      os << "char";
      break;
    case CompileSpec::DataType::kHalf:
      os << "half";
      break;
    case CompileSpec::DataType::kInt:
      os << "int";
      break;
    case CompileSpec::DataType::kBool:
      os << "bool";
      break;
    case CompileSpec::DataType::kFloat:
      os << "float";
      break;
    case CompileSpec::DataType::kUnknown:
    default:
      os << "unknown";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const CompileSpec::TensorFormat& format) {
  switch (format) {
    case CompileSpec::TensorFormat::kChannelsLast:
      os << "channels last";
      break;
    case CompileSpec::TensorFormat::kContiguous:
      os << "contiguous";
      break;
    case CompileSpec::TensorFormat::kUnknown:
    default:
      os << "unknown";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const CompileSpec::Input& input) {
  auto vec_to_str = [](std::vector<int64_t> shape) -> std::string {
    std::stringstream ss;
    ss << '[';
    for (auto i : shape) {
      ss << i << ',';
    }
    ss << ']';
    return ss.str();
  };

  if (!input.input_is_dynamic) {
    os << "Input(shape: " << vec_to_str(input.shape) << ", dtype: " << input.dtype << ", format: " << input.format
       << ')';
  } else {
    os << "Input(shape: " << vec_to_str(input.shape) << ", min: " << vec_to_str(input.min_shape)
       << ", opt: " << vec_to_str(input.opt_shape) << ", max: " << vec_to_str(input.max_shape)
       << ", dtype: " << input.dtype << ", format: " << input.format << ')';
  }
  return os;
}

nvinfer1::DataType toTRTDataType(CompileSpec::DataType value) {
  switch (value) {
    case CompileSpec::DataType::kChar:
      return nvinfer1::DataType::kINT8;
    case CompileSpec::DataType::kHalf:
      return nvinfer1::DataType::kHALF;
    case CompileSpec::DataType::kInt:
      return nvinfer1::DataType::kINT32;
    case CompileSpec::DataType::kBool:
      return nvinfer1::DataType::kBOOL;
    case CompileSpec::DataType::kFloat:
    default:
      return nvinfer1::DataType::kFLOAT;
  }
}

nvinfer1::TensorFormat toTRTTensorFormat(CompileSpec::TensorFormat value) {
  TRTORCH_CHECK(!(value == CompileSpec::TensorFormat::kUnknown), "Tensor format is unknown");
  switch (value) {
    case CompileSpec::TensorFormat::kChannelsLast:
      return nvinfer1::TensorFormat::kHWC;
    case CompileSpec::TensorFormat::kContiguous:
    default:
      return nvinfer1::TensorFormat::kLINEAR;
  }
}

CompileSpec::DataType::DataType(c10::ScalarType t) {
  TRTORCH_CHECK(
      t == at::kHalf || t == at::kFloat || t == at::kChar || t == at::kInt || t == at::kBool,
      "Data type is unsupported (" << t << ")");
  switch (t) {
    case at::kHalf:
      value = DataType::kHalf;
      break;
    case at::kChar:
      value = DataType::kChar;
      break;
    case at::kInt:
      value = DataType::kInt;
      break;
    case at::kBool:
      value = DataType::kBool;
      break;
    case at::kFloat:
    default:
      value = DataType::kFloat;
      break;
  }
}

CompileSpec::TensorFormat::TensorFormat(at::MemoryFormat t) {
  TRTORCH_CHECK(
      t == at::MemoryFormat::Contiguous || t == at::MemoryFormat::ChannelsLast,
      "Tensor format is unsupported (" << t << ")");

  switch (t) {
    case at::MemoryFormat::ChannelsLast:
      value = TensorFormat::kChannelsLast;
    case at::MemoryFormat::Contiguous:
    default:
      value = TensorFormat::kContiguous;
      break;
  }
}

CompileSpec::Device::DeviceType::DeviceType(c10::DeviceType t) {
  TRTORCH_CHECK(t == at::kCUDA, "Device type when specified using torch device enum must be torch::kCUDA");
  value = DeviceType::kGPU;
}

CompileSpec::CompileSpec(std::vector<c10::ArrayRef<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    inputs.push_back(Input(in));
  }
}

CompileSpec::CompileSpec(std::vector<std::vector<int64_t>> fixed_sizes) {
  for (auto in : fixed_sizes) {
    inputs.push_back(Input(in));
  }
}

/* ====== DEFINE INPUTS CLASS MEMBERS ======*/
CompileSpec::Input::Input(std::vector<int64_t> shape, TensorFormat format) {
  this->opt_shape = shape;
  this->min_shape = shape;
  this->max_shape = shape;
  this->shape = shape;
  this->dtype = CompileSpec::DataType::kUnknown;
  this->format = format;
  this->input_is_dynamic = false;
}

CompileSpec::Input::Input(std::vector<int64_t> shape, DataType dtype, TensorFormat format) {
  this->opt_shape = shape;
  this->min_shape = shape;
  this->max_shape = shape;
  this->shape = shape;
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = false;
}

CompileSpec::Input::Input(c10::IntArrayRef shape, TensorFormat format) {
  this->opt_shape = core::util::toVec(shape);
  this->min_shape = core::util::toVec(shape);
  this->max_shape = core::util::toVec(shape);
  this->shape = core::util::toVec(shape);
  this->dtype = CompileSpec::DataType::kUnknown;
  this->format = format;
  this->input_is_dynamic = false;
}

CompileSpec::Input::Input(c10::IntArrayRef shape, DataType dtype, TensorFormat format) {
  this->opt_shape = core::util::toVec(shape);
  this->min_shape = core::util::toVec(shape);
  this->max_shape = core::util::toVec(shape);
  this->shape = core::util::toVec(shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = false;
}

CompileSpec::Input::Input(
    std::vector<int64_t> min_shape,
    std::vector<int64_t> opt_shape,
    std::vector<int64_t> max_shape,
    TensorFormat format) {
  this->opt_shape = opt_shape;
  this->min_shape = min_shape;
  this->max_shape = max_shape;
  this->shape = core::util::toVec(core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = CompileSpec::DataType::kUnknown;
  this->format = format;
  this->input_is_dynamic = true;
}

CompileSpec::Input::Input(
    std::vector<int64_t> min_shape,
    std::vector<int64_t> opt_shape,
    std::vector<int64_t> max_shape,
    DataType dtype,
    TensorFormat format) {
  this->opt_shape = opt_shape;
  this->min_shape = min_shape;
  this->max_shape = max_shape;
  this->shape = core::util::toVec(core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = true;
}

CompileSpec::Input::Input(
    c10::IntArrayRef min_shape,
    c10::IntArrayRef opt_shape,
    c10::IntArrayRef max_shape,
    TensorFormat format) {
  this->opt_shape = core::util::toVec(opt_shape);
  this->min_shape = core::util::toVec(min_shape);
  this->max_shape = core::util::toVec(max_shape);
  this->shape = core::util::toVec(core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = CompileSpec::DataType::kUnknown;
  this->format = format;
  this->input_is_dynamic = true;
}

CompileSpec::Input::Input(
    c10::IntArrayRef min_shape,
    c10::IntArrayRef opt_shape,
    c10::IntArrayRef max_shape,
    DataType dtype,
    TensorFormat format) {
  this->opt_shape = core::util::toVec(opt_shape);
  this->min_shape = core::util::toVec(min_shape);
  this->max_shape = core::util::toVec(max_shape);
  this->shape = core::util::toVec(core::ir::Input(this->min_shape, this->opt_shape, this->max_shape).input_shape);
  this->dtype = dtype;
  this->format = format;
  this->input_is_dynamic = true;
}

CompileSpec::Input::Input(at::Tensor tensor) {
  this->opt_shape = tensor.sizes().vec();
  this->min_shape = tensor.sizes().vec();
  this->max_shape = tensor.sizes().vec();
  this->shape = tensor.sizes().vec();
  this->dtype = tensor.scalar_type();
  TRTORCH_ASSERT(
      tensor.is_contiguous(at::MemoryFormat::ChannelsLast) || tensor.is_contiguous(at::MemoryFormat::Contiguous),
      "Tensor does not have a supported contiguous memory format, supported formats are contiguous or channel_last");
  at::MemoryFormat frmt;
  if (tensor.is_contiguous(at::MemoryFormat::Contiguous)) {
    frmt = at::MemoryFormat::Contiguous;
  } else {
    frmt = at::MemoryFormat::ChannelsLast;
  }
  this->format = frmt;
  this->input_is_dynamic = false;
}

/* ==========================================*/

core::ir::Input to_internal_input(CompileSpec::Input& i) {
  return core::ir::Input(
      i.min_shape,
      i.opt_shape,
      i.max_shape,
      toTRTDataType(i.dtype),
      toTRTTensorFormat(i.format),
      !(i.dtype == CompileSpec::DataType::kUnknown));
}

std::vector<core::ir::Input> to_vec_internal_inputs(std::vector<CompileSpec::Input>& external) {
  std::vector<core::ir::Input> internal;
  for (auto range : external) {
    internal.push_back(to_internal_input(range));
  }
  return internal;
}

core::runtime::CudaDevice to_internal_cuda_device(CompileSpec::Device device) {
  auto device_type = nvinfer1::DeviceType::kGPU;
  switch (device.device_type) {
    case CompileSpec::Device::DeviceType::kDLA:
      device_type = nvinfer1::DeviceType::kDLA;
      break;
    case CompileSpec::Device::DeviceType::kGPU:
    default:
      device_type = nvinfer1::DeviceType::kGPU;
  }
  return core::runtime::CudaDevice(device.gpu_id, device_type);
}

core::CompileSpec to_internal_compile_spec(CompileSpec external) {
  core::CompileSpec internal(to_vec_internal_inputs(external.inputs));

  for (auto p : external.enabled_precisions) {
    internal.convert_info.engine_settings.enabled_precisions.insert(toTRTDataType(p));
  }

  internal.convert_info.engine_settings.sparse_weights = external.sparse_weights;
  internal.convert_info.engine_settings.disable_tf32 = external.disable_tf32;
  internal.convert_info.engine_settings.refit = external.refit;
  internal.convert_info.engine_settings.debug = external.debug;
  internal.convert_info.engine_settings.truncate_long_and_double = external.truncate_long_and_double;
  internal.convert_info.engine_settings.strict_types = external.strict_types;
  internal.convert_info.engine_settings.device.allow_gpu_fallback = external.device.allow_gpu_fallback;
  internal.convert_info.engine_settings.max_batch_size = external.max_batch_size;

  TRTORCH_CHECK(
      !(external.require_full_compilation && (external.torch_executed_ops.size() > 0)),
      "require_full_compilation is enabled however the list of ops to run in torch is not empty (Found "
          << external.torch_executed_ops.size() << " ops)");

  TRTORCH_CHECK(
      !(external.require_full_compilation && (external.torch_executed_modules.size() > 0)),
      "require_full_compilation is enabled however the list of modules to run in torch is not empty (Found "
          << external.torch_executed_modules.size() << " modules)");

  internal.partition_info.enabled = external.require_full_compilation;
  internal.partition_info.min_block_size = external.min_block_size;
  internal.partition_info.forced_fallback_operators = std::move(external.torch_executed_ops);
  internal.lower_info.forced_fallback_modules = std::move(external.torch_executed_modules);

  switch (external.device.device_type) {
    case CompileSpec::Device::DeviceType::kDLA:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kDLA;
      break;
    case CompileSpec::Device::DeviceType::kGPU:
    default:
      internal.convert_info.engine_settings.device.device_type = nvinfer1::DeviceType::kGPU;
  }

  switch (external.capability) {
    case CompileSpec::EngineCapability::kSAFETY:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_SAFETY;
      break;
    case CompileSpec::EngineCapability::kDLA_STANDALONE:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_DLA_STANDALONE;
      break;
    case CompileSpec::EngineCapability::kSTANDARD:
    default:
      internal.convert_info.engine_settings.capability = TRT_ENGINE_CAPABILITY_STANDARD;
  }

  internal.convert_info.engine_settings.device.gpu_id = external.device.gpu_id;
  internal.convert_info.engine_settings.device.dla_core = external.device.dla_core;
  internal.convert_info.engine_settings.num_min_timing_iters = external.num_min_timing_iters;
  internal.convert_info.engine_settings.num_avg_timing_iters = external.num_avg_timing_iters;
  internal.convert_info.engine_settings.workspace_size = external.workspace_size;

  if (internal.convert_info.engine_settings.enabled_precisions.find(nvinfer1::DataType::kINT8) !=
      internal.convert_info.engine_settings.enabled_precisions.end()) {
    if (external.ptq_calibrator) {
      internal.convert_info.engine_settings.calibrator = external.ptq_calibrator;
    } else {
      internal.lower_info.unfreeze_module = true;
      internal.lower_info.disable_cse = true;
      internal.convert_info.engine_settings.calibrator = nullptr;
    }
  } else {
    internal.convert_info.engine_settings.calibrator = nullptr;
  }

  return internal;
}

} // namespace trtorch
