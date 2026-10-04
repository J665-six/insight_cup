#include "insight_cup/rknn.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace insight_cup {

namespace {

using json = nlohmann::json;

std::mutex& performanceOutputMutex() {
  static std::mutex mutex;
  return mutex;
}

void checkRknn(int result, const std::string& operation) {
  if (result != RKNN_SUCC) {
    throw std::runtime_error(operation + " failed with RKNN error " +
                             std::to_string(result));
  }
}

void validateModel(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("RKNN model not found: " + path.string());
  }
  const std::streamsize length = stream.tellg();
  if (length <= 0) {
    throw std::runtime_error("RKNN model is empty: " + path.string());
  }
}

std::size_t elementCount(const std::vector<int64_t>& shape) {
  if (shape.empty() ||
      std::any_of(shape.begin(), shape.end(),
                  [](int64_t value) { return value <= 0; })) {
    throw std::runtime_error("RKNN received an invalid input shape");
  }
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         [](std::size_t left, int64_t right) {
                           return left * static_cast<std::size_t>(right);
                         });
}

rknn_core_mask coreMask(int coreIndex) {
  switch (coreIndex) {
    case 0:
      return RKNN_NPU_CORE_0;
    case 1:
      return RKNN_NPU_CORE_1;
    case 2:
      return RKNN_NPU_CORE_2;
    default:
      throw std::runtime_error("RKNN core index must be 0, 1, or 2");
  }
}

std::vector<float> nchwToNhwc(const std::vector<float>& source,
                              const std::vector<int64_t>& shape) {
  if (shape.size() != 4) {
    throw std::runtime_error("RKNN input must be a four-dimensional NCHW tensor");
  }
  const std::size_t batches = static_cast<std::size_t>(shape[0]);
  const std::size_t channels = static_cast<std::size_t>(shape[1]);
  const std::size_t height = static_cast<std::size_t>(shape[2]);
  const std::size_t width = static_cast<std::size_t>(shape[3]);
  std::vector<float> destination(source.size());
  for (std::size_t batch = 0; batch < batches; ++batch) {
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t column = 0; column < width; ++column) {
        for (std::size_t channel = 0; channel < channels; ++channel) {
          const std::size_t sourceIndex =
              ((batch * channels + channel) * height + row) * width + column;
          const std::size_t destinationIndex =
              ((batch * height + row) * width + column) * channels + channel;
          destination[destinationIndex] = source[sourceIndex];
        }
      }
    }
  }
  return destination;
}

}  // namespace

RknnSession::RknnSession(const fs::path& modelPath, int coreIndex)
    : modelPath_(modelPath), coreIndex_(coreIndex) {
  if (const char* output = std::getenv("INSIGHT_CUP_RKNN_PERF_OUTPUT");
      output != nullptr && output[0] != '\0') {
    performanceOutput_ = fs::path(output);
  }
  validateModel(modelPath_);
  std::string nativeModelPath = modelPath_.string();
  try {
    checkRknn(rknn_init(&context_, nativeModelPath.data(), 0, 0, nullptr),
              "rknn_init(" + modelPath_.string() + ")");
    checkRknn(rknn_set_core_mask(context_, coreMask(coreIndex_)),
              "rknn_set_core_mask");

    rknn_sdk_version version{};
    checkRknn(rknn_query(context_, RKNN_QUERY_SDK_VERSION, &version,
                         sizeof(version)),
              "RKNN_QUERY_SDK_VERSION");
    apiVersion_ = version.api_version;
    driverVersion_ = version.drv_version;

    rknn_input_output_num count{};
    checkRknn(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &count,
                         sizeof(count)),
              "RKNN_QUERY_IN_OUT_NUM");
    if (count.n_input != 1 || count.n_output == 0) {
      throw std::runtime_error(
          "RKNN model must have one input and at least one output: " +
          modelPath_.string());
    }

    inputAttribute_.index = 0;
    checkRknn(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &inputAttribute_,
                         sizeof(inputAttribute_)),
              "RKNN_QUERY_INPUT_ATTR");
    if (inputAttribute_.n_dims != 4 ||
        inputAttribute_.fmt != RKNN_TENSOR_NHWC) {
      throw std::runtime_error(
          "RKNN model input must be a fixed NHWC tensor: " +
          modelPath_.string());
    }
    inputShape_ = {
        static_cast<int64_t>(inputAttribute_.dims[0]),
        static_cast<int64_t>(inputAttribute_.dims[3]),
        static_cast<int64_t>(inputAttribute_.dims[1]),
        static_cast<int64_t>(inputAttribute_.dims[2]),
    };

    outputAttributes_.resize(count.n_output);
    for (uint32_t index = 0; index < count.n_output; ++index) {
      outputAttributes_[index].index = index;
      checkRknn(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR,
                           &outputAttributes_[index],
                           sizeof(outputAttributes_[index])),
                "RKNN_QUERY_OUTPUT_ATTR");
    }
  } catch (...) {
    if (context_ != 0) {
      rknn_destroy(context_);
      context_ = 0;
    }
    throw;
  }
}

RknnSession::~RknnSession() {
  if (performanceOutput_ && performanceRunCount_ > 0) {
    try {
      const json payload = {
          {"schema", "insight_cup_rknn_performance.v1"},
          {"model", modelPath_.string()},
          {"core", coreIndex_},
          {"runs", performanceRunCount_},
          {"total_run_duration_us", performanceDurationUs_},
          {"min_run_duration_us", performanceMinUs_},
          {"max_run_duration_us", performanceMaxUs_},
          {"query_errors", performanceQueryErrors_},
      };
      std::lock_guard<std::mutex> outputLock(performanceOutputMutex());
      std::ofstream output(*performanceOutput_, std::ios::app);
      if (output) {
        output << payload.dump() << '\n';
      }
    } catch (...) {
      // Performance output is optional diagnostics and must not affect shutdown.
    }
  }
  if (context_ != 0) {
    rknn_destroy(context_);
  }
}

std::vector<TensorOutput> RknnSession::run(
    const std::vector<float>& input,
    const std::vector<int64_t>& shape) const {
  if (shape != inputShape_ || input.size() != elementCount(shape)) {
    throw std::runtime_error("RKNN input buffer does not match the model shape");
  }
  std::vector<float> formatted = nchwToNhwc(input, shape);
  std::lock_guard<std::mutex> lock(mutex_);

  rknn_input descriptor{};
  descriptor.index = 0;
  descriptor.buf = formatted.data();
  descriptor.size = static_cast<uint32_t>(formatted.size() * sizeof(float));
  descriptor.pass_through = 0;
  descriptor.type = RKNN_TENSOR_FLOAT32;
  descriptor.fmt = RKNN_TENSOR_NHWC;
  checkRknn(rknn_inputs_set(context_, 1, &descriptor), "rknn_inputs_set");
  checkRknn(rknn_run(context_, nullptr), "rknn_run");

  std::vector<rknn_output> nativeOutputs(outputAttributes_.size());
  for (std::size_t index = 0; index < nativeOutputs.size(); ++index) {
    nativeOutputs[index].index = static_cast<uint32_t>(index);
    nativeOutputs[index].want_float = 1;
    nativeOutputs[index].is_prealloc = 0;
  }
  checkRknn(rknn_outputs_get(context_,
                             static_cast<uint32_t>(nativeOutputs.size()),
                             nativeOutputs.data(), nullptr),
            "rknn_outputs_get");
  if (performanceOutput_) {
    rknn_perf_run performance{};
    const int queryResult =
        rknn_query(context_, RKNN_QUERY_PERF_RUN, &performance,
                   sizeof(performance));
    if (queryResult == RKNN_SUCC && performance.run_duration >= 0) {
      const auto duration =
          static_cast<std::uint64_t>(performance.run_duration);
      performanceDurationUs_ += duration;
      performanceMinUs_ = performanceRunCount_ == 0
                              ? duration
                              : std::min(performanceMinUs_, duration);
      performanceMaxUs_ = std::max(performanceMaxUs_, duration);
      ++performanceRunCount_;
    } else {
      ++performanceQueryErrors_;
    }
  }

  std::vector<TensorOutput> outputs;
  outputs.reserve(outputAttributes_.size());
  try {
    for (std::size_t index = 0; index < outputAttributes_.size(); ++index) {
      const rknn_tensor_attr& attribute = outputAttributes_[index];
      TensorOutput output;
      output.shape.reserve(attribute.n_dims);
      for (uint32_t dimension = 0; dimension < attribute.n_dims; ++dimension) {
        output.shape.push_back(attribute.dims[dimension]);
      }
      const float* data = static_cast<const float*>(nativeOutputs[index].buf);
      output.data.assign(data, data + attribute.n_elems);
      outputs.push_back(std::move(output));
    }
  } catch (...) {
    rknn_outputs_release(context_, static_cast<uint32_t>(nativeOutputs.size()),
                         nativeOutputs.data());
    throw;
  }
  checkRknn(rknn_outputs_release(context_,
                                 static_cast<uint32_t>(nativeOutputs.size()),
                                 nativeOutputs.data()),
            "rknn_outputs_release");
  return outputs;
}

std::optional<std::string> RknnSession::metadata(
    const std::string&) const {
  return std::nullopt;
}

}  // namespace insight_cup
