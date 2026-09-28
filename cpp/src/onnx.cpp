#include "insight_cup/onnx.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace insight_cup {

namespace {

Ort::Env& ortEnvironment() {
  static Ort::Env environment(ORT_LOGGING_LEVEL_ERROR, "insight-cup-cpp");
  return environment;
}

std::size_t elementCount(const std::vector<int64_t>& shape) {
  if (shape.empty() ||
      std::any_of(shape.begin(), shape.end(), [](int64_t value) { return value < 0; })) {
    throw std::runtime_error("Tensor has an unresolved dynamic shape");
  }
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         [](std::size_t left, int64_t right) {
                           return left * static_cast<std::size_t>(right);
                         });
}

}  // namespace

OnnxSession::OnnxSession(const fs::path& modelPath, int cpuThreads,
                         const std::string& provider) {
  if (!fs::is_regular_file(modelPath)) {
    throw std::runtime_error("ONNX model not found: " + modelPath.string());
  }
  if (provider != "auto" && provider != "cpu") {
    throw std::runtime_error(
        "This C++ ONNX Runtime package provides CPUExecutionProvider only; "
        "use --provider cpu or --provider auto");
  }

  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(std::max(1, cpuThreads));
  options.SetInterOpNumThreads(1);
  options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.AddConfigEntry("session.intra_op.allow_spinning", "0");
  options.AddConfigEntry("session.inter_op.allow_spinning", "0");
  session_ = std::make_unique<Ort::Session>(ortEnvironment(), modelPath.c_str(), options);

  Ort::AllocatorWithDefaultOptions allocator;
  const std::size_t inputCount = session_->GetInputCount();
  const std::size_t outputCount = session_->GetOutputCount();
  if (inputCount != 1 || outputCount == 0) {
    throw std::runtime_error("ONNX model must have one input and at least one output: " +
                             modelPath.string());
  }
  for (std::size_t index = 0; index < inputCount; ++index) {
    auto name = session_->GetInputNameAllocated(index, allocator);
    inputNames_.emplace_back(name.get());
  }
  for (std::size_t index = 0; index < outputCount; ++index) {
    auto name = session_->GetOutputNameAllocated(index, allocator);
    outputNames_.emplace_back(name.get());
  }
  for (const auto& name : inputNames_) {
    inputNamePointers_.push_back(name.c_str());
  }
  for (const auto& name : outputNames_) {
    outputNamePointers_.push_back(name.c_str());
  }
  inputShape_ = session_->GetInputTypeInfo(0)
                    .GetTensorTypeAndShapeInfo()
                    .GetShape();
}

std::vector<TensorOutput> OnnxSession::run(
    const std::vector<float>& input, const std::vector<int64_t>& shape) const {
  if (input.size() != elementCount(shape)) {
    throw std::runtime_error("ONNX input buffer size does not match its shape");
  }
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value tensor = Ort::Value::CreateTensor<float>(
      memory, const_cast<float*>(input.data()), input.size(), shape.data(), shape.size());
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, inputNamePointers_.data(),
                               &tensor, 1, outputNamePointers_.data(),
                               outputNamePointers_.size());
  std::vector<TensorOutput> copied;
  copied.reserve(outputs.size());
  for (auto& output : outputs) {
    if (!output.IsTensor()) {
      throw std::runtime_error("ONNX model returned a non-tensor output");
    }
    const auto info = output.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("ONNX model returned a non-float output");
    }
    TensorOutput value;
    value.shape = info.GetShape();
    const std::size_t count = info.GetElementCount();
    const float* data = output.GetTensorData<float>();
    value.data.assign(data, data + count);
    copied.push_back(std::move(value));
  }
  return copied;
}

std::optional<std::string> OnnxSession::metadata(const std::string& key) const {
  Ort::AllocatorWithDefaultOptions allocator;
  try {
    auto modelMetadata = session_->GetModelMetadata();
    auto value = modelMetadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
    if (value && value.get()[0] != '\0') {
      return std::string(value.get());
    }
  } catch (const Ort::Exception&) {
  }
  return std::nullopt;
}

}  // namespace insight_cup
