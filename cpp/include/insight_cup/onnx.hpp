#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "insight_cup/inference.hpp"

namespace insight_cup {

class OnnxSession final : public InferenceSession {
 public:
  OnnxSession(const fs::path& modelPath, int cpuThreads,
              const std::string& provider);
  ~OnnxSession() override = default;

  OnnxSession(const OnnxSession&) = delete;
  OnnxSession& operator=(const OnnxSession&) = delete;

  const std::vector<std::string>& inputNames() const { return inputNames_; }
  const std::vector<std::string>& outputNames() const { return outputNames_; }
  const std::vector<int64_t>& inputShape() const override {
    return inputShape_;
  }
  std::vector<TensorOutput> run(const std::vector<float>& input,
                                const std::vector<int64_t>& shape) const override;
  std::optional<std::string> metadata(
      const std::string& key) const override;
  const fs::path& modelPath() const override { return modelPath_; }
  std::string backend() const override { return "onnxruntime_cpp"; }
  std::string device() const override { return "cpu"; }

 private:
  fs::path modelPath_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;
  std::vector<const char*> inputNamePointers_;
  std::vector<const char*> outputNamePointers_;
  std::vector<int64_t> inputShape_;
};

}  // namespace insight_cup
