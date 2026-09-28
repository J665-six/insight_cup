#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace insight_cup {

namespace fs = std::filesystem;

struct TensorOutput {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

class OnnxSession {
 public:
  OnnxSession(const fs::path& modelPath, int cpuThreads,
              const std::string& provider);
  ~OnnxSession() = default;

  OnnxSession(const OnnxSession&) = delete;
  OnnxSession& operator=(const OnnxSession&) = delete;

  const std::vector<std::string>& inputNames() const { return inputNames_; }
  const std::vector<std::string>& outputNames() const { return outputNames_; }
  const std::vector<int64_t>& inputShape() const { return inputShape_; }
  std::vector<TensorOutput> run(const std::vector<float>& input,
                                const std::vector<int64_t>& shape) const;
  std::optional<std::string> metadata(const std::string& key) const;

 private:
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;
  std::vector<const char*> inputNamePointers_;
  std::vector<const char*> outputNamePointers_;
  std::vector<int64_t> inputShape_;
};

}  // namespace insight_cup
