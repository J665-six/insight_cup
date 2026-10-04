#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace insight_cup {

namespace fs = std::filesystem;

struct TensorOutput {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

class InferenceSession {
 public:
  virtual ~InferenceSession() = default;

  virtual const std::vector<int64_t>& inputShape() const = 0;
  virtual std::vector<TensorOutput> run(
      const std::vector<float>& input,
      const std::vector<int64_t>& shape) const = 0;
  virtual std::optional<std::string> metadata(
      const std::string& key) const = 0;
  virtual const fs::path& modelPath() const = 0;
  virtual std::string backend() const = 0;
  virtual std::string device() const = 0;
};

}  // namespace insight_cup
