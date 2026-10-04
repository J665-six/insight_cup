#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rknn_api.h>

#include "insight_cup/inference.hpp"

namespace insight_cup {

class RknnSession final : public InferenceSession {
 public:
  RknnSession(const fs::path& modelPath, int coreIndex);
  ~RknnSession() override;

  RknnSession(const RknnSession&) = delete;
  RknnSession& operator=(const RknnSession&) = delete;

  const std::vector<int64_t>& inputShape() const override {
    return inputShape_;
  }
  std::vector<TensorOutput> run(
      const std::vector<float>& input,
      const std::vector<int64_t>& shape) const override;
  std::optional<std::string> metadata(
      const std::string& key) const override;
  const fs::path& modelPath() const override { return modelPath_; }
  std::string backend() const override { return "rknn_cpp"; }
  std::string device() const override { return "npu"; }

  const std::string& apiVersion() const { return apiVersion_; }
  const std::string& driverVersion() const { return driverVersion_; }
  int coreIndex() const { return coreIndex_; }

 private:
  fs::path modelPath_;
  rknn_context context_ = 0;
  rknn_tensor_attr inputAttribute_{};
  std::vector<rknn_tensor_attr> outputAttributes_;
  std::vector<int64_t> inputShape_;
  std::string apiVersion_;
  std::string driverVersion_;
  int coreIndex_ = 0;
  std::optional<fs::path> performanceOutput_;
  mutable std::uint64_t performanceRunCount_ = 0;
  mutable std::uint64_t performanceDurationUs_ = 0;
  mutable std::uint64_t performanceMinUs_ = 0;
  mutable std::uint64_t performanceMaxUs_ = 0;
  mutable std::uint64_t performanceQueryErrors_ = 0;
  mutable std::mutex mutex_;
};

}  // namespace insight_cup
