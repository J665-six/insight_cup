#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "insight_cup/config.hpp"
#include "insight_cup/types.hpp"

namespace insight_cup {

using json = nlohmann::json;

class KnifeRecognitionBackend {
 public:
  virtual ~KnifeRecognitionBackend() = default;

  virtual Stage2Decision recognize(const cv::Mat& crop) const = 0;
  virtual json metadata() const = 0;
  virtual const std::string& mode() const = 0;
  virtual std::size_t classCount() const = 0;
  virtual float threshold() const = 0;
  virtual float minMargin() const = 0;
};

std::unique_ptr<KnifeRecognitionBackend> createKnifeBackend(
    const RuntimeConfig& config);

std::unique_ptr<KnifeRecognitionBackend> createClassificationKnifeBackend(
    const RuntimeConfig& config);
std::unique_ptr<KnifeRecognitionBackend> createRetrievalKnifeBackend(
    const RuntimeConfig& config);
std::unique_ptr<KnifeRecognitionBackend> createTraditionalKnifeBackend(
    const RuntimeConfig& config);

}  // namespace insight_cup
