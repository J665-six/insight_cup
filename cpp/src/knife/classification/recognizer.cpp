#include "insight_cup/knife/backend.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "../common.hpp"

namespace insight_cup {

namespace {

class ClassificationKnifeBackend final : public KnifeRecognitionBackend {
 public:
  explicit ClassificationKnifeBackend(const RuntimeConfig& config)
      : modelPath_(config.knifeModelDir / "inference.onnx"),
        labels_(loadKnifeLabels(config.knifeModelDir / "labels.txt")),
        threshold_(config.knifeThreshold),
        minMargin_(config.knifeMinMargin),
        cpuThreads_(config.knifeCpuThreads) {
    session_ = createKnifeInferenceSession(
        modelPath_, config.rknnModelDir / "knife_pplcnetv2.rknn",
        config.provider, cpuThreads_, 2);
    modelPath_ = session_->modelPath();
  }

  Stage2Decision recognize(const cv::Mat& crop) const override {
    const auto outputs = session_->run(prepareKnifeModelInput(crop),
                                       {1, 3, 224, 224});
    if (outputs.size() != 1) {
      throw std::runtime_error("Knife model must return one output");
    }
    if (outputs[0].data.size() != labels_.size()) {
      throw std::runtime_error("Knife classifier output does not match labels");
    }

    GalleryMatch match;
    float sum = 0.0F;
    for (std::size_t index = 0; index < labels_.size(); ++index) {
      const float score = outputs[0].data[index];
      if (!std::isfinite(score) || score < -1e-5F) {
        throw std::runtime_error("Knife classifier produced invalid probabilities");
      }
      sum += score;
      match.classScores[labels_[index]] = score;
    }
    if (std::abs(sum - 1.0F) > 1e-3F) {
      throw std::runtime_error("Knife classifier output is not softmax-normalized");
    }

    std::vector<std::size_t> indexes(labels_.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::partial_sort(
        indexes.begin(), indexes.begin() + std::min<std::size_t>(2, indexes.size()),
        indexes.end(), [&](std::size_t left, std::size_t right) {
          return outputs[0].data[left] > outputs[0].data[right];
        });
    match.candidateClass = labels_[indexes[0]];
    match.score = outputs[0].data[indexes[0]];
    if (indexes.size() > 1) {
      match.secondBestClass = labels_[indexes[1]];
      match.secondBestScore = outputs[0].data[indexes[1]];
      match.margin = match.score - match.secondBestScore;
    }
    if (match.candidateClass == "other" || match.score < threshold_) {
      match.status = "unknown";
    } else if (!match.secondBestClass.empty() && match.margin < minMargin_) {
      match.status = "ambiguous";
    } else {
      match.status = "matched";
      match.predictedClass = match.candidateClass;
    }
    return knifeDecisionFromMatch(match, "softmax_confidence");
  }

  json metadata() const override {
    return {
        {"backend", "paddleclas_cpp_" + session_->backend()},
        {"upstream_commit", "f1233c18455b8acde4fc42ab0bea575fa06daa8e"},
        {"model_path", modelPath_.string()},
        {"mode", mode_},
        {"labels", labels_},
        {"device", session_->device()},
        {"input_size", {224, 224}},
        {"threshold", threshold_},
        {"min_margin", minMargin_},
        {"cpu_threads", cpuThreads_},
        {"architecture", "PPLCNetV2_base"},
        {"score_kind", "softmax_confidence"},
    };
  }

  const std::string& mode() const override { return mode_; }
  std::size_t classCount() const override { return labels_.size(); }
  float threshold() const override { return threshold_; }
  float minMargin() const override { return minMargin_; }

 private:
  const std::string mode_ = "classification";
  fs::path modelPath_;
  std::unique_ptr<InferenceSession> session_;
  std::vector<std::string> labels_;
  float threshold_ = 0.50F;
  float minMargin_ = 0.15F;
  int cpuThreads_ = 1;
};

}  // namespace

std::unique_ptr<KnifeRecognitionBackend> createClassificationKnifeBackend(
    const RuntimeConfig& config) {
  return std::make_unique<ClassificationKnifeBackend>(config);
}

}  // namespace insight_cup
