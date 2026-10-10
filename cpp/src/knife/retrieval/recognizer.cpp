#include "insight_cup/knife/backend.hpp"

#include <stdexcept>

#include "../common.hpp"

namespace insight_cup {

namespace {

class RetrievalKnifeBackend final : public KnifeRecognitionBackend {
 public:
  explicit RetrievalKnifeBackend(const RuntimeConfig& config)
      : modelPath_(config.knifeRetrievalModel),
        galleryPath_(config.knifeGallery),
        gallery_(std::make_unique<FeatureGallery>(galleryPath_)),
        threshold_(config.knifeRetrievalThreshold),
        minMargin_(config.knifeRetrievalMinMargin),
        cpuThreads_(config.knifeCpuThreads) {
    if (gallery_->dimensions() != 512) {
      throw std::runtime_error(
          "Knife gallery must contain 512-dimensional embeddings");
    }
    session_ = createKnifeInferenceSession(
        modelPath_, fs::path(), config.provider, cpuThreads_, 2);
    modelPath_ = session_->modelPath();
  }

  Stage2Decision recognize(const cv::Mat& crop) const override {
    const auto outputs = session_->run(prepareKnifeModelInput(crop),
                                       {1, 3, 224, 224});
    if (outputs.size() != 1) {
      throw std::runtime_error("Knife model must return one output");
    }
    std::vector<float> embedding = outputs[0].data;
    if (embedding.size() != gallery_->dimensions()) {
      throw std::runtime_error(
          "Knife retrieval embedding dimension mismatch");
    }
    normalizeKnifeEmbedding(embedding);
    return knifeDecisionFromMatch(
        gallery_->matchByIdentityMax(embedding, threshold_, minMargin_),
        "cosine_similarity");
  }

  json metadata() const override {
    return {
        {"backend", "paddleclas_ppshitu_v2_cpp_" + session_->backend()},
        {"upstream_commit", "f1233c18455b8acde4fc42ab0bea575fa06daa8e"},
        {"model_path", modelPath_.string()},
        {"mode", mode_},
        {"labels", defaultKnifeLabels()},
        {"device", session_->device()},
        {"input_size", {224, 224}},
        {"threshold", threshold_},
        {"min_margin", minMargin_},
        {"cpu_threads", cpuThreads_},
        {"architecture", "GeneralRecognitionV2_PPLCNetV2_base"},
        {"gallery", galleryPath_.string()},
        {"gallery_references", gallery_->rows()},
        {"embedding_size", gallery_->dimensions()},
        {"index_method", "exact_inner_product_cpp"},
        {"distance", "cosine_similarity"},
    };
  }

  const std::string& mode() const override { return mode_; }
  std::size_t classCount() const override { return defaultKnifeLabels().size(); }
  float threshold() const override { return threshold_; }
  float minMargin() const override { return minMargin_; }

 private:
  const std::string mode_ = "retrieval";
  fs::path modelPath_;
  fs::path galleryPath_;
  std::unique_ptr<InferenceSession> session_;
  std::unique_ptr<FeatureGallery> gallery_;
  float threshold_ = 0.50F;
  float minMargin_ = 0.02F;
  int cpuThreads_ = 1;
};

}  // namespace

std::unique_ptr<KnifeRecognitionBackend> createRetrievalKnifeBackend(
    const RuntimeConfig& config) {
  return std::make_unique<RetrievalKnifeBackend>(config);
}

}  // namespace insight_cup
