#include "common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "insight_cup/knife/backend.hpp"
#include "insight_cup/models.hpp"
#include "insight_cup/onnx.hpp"
#if INSIGHT_CUP_HAS_RKNN
#include "insight_cup/rknn.hpp"
#endif

namespace insight_cup {

std::vector<std::string> defaultKnifeLabels() {
  std::vector<std::string> labels;
  labels.reserve(10);
  for (int index = 1; index <= 10; ++index) {
    labels.push_back("k" + std::to_string(index));
  }
  return labels;
}

std::vector<std::string> loadKnifeLabels(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Knife labels not found: " + path.string());
  }
  std::map<int, std::string> indexed;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream parser(line);
    int index = -1;
    std::string label;
    parser >> index >> label;
    if (index < 0 || label.empty()) {
      throw std::runtime_error("Invalid knife label line: " + line);
    }
    indexed[index] = label;
  }
  std::vector<std::string> labels(indexed.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const auto found = indexed.find(static_cast<int>(index));
    if (found == indexed.end()) {
      throw std::runtime_error("Knife labels must be contiguous from zero");
    }
    labels[index] = found->second;
  }
  for (int index = 1; index <= 10; ++index) {
    if (labels.size() < static_cast<std::size_t>(index) ||
        labels[index - 1] != "k" + std::to_string(index)) {
      throw std::runtime_error("The first ten knife labels must be k1...k10");
    }
  }
  return labels;
}

std::vector<float> prepareKnifeModelInput(const cv::Mat& crop) {
  if (crop.empty() || crop.type() != CV_8UC3) {
    throw std::runtime_error("Knife input must be a non-empty BGR image");
  }
  cv::Mat square = crop;
  if (crop.rows != crop.cols) {
    square = squareContextCrop(crop, Box{0, 0, crop.cols, crop.rows}, 1.0F);
  }
  cv::Mat resized;
  cv::resize(square, resized, cv::Size(224, 224), 0.0, 0.0, cv::INTER_LINEAR);
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
  const std::array<float, 3> means = {0.485F, 0.456F, 0.406F};
  const std::array<float, 3> standardDeviations = {0.229F, 0.224F, 0.225F};
  std::vector<cv::Mat> channels;
  cv::split(resized, channels);
  std::vector<float> output(3 * 224 * 224);
  const std::size_t plane = 224 * 224;
  for (std::size_t channel = 0; channel < channels.size(); ++channel) {
    channels[channel] = (channels[channel] - means[channel]) /
                        standardDeviations[channel];
    if (!channels[channel].isContinuous()) {
      channels[channel] = channels[channel].clone();
    }
    std::copy(channels[channel].ptr<float>(),
              channels[channel].ptr<float>() + plane,
              output.begin() + channel * plane);
  }
  return output;
}

float normalizeKnifeEmbedding(std::vector<float>& values) {
  float squaredNorm = 0.0F;
  for (float value : values) {
    squaredNorm += value * value;
  }
  const float norm = std::sqrt(squaredNorm);
  if (!std::isfinite(norm) || norm <= 0.0F) {
    throw std::runtime_error("Model produced an invalid embedding");
  }
  for (float& value : values) {
    value /= norm;
  }
  return norm;
}

Stage2Decision knifeDecisionFromMatch(const GalleryMatch& match,
                                      const std::string& scoreKind) {
  Stage2Decision decision;
  decision.module = "knife";
  decision.status = match.status;
  if (!match.predictedClass.empty()) {
    decision.predictedClass = match.predictedClass;
  }
  decision.candidateClass = match.candidateClass;
  decision.score = match.score;
  decision.scoreKind = scoreKind;
  if (!match.secondBestClass.empty()) {
    decision.secondBestClass = match.secondBestClass;
    decision.secondBestScore = match.secondBestScore;
    decision.margin = match.margin;
  }
  decision.classScores = match.classScores;
  return decision;
}

std::unique_ptr<InferenceSession> createKnifeInferenceSession(
    const fs::path& onnxPath, const fs::path& rknnPath,
    const std::string& provider, int cpuThreads, int npuCore) {
  (void)npuCore;
  const bool explicitRknnModel = onnxPath.extension() == ".rknn";
  const fs::path selectedRknn = explicitRknnModel ? onnxPath : rknnPath;
  bool autoRknn = false;
#if INSIGHT_CUP_HAS_RKNN
  autoRknn = provider == "auto" && fs::is_regular_file(selectedRknn);
#endif
  const bool useRknn =
      provider == "rknn" || explicitRknnModel || autoRknn;
  if (useRknn) {
    if (!fs::is_regular_file(selectedRknn)) {
      throw std::runtime_error("RKNN model not found: " +
                               selectedRknn.string());
    }
#if INSIGHT_CUP_HAS_RKNN
    return std::make_unique<RknnSession>(selectedRknn, npuCore);
#else
    throw std::runtime_error(
        "This build has no RKNN Runtime support; use --provider cpu");
#endif
  }
  return std::make_unique<OnnxSession>(onnxPath, cpuThreads, "cpu");
}

std::unique_ptr<KnifeRecognitionBackend> createKnifeBackend(
    const RuntimeConfig& config) {
  if (config.knifeMode == "classification") {
    return createClassificationKnifeBackend(config);
  }
  if (config.knifeMode == "retrieval") {
    return createRetrievalKnifeBackend(config);
  }
  if (config.knifeMode == "traditional") {
    return createTraditionalKnifeBackend(config);
  }
  throw std::runtime_error("Unsupported knife recognition mode: " +
                           config.knifeMode);
}

}  // namespace insight_cup
