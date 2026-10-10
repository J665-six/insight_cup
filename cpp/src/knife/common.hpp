#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "insight_cup/config.hpp"
#include "insight_cup/gallery.hpp"
#include "insight_cup/inference.hpp"
#include "insight_cup/types.hpp"

namespace insight_cup {

namespace fs = std::filesystem;

std::vector<std::string> defaultKnifeLabels();
std::vector<std::string> loadKnifeLabels(const fs::path& path);
std::vector<float> prepareKnifeModelInput(const cv::Mat& crop);
float normalizeKnifeEmbedding(std::vector<float>& values);
Stage2Decision knifeDecisionFromMatch(const GalleryMatch& match,
                                      const std::string& scoreKind);
std::unique_ptr<InferenceSession> createKnifeInferenceSession(
    const fs::path& onnxPath, const fs::path& rknnPath,
    const std::string& provider, int cpuThreads, int npuCore);

}  // namespace insight_cup
