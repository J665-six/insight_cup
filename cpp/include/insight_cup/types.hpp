#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

namespace insight_cup {

using json = nlohmann::json;

struct Box {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  int width() const { return x2 - x1; }
  int height() const { return y2 - y1; }
  bool valid() const { return x2 > x1 && y2 > y1; }
  json toJson() const { return json::array({x1, y1, x2, y2}); }
};

struct FloatBox {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;

  json toJson() const { return json::array({x1, y1, x2, y2}); }
};

struct Stage1Detection {
  std::string detectionId;
  int frameIndex = 0;
  std::string majorClass;
  float confidence = 0.0F;
  Box bbox;
  cv::Mat crop;

  json handoffJson(const std::optional<std::string>& cropPath) const;
};

struct YoloDebugInfo {
  int rawClassId = -1;
  std::string rawClassName;

  json toJson() const;
};

struct DetectedRegion {
  Stage1Detection handoff;
  YoloDebugInfo debug;
};

struct DetectionFrame {
  std::vector<DetectedRegion> regions;
  double inferenceMs = 0.0;
  std::vector<std::string> ignoredClasses;
};

struct Stage2Decision {
  std::string module;
  std::string status;
  std::optional<std::string> predictedClass;
  std::optional<std::string> candidateClass;
  std::optional<float> score;
  std::optional<std::string> scoreKind;
  std::optional<std::string> secondBestClass;
  std::optional<float> secondBestScore;
  std::optional<float> margin;
  std::map<std::string, float> classScores;
  json metadata = json::object();

  json toJson() const;
};

struct Stage2Resolution {
  Stage2Decision decision;
  std::string mode;
  double inferenceMs = 0.0;
  std::optional<int> trackId;
  std::optional<int> cacheAgeFrames;
  std::optional<std::string> sourceDetectionId;
  std::optional<std::string> sourceCropPath;

  bool hasNewResult() const { return mode == "direct" || mode == "fresh"; }
};

struct FinalStage2Result {
  Stage2Resolution resolution;
  Stage1Detection detection;
};

}  // namespace insight_cup
