#pragma once

#include <chrono>
#include <deque>
#include <functional>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "insight_cup/config.hpp"
#include "insight_cup/logging.hpp"
#include "insight_cup/types.hpp"

namespace insight_cup {

struct SafeObstacleState {
  bool obstacle = false;
  bool configured = false;
  std::string status = "disabled";
  int distanceCm = 0;
  double distanceM = 0.0;
  double stableSeconds = 0.0;
  float confidence = 0.0F;
  int stableFrames = 0;
  Box bbox;
  std::string configPath;

  json toJson() const;
};

using ObstacleDistanceReader = std::function<std::optional<int>(const Box&)>;

class SafeObstacleMonitor {
 public:
  SafeObstacleMonitor(const RuntimeConfig& config, Logger& logger);

  SafeObstacleMonitor(const SafeObstacleMonitor&) = delete;
  SafeObstacleMonitor& operator=(const SafeObstacleMonitor&) = delete;

  SafeObstacleState update(const DetectionFrame& frame, const cv::Size& frameSize,
                           int processedIndex,
                           std::optional<double> sourceTimeMs,
                           const ObstacleDistanceReader& depthReader = {});
  json metadata() const;
  const std::vector<cv::Point2f>& obstacleRoi() const { return obstacleRoi_; }

 private:
  struct Zone {
    int sourceId = 0;
    int distanceCm = 0;
    std::vector<cv::Point2f> polygon;
    cv::Point2f center;
    double area = 0.0;
  };

  struct PixelSample {
    int distanceCm = 0;
    double scale = 0.0;
    double width = 0.0;
    double height = 0.0;
    double area = 0.0;
  };

  void loadConfiguration();
  void loadPayload(const json& payload, const std::filesystem::path& path);
  void inferDistancePayload(const json& payload,
                            const std::filesystem::path& path);
  void loadPixelCalibration(const std::filesystem::path& horizontalPath,
                            const std::filesystem::path& verticalPath,
                            const std::filesystem::path& roiPath);
  bool pointInPolygon(const cv::Point2f& point,
                     const std::vector<cv::Point2f>& polygon) const;
  std::optional<int> distanceForBox(const Box& box) const;
  void resetCandidate();

  const RuntimeConfig& config_;
  Logger& logger_;
  std::filesystem::path loadedConfigPath_;
  std::vector<cv::Point2f> obstacleRoi_;
  std::vector<Zone> zones_;
  std::vector<PixelSample> horizontalSamples_;
  std::vector<PixelSample> verticalSamples_;
  double shapeAspectThreshold_ = 1.15;
  bool configured_ = false;

  std::optional<int> candidateDistanceCm_;
  std::deque<int> distanceHistoryCm_;
  std::optional<double> filteredDistanceCm_;
  cv::Point2f candidateCenter_;
  Box candidateBox_;
  float candidateConfidence_ = 0.0F;
  int candidateFrames_ = 0;
  int lastProcessedIndex_ = -1;
  std::optional<double> candidateStartSourceMs_;
  int candidateStartFrame_ = -1;
  SafeObstacleState lastState_;
};

}  // namespace insight_cup
