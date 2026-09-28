#pragma once

#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "insight_cup/config.hpp"
#include "insight_cup/logging.hpp"
#include "insight_cup/models.hpp"
#include "insight_cup/scheduler.hpp"
#include "insight_cup/state.hpp"

namespace insight_cup {

fs::path createSessionDirectory(const fs::path& outputRoot,
                                const std::optional<std::string>& sessionId);

class RecognitionRuntime {
 public:
  RecognitionRuntime(RuntimeConfig config, fs::path sessionDirectory,
                     RuntimeState& state, EventJournal& journal,
                     Logger& logger);
  ~RecognitionRuntime();

  RecognitionRuntime(const RecognitionRuntime&) = delete;
  RecognitionRuntime& operator=(const RecognitionRuntime&) = delete;

  void run();

 private:
  void loadModels();
  void runSource();
  void runRealSense();
  void runImages(const fs::path& source);
  void runCapture();
  cv::Mat processFrame(const cv::Mat& frame, int frameIndex,
                       int processedIndex, const std::string& sourceLabel,
                       std::optional<double> sourceTimeMs);
  void drainStage2();
  void writeFinalStage2Event(const FinalStage2Result& result);
  std::optional<std::string> saveCrop(const DetectedRegion& region);
  json buildEvent(const DetectedRegion& region,
                  const Stage2Resolution& resolution, const cv::Mat& frame,
                  int processedIndex, const std::string& sourceLabel,
                  std::optional<double> sourceTimeMs, double yoloMs) const;
  void drawResult(cv::Mat& image, const DetectedRegion& region,
                  const Stage2Decision& decision) const;
  void writeManifest() const;
  void writeSummary() const;

  RuntimeConfig config_;
  fs::path sessionDirectory_;
  RuntimeState& state_;
  EventJournal& journal_;
  Logger& logger_;
  std::unique_ptr<YoloDetector> detector_;
  std::shared_ptr<FaceRecognizer> face_;
  std::shared_ptr<KnifeRecognizer> knife_;
  std::unique_ptr<RecognitionRouter> router_;
  std::unique_ptr<AsyncStage2Scheduler> scheduler_;
  json modelMetadata_ = json::object();
  std::vector<std::string> mediaOutputs_;
  std::deque<double> frameIntervalsMs_;
  std::optional<std::chrono::steady_clock::time_point> lastFrameCompletedAt_;
  cv::Mat lastEventFrame_;
  std::optional<int> lastProcessedIndex_;
  std::optional<std::string> lastSourceLabel_;
  std::optional<double> lastSourceTimeMs_;
  std::map<int, YoloDebugInfo> trackDebug_;
};

}  // namespace insight_cup
