#pragma once

#include <array>
#include <deque>
#include <filesystem>
#include <fstream>
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
  struct Stage1Track;

  void loadModels();
  void runSource();
  void runRealSense();
  void runImages(const fs::path& source);
  void runCapture();
  DetectionFrame stage1Frame(const cv::Mat& frame, int frameIndex,
                             int processedIndex);
  std::optional<float> stage1AssociationScore(const Stage1Track& track,
                                              const DetectedRegion& region,
                                              const cv::Size& frameSize) const;
  DetectionFrame trackYoloDetections(DetectionFrame detectionFrame,
                                     const cv::Size& frameSize,
                                     int frameIndex);
  DetectionFrame predictTrackedDetections(const cv::Size& frameSize,
                                          int frameIndex) const;
  void requestYoloOnUnstableResult(const Stage2Resolution& resolution);
  cv::Mat processFrame(const cv::Mat& frame, int frameIndex,
                       int processedIndex, const std::string& sourceLabel,
                       std::optional<double> sourceTimeMs);
  void drainStage2();
  void writeFinalStage2Event(const FinalStage2Result& result);
  std::optional<std::string> saveCrop(const Stage1Detection& detection,
                                      const cv::Mat& crop);
  json buildEvent(const DetectedRegion& region,
                  const Stage2Resolution& resolution, const cv::Size& frameSize,
                  int processedIndex, const std::string& sourceLabel,
                  std::optional<double> sourceTimeMs, double yoloMs) const;
  json buildShadowEvent(const Stage2Resolution& resolution) const;
  void writeShadowEvent(const Stage2Resolution& resolution);
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
  cv::Size lastEventFrameSize_;
  std::optional<int> lastProcessedIndex_;
  std::optional<std::string> lastSourceLabel_;
  std::optional<double> lastSourceTimeMs_;
  std::ofstream shadowStream_;
  json shadowInput_ = json::object();
  std::size_t shadowRecordCount_ = 0;

  struct Stage1Track {
    int trackId = 0;
    DetectedRegion region;
    int lastYoloFrame = 0;
    std::array<float, 4> velocity = {0.0F, 0.0F, 0.0F, 0.0F};
    int hits = 1;
    int missedYoloFrames = 0;
  };
  std::vector<Stage1Track> stage1Tracks_;
  int nextStage1TrackId_ = 1;
  std::optional<int> lastFullYoloProcessedIndex_;
  int lastFullYoloTargetCount_ = 0;
  bool forceNextYoloFrame_ = true;
};

}  // namespace insight_cup
