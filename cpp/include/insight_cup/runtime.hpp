#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <set>
#include <vector>

#include <opencv2/core.hpp>

#include "insight_cup/config.hpp"
#include "insight_cup/logging.hpp"
#include "insight_cup/models.hpp"
#include "insight_cup/scheduler.hpp"
#include "insight_cup/safe_obstacle.hpp"
#include "insight_cup/state.hpp"

namespace insight_cup {

fs::path createSessionDirectory(const fs::path& outputRoot,
                                const std::optional<std::string>& sessionId);

class RecognitionRuntime {
 public:
  RecognitionRuntime(RuntimeConfig config, fs::path sessionDirectory,
                     RuntimeState& state, EventJournal& journal,
                     Logger& logger,
                     std::function<void(const std::vector<std::string>&)>
                         visionPublisher = {},
                     std::function<void(const json&)> insightPublisher = {},
                     std::function<void(const SafeObstacleState&)>
                         safePublisher = {},
                     std::function<bool(bool, const std::string&)>
                         motionHoldRequester = {});
  ~RecognitionRuntime();

  RecognitionRuntime(const RecognitionRuntime&) = delete;
  RecognitionRuntime& operator=(const RecognitionRuntime&) = delete;

  void run();

 private:
  struct Stage1Track;

  void loadModels();
  void loadStage1Roi();
  void runSource();
  void runRealSense();
  void runRosSubscriber();
  void runImages(const fs::path& source);
  void runCapture();
  DetectionFrame stage1Frame(const cv::Mat& frame, int frameIndex,
                             int processedIndex);
  DetectionFrame detectStage1Region(const cv::Mat& frame, int frameIndex);
  bool stage1PointAllowed(const std::string& majorClass,
                          const cv::Point2f& center) const;
  const std::vector<cv::Point2f>* configuredObstacleRoi() const;
  bool stage2AreaEligible(const DetectedRegion& region) const;
  double normalizedFrameGap(int frameGap) const;
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
                       std::optional<double> sourceTimeMs,
                       const ObstacleDistanceReader& depthReader = {});
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
  std::function<void(const std::vector<std::string>&)> visionPublisher_;
  std::function<void(const json&)> insightPublisher_;
  std::function<void(const SafeObstacleState&)> safePublisher_;
  std::function<bool(bool, const std::string&)> motionHoldRequester_;
  std::unique_ptr<YoloDetector> detector_;
  std::shared_ptr<FaceRecognizer> face_;
  std::shared_ptr<KnifeRecognizer> knife_;
  std::unique_ptr<RecognitionRouter> router_;
  std::unique_ptr<AsyncStage2Scheduler> scheduler_;
  std::unique_ptr<SafeObstacleMonitor> safeMonitor_;
  std::vector<cv::Point2f> stage1Roi_;
  std::vector<cv::Point2f> obstacleRoiForDisplay_;
  fs::path stage1RoiConfigPath_;
  double stage2ReferenceAreaPx_ = 0.0;
  double stage2PersonReferenceAreaPx_ = 0.0;
  double stage2KnifeReferenceAreaPx_ = 0.0;
  bool stage1RoiConfigured_ = false;
  json modelMetadata_ = json::object();
  std::vector<std::string> mediaOutputs_;
  std::deque<double> frameIntervalsMs_;
  std::optional<std::chrono::steady_clock::time_point> lastFrameCompletedAt_;
  // Motion prediction uses a 30-FPS reference unit so it remains stable when
  // the shared camera stream is configured for 15 FPS.
  double sourceFps_ = 15.0;
  cv::Size lastEventFrameSize_;
  std::optional<int> lastProcessedIndex_;
  std::optional<std::string> lastSourceLabel_;
  std::optional<double> lastSourceTimeMs_;
  std::ofstream shadowStream_;
  json shadowInput_ = json::object();
  std::size_t shadowRecordCount_ = 0;

  bool inspectionEpisodeActive_ = false;
  bool inspectionHoldActive_ = false;
  bool inspectionReported_ = false;
  bool inspectionReleasePending_ = false;
  int inspectionClearFrames_ = 0;
  int inspectionHoldFrames_ = 0;
  int inspectionStableFrames_ = 0;
  std::set<std::string> inspectionStableLabels_;
  std::uint64_t recognitionEventId_ = 0;
  bool motionHoldUnavailableWarned_ = false;

  struct Stage1Track {
    int trackId = 0;
    DetectedRegion region;
    int lastYoloFrame = 0;
    std::array<float, 4> velocity = {0.0F, 0.0F, 0.0F, 0.0F};
    int hits = 1;
    int stableBoxHits = 1;
    int missedYoloFrames = 0;
  };
  std::vector<Stage1Track> stage1Tracks_;
  int nextStage1TrackId_ = 1;
  std::optional<int> lastFullYoloProcessedIndex_;
  int lastFullYoloTargetCount_ = 0;
  bool forceNextYoloFrame_ = true;
  // ROS image topics may publish the navigation camera at a different
  // resolution than the calibrated 848x480 runtime geometry.  Keep one
  // warning so the normalization is visible without flooding the log.
  bool rosSourceResizeLogged_ = false;
};

}  // namespace insight_cup
