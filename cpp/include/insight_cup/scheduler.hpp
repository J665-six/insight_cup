#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "insight_cup/types.hpp"

namespace insight_cup {

using RecognizeFunction =
    std::function<Stage2Decision(const Stage1Detection&)>;

struct PreparedStage2Input {
  cv::Mat crop;
  std::optional<std::string> cropPath;
};

using PrepareJob = std::function<PreparedStage2Input()>;

class AsyncStage2Scheduler {
 public:
  AsyncStage2Scheduler(int refreshFrames, int stableRefreshFrames,
                       float iouThreshold, bool asynchronous,
                       int workersPerClass, std::string smoothingMethod,
                       int smoothingWindow, int switchConfirmations,
                       float faceThreshold, float faceMinMargin,
                       float knifeThreshold, float knifeMinMargin);
  ~AsyncStage2Scheduler();

  AsyncStage2Scheduler(const AsyncStage2Scheduler&) = delete;
  AsyncStage2Scheduler& operator=(const AsyncStage2Scheduler&) = delete;

  std::vector<Stage2Resolution> resolveFrame(
      const std::vector<DetectedRegion>& regions,
      const RecognizeFunction& recognize,
      const std::vector<PrepareJob>& prepareJobs);
  // Refresh and tracking settings are expressed against the historical
  // 30-FPS reference. The source rate is set once the capture is opened so
  // those settings preserve the same wall-clock behavior at 15 FPS.
  void setFrameRate(double frameRate);
  int refreshFrames() const;
  int stableRefreshFrames() const;
  bool waitForIdle(std::chrono::milliseconds timeout);
  std::vector<FinalStage2Result> finalizeResults();
  void close();
  bool asynchronous() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace insight_cup
