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
using PrepareJob = std::function<std::optional<std::string>()>;

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
      const std::vector<Stage1Detection>& detections,
      const RecognizeFunction& recognize,
      const std::vector<PrepareJob>& prepareJobs);
  bool waitForIdle(std::chrono::milliseconds timeout);
  std::vector<FinalStage2Result> finalizeResults();
  void close();
  bool asynchronous() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace insight_cup
