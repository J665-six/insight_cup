#include "insight_cup/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>

#include "insight_cup/models.hpp"

namespace insight_cup {

namespace {

using Clock = std::chrono::steady_clock;

class ThreadPool {
 public:
  explicit ThreadPool(int workers) {
    for (int index = 0; index < workers; ++index) {
      workers_.emplace_back([this] { workerLoop(); });
    }
  }

  ~ThreadPool() { shutdown(); }

  void submit(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (stopping_) {
        return;
      }
      jobs_.push(std::move(job));
    }
    condition_.notify_one();
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (stopping_) {
        return;
      }
      stopping_ = true;
      std::queue<std::function<void()>> empty;
      jobs_.swap(empty);
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    workers_.clear();
  }

 private:
  void workerLoop() {
    while (true) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
        if (stopping_ && jobs_.empty()) {
          return;
        }
        job = std::move(jobs_.front());
        jobs_.pop();
      }
      job();
    }
  }

  std::mutex mutex_;
  std::condition_variable condition_;
  std::queue<std::function<void()>> jobs_;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
};

std::map<std::string, float> decisionScores(const Stage2Decision& decision) {
  if (!decision.classScores.empty()) {
    return decision.classScores;
  }
  std::map<std::string, float> scores;
  if (decision.candidateClass && decision.score) {
    scores[*decision.candidateClass] = *decision.score;
  }
  if (decision.secondBestClass && decision.secondBestScore) {
    scores[*decision.secondBestClass] = *decision.secondBestScore;
  }
  return scores;
}

json rawSummary(const Stage2Decision& decision) {
  return {
      {"status", decision.status},
      {"predicted_class",
       decision.predictedClass ? json(*decision.predictedClass) : json(nullptr)},
      {"candidate_class",
       decision.candidateClass ? json(*decision.candidateClass) : json(nullptr)},
      {"score", decision.score ? json(*decision.score) : json(nullptr)},
      {"second_best_class",
       decision.secondBestClass ? json(*decision.secondBestClass) : json(nullptr)},
      {"second_best_score",
       decision.secondBestScore ? json(*decision.secondBestScore) : json(nullptr)},
      {"margin", decision.margin ? json(*decision.margin) : json(nullptr)},
  };
}

class TemporalDecisionSmoother {
 public:
  TemporalDecisionSmoother(std::string method, int windowSize,
                           int switchConfirmations, float threshold,
                           float minMargin)
      : method_(std::move(method)),
        windowSize_(windowSize),
        switchConfirmations_(switchConfirmations),
        threshold_(threshold),
        minMargin_(minMargin) {}

  Stage2Decision update(const Stage2Decision& decision) {
    const std::string expectedPrefix = decision.module == "face" ? "f" : "k";
    auto scores = decisionScores(decision);
    for (auto iterator = scores.begin(); iterator != scores.end();) {
      if (iterator->first.rfind(expectedPrefix, 0) != 0) {
        iterator = scores.erase(iterator);
      } else {
        ++iterator;
      }
    }
    if (!decision.candidateClass ||
        decision.candidateClass->rfind(expectedPrefix, 0) != 0 ||
        scores.empty()) {
      return holdOrReject(decision);
    }

    history_.push_back(decision);
    while (static_cast<int>(history_.size()) > windowSize_) {
      history_.pop_front();
    }
    std::vector<std::map<std::string, float>> historyScores;
    std::set<std::string> labelSet;
    for (const auto& item : history_) {
      historyScores.push_back(decisionScores(item));
      for (const auto& [label, score] : historyScores.back()) {
        (void)score;
        if (label.rfind(expectedPrefix, 0) == 0) {
          labelSet.insert(label);
        }
      }
    }
    std::map<std::string, float> means;
    for (const auto& label : labelSet) {
      float total = 0.0F;
      for (const auto& item : historyScores) {
        const auto found = item.find(label);
        total += found == item.end() ? 0.0F : found->second;
      }
      means[label] = total / historyScores.size();
    }
    if (means.empty()) {
      return holdOrReject(decision);
    }

    std::vector<std::string> ranked;
    if (method_ == "mean") {
      for (const auto& [label, score] : means) {
        (void)score;
        ranked.push_back(label);
      }
      std::stable_sort(ranked.begin(), ranked.end(), [&](const auto& left,
                                                          const auto& right) {
        return means[left] > means[right];
      });
    } else {
      std::map<std::string, int> votes;
      std::map<std::string, int> latestPosition;
      std::vector<std::string> firstSeen;
      int position = 0;
      for (const auto& item : history_) {
        if (item.candidateClass && means.count(*item.candidateClass) != 0) {
          if (votes.count(*item.candidateClass) == 0) {
            firstSeen.push_back(*item.candidateClass);
          }
          ++votes[*item.candidateClass];
          latestPosition[*item.candidateClass] = position;
        }
        ++position;
      }
      ranked = firstSeen;
      std::stable_sort(ranked.begin(), ranked.end(), [&](const auto& left,
                                                          const auto& right) {
        return std::make_tuple(votes[left], means[left], latestPosition[left]) >
               std::make_tuple(votes[right], means[right], latestPosition[right]);
      });
    }

    const std::string proposed = ranked.front();
    const float proposedScore = means[proposed];
    std::vector<std::string> competitors;
    for (const auto& [label, score] : means) {
      (void)score;
      if (label != proposed) {
        competitors.push_back(label);
      }
    }
    std::sort(competitors.begin(), competitors.end(), [&](const auto& left,
                                                            const auto& right) {
      return means[left] > means[right];
    });
    const std::optional<std::string> proposedSecond =
        competitors.empty() ? std::nullopt
                            : std::optional<std::string>(competitors.front());
    const std::optional<float> proposedSecondScore =
        proposedSecond ? std::optional<float>(means[*proposedSecond]) : std::nullopt;
    const std::optional<float> proposedMargin =
        proposedSecondScore
            ? std::optional<float>(proposedScore - *proposedSecondScore)
            : std::nullopt;
    std::string proposedStatus = "matched";
    if (proposedScore < threshold_) {
      proposedStatus = "unknown";
    } else if (proposedMargin && *proposedMargin < minMargin_) {
      proposedStatus = "ambiguous";
    }

    if (proposedStatus == "matched") {
      uncertainCount_ = 0;
      if (!stableClass_) {
        updatePending(proposed);
        if (pendingCount_ >= switchConfirmations_) {
          stableClass_ = proposed;
          clearPending();
        }
      } else if (proposed == *stableClass_) {
        clearPending();
      } else {
        updatePending(proposed);
        if (pendingCount_ >= switchConfirmations_) {
          stableClass_ = proposed;
          clearPending();
        }
      }
    } else {
      ++uncertainCount_;
      clearPending();
      if (uncertainCount_ >= windowSize_) {
        stableClass_.reset();
      }
    }

    Stage2Decision output;
    output.module = decision.module;
    output.scoreKind = decision.scoreKind;
    output.classScores = means;
    if (!stableClass_) {
      output.status = proposedStatus == "matched" ? "confirming" : proposedStatus;
      output.candidateClass = proposed;
      output.score = proposedScore;
      output.secondBestClass = proposedSecond;
      output.secondBestScore = proposedSecondScore;
      output.margin = proposedMargin;
    } else {
      output.status = "matched";
      output.predictedClass = *stableClass_;
      output.candidateClass = *stableClass_;
      output.score = means.count(*stableClass_) != 0 ? means[*stableClass_] : 0.0F;
      std::vector<std::string> stableCompetitors;
      for (const auto& [label, score] : means) {
        (void)score;
        if (label != *stableClass_) {
          stableCompetitors.push_back(label);
        }
      }
      std::sort(stableCompetitors.begin(), stableCompetitors.end(),
                [&](const auto& left, const auto& right) {
                  return means[left] > means[right];
                });
      if (!stableCompetitors.empty()) {
        output.secondBestClass = stableCompetitors.front();
        output.secondBestScore = means[stableCompetitors.front()];
        output.margin = *output.score - *output.secondBestScore;
      }
    }
    output.metadata = temporalMetadata(decision, proposed);
    lastOutput_ = output;
    return output;
  }

 private:
  json temporalMetadata(const Stage2Decision& decision,
                        const std::optional<std::string>& proposed) const {
    json metadata = decision.metadata;
    metadata["temporal"] = {
        {"method", method_},
        {"window_size", windowSize_},
        {"samples", history_.size()},
        {"switch_confirmations", switchConfirmations_},
        {"stable_class", stableClass_ ? json(*stableClass_) : json(nullptr)},
        {"proposed_class", proposed ? json(*proposed) : json(nullptr)},
        {"pending_class", pendingClass_ ? json(*pendingClass_) : json(nullptr)},
        {"pending_count", pendingCount_},
        {"raw", rawSummary(decision)},
    };
    return metadata;
  }

  Stage2Decision holdOrReject(const Stage2Decision& decision) {
    ++uncertainCount_;
    clearPending();
    if (stableClass_ && lastOutput_ && uncertainCount_ < windowSize_) {
      Stage2Decision held;
      held.module = decision.module;
      held.status = "matched";
      held.predictedClass = *stableClass_;
      held.candidateClass = *stableClass_;
      held.score = lastOutput_->score;
      held.scoreKind = lastOutput_->scoreKind;
      held.secondBestClass = lastOutput_->secondBestClass;
      held.secondBestScore = lastOutput_->secondBestScore;
      held.margin = lastOutput_->margin;
      held.classScores = lastOutput_->classScores;
      held.metadata = temporalMetadata(decision, std::nullopt);
      lastOutput_ = held;
      return held;
    }
    stableClass_.reset();
    Stage2Decision rejected = decision;
    rejected.predictedClass.reset();
    rejected.metadata = temporalMetadata(decision, decision.candidateClass);
    lastOutput_ = rejected;
    return rejected;
  }

  void updatePending(const std::string& proposed) {
    if (pendingClass_ && *pendingClass_ == proposed) {
      ++pendingCount_;
    } else {
      pendingClass_ = proposed;
      pendingCount_ = 1;
    }
  }

  void clearPending() {
    pendingClass_.reset();
    pendingCount_ = 0;
  }

  std::string method_;
  int windowSize_ = 3;
  int switchConfirmations_ = 2;
  float threshold_ = 0.0F;
  float minMargin_ = 0.0F;
  std::deque<Stage2Decision> history_;
  std::optional<std::string> stableClass_;
  std::optional<std::string> pendingClass_;
  int pendingCount_ = 0;
  int uncertainCount_ = 0;
  std::optional<Stage2Decision> lastOutput_;
};

struct Track {
  int trackId = 0;
  std::string majorClass;
  Box bbox;
  int lastSeenFrame = 0;
  std::optional<Stage2Decision> decision;
  std::optional<int> resultFrame;
  std::optional<std::string> resultDetectionId;
  std::optional<Stage1Detection> resultDetection;
  std::optional<std::string> resultCropPath;
  double inferenceMs = 0.0;
  int revision = 0;
  int emittedRevision = 0;
  bool inFlight = false;
  int jobToken = 0;
  std::unique_ptr<TemporalDecisionSmoother> smoother;
  std::array<float, 4> velocity = {0.0F, 0.0F, 0.0F, 0.0F};
  int hits = 1;
};

std::array<float, 4> boxValues(const Box& box) {
  return {static_cast<float>(box.x1), static_cast<float>(box.y1),
          static_cast<float>(box.x2), static_cast<float>(box.y2)};
}

std::tuple<float, float, float, float> boxGeometry(const Box& box) {
  const float width = std::max(1, box.width());
  const float height = std::max(1, box.height());
  return {(box.x1 + box.x2) * 0.5F, (box.y1 + box.y2) * 0.5F, width,
          height};
}

}  // namespace

class AsyncStage2Scheduler::Impl {
 public:
  Impl(int refreshFrames, int stableRefreshFrames, float iouThreshold,
       bool asynchronous, int workersPerClass, std::string smoothingMethod,
       int smoothingWindow, int switchConfirmations, float faceThreshold,
       float faceMinMargin, float knifeThreshold, float knifeMinMargin)
      : refreshFrames_(refreshFrames),
        stableRefreshFrames_(stableRefreshFrames),
        iouThreshold_(iouThreshold),
        asynchronous_(asynchronous),
        workersPerClass_(workersPerClass),
        smoothingMethod_(std::move(smoothingMethod)),
        smoothingWindow_(smoothingWindow),
        switchConfirmations_(switchConfirmations),
        maxIdleFrames_(std::max(15, refreshFrames * 2)),
        staleJobGapFrames_(std::max(3, std::min(refreshFrames, 5))) {
    thresholds_["f"] = {faceThreshold, faceMinMargin};
    thresholds_["k"] = {knifeThreshold, knifeMinMargin};
    if (asynchronous_) {
      facePool_ = std::make_unique<ThreadPool>(workersPerClass_);
      knifePool_ = std::make_unique<ThreadPool>(workersPerClass_);
    }
  }

  std::vector<Stage2Resolution> resolveFrame(
      const std::vector<Stage1Detection>& detections,
      const RecognizeFunction& recognize,
      const std::vector<PrepareJob>& prepareJobs) {
    if (detections.empty()) {
      return {};
    }
    if (detections.size() != prepareJobs.size()) {
      throw std::runtime_error("Prepare jobs must match detection count");
    }
    for (const auto& detection : detections) {
      if (detection.frameIndex != detections.front().frameIndex) {
        throw std::runtime_error("Stage-2 detections must belong to one frame");
      }
    }
    if (!asynchronous_) {
      std::vector<Stage2Resolution> direct;
      direct.reserve(detections.size());
      for (std::size_t index = 0; index < detections.size(); ++index) {
        std::optional<std::string> cropPath;
        if (detections[index].majorClass != "b0" && prepareJobs[index]) {
          cropPath = prepareJobs[index]();
        }
        const auto started = Clock::now();
        Stage2Decision decision = recognize(detections[index]);
        direct.push_back({std::move(decision), "direct",
                          std::chrono::duration<double, std::milli>(Clock::now() -
                                                                   started)
                              .count(),
                          std::nullopt, std::nullopt,
                          detections[index].detectionId, cropPath});
      }
      return direct;
    }

    std::vector<std::optional<Stage2Resolution>> results(detections.size());
    std::vector<std::size_t> asyncIndexes;
    for (std::size_t index = 0; index < detections.size(); ++index) {
      if (detections[index].majorClass == "b0") {
        const auto started = Clock::now();
        Stage2Decision decision = recognize(detections[index]);
        results[index] = Stage2Resolution{
            std::move(decision),
            "direct",
            std::chrono::duration<double, std::milli>(Clock::now() - started)
                .count(),
            std::nullopt,
            std::nullopt,
            detections[index].detectionId,
            std::nullopt};
      } else {
        asyncIndexes.push_back(index);
      }
    }

    std::lock_guard<std::mutex> guard(mutex_);
    if (closed_) {
      throw std::runtime_error("Stage-2 scheduler is closed");
    }
    startFrame(detections.front().frameIndex);
    std::vector<Stage1Detection> asyncDetections;
    for (std::size_t index : asyncIndexes) {
      asyncDetections.push_back(detections[index]);
    }
    const auto associated = associateTracks(asyncDetections);
    for (std::size_t position = 0; position < asyncIndexes.size(); ++position) {
      const std::size_t index = asyncIndexes[position];
      results[index] = resolveTrack(associated[position], detections[index],
                                    recognize, prepareJobs[index]);
    }
    std::vector<Stage2Resolution> resolved;
    resolved.reserve(results.size());
    for (auto& item : results) {
      if (!item) {
        throw std::runtime_error("Stage-2 frame resolution is incomplete");
      }
      resolved.push_back(std::move(*item));
    }
    return resolved;
  }

  bool waitForIdle(std::chrono::milliseconds timeout) {
    const auto deadline = Clock::now() + timeout;
    std::unique_lock<std::mutex> lock(mutex_);
    return idleCondition_.wait_until(lock, deadline, [&] {
      return std::none_of(tracks_.begin(), tracks_.end(),
                          [](const auto& item) { return item.second->inFlight; });
    });
  }

  std::vector<FinalStage2Result> finalizeResults() {
    if (!asynchronous_) {
      return {};
    }
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<FinalStage2Result> output;
    for (const auto& [trackId, track] : tracks_) {
      (void)trackId;
      if (!track->decision || !track->resultDetection ||
          track->revision <= track->emittedRevision) {
        continue;
      }
      track->emittedRevision = track->revision;
      Stage2Resolution resolution;
      resolution.decision = *track->decision;
      resolution.mode = "fresh";
      resolution.inferenceMs = track->inferenceMs;
      resolution.trackId = track->trackId;
      resolution.sourceDetectionId = track->resultDetectionId;
      resolution.sourceCropPath = track->resultCropPath;
      output.push_back({std::move(resolution), *track->resultDetection});
    }
    return output;
  }

  void close() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (closed_) {
        return;
      }
      closed_ = true;
      for (auto& [trackId, track] : tracks_) {
        (void)trackId;
        ++track->jobToken;
        track->inFlight = false;
      }
    }
    if (facePool_) {
      facePool_->shutdown();
    }
    if (knifePool_) {
      knifePool_->shutdown();
    }
    idleCondition_.notify_all();
  }

  bool asynchronous() const { return asynchronous_; }

 private:
  void startFrame(int frameIndex) {
    if (activeFrame_ && *activeFrame_ == frameIndex) {
      return;
    }
    activeFrame_ = frameIndex;
    usedTrackIds_.clear();
    for (auto iterator = tracks_.begin(); iterator != tracks_.end();) {
      if (frameIndex - iterator->second->lastSeenFrame > maxIdleFrames_) {
        ++iterator->second->jobToken;
        iterator = tracks_.erase(iterator);
      } else {
        ++iterator;
      }
    }
  }

  Box predictedBox(const Track& track, int frameIndex) const {
    const int gap = std::max(0, frameIndex - track.lastSeenFrame);
    const auto values = boxValues(track.bbox);
    Box predicted{
        static_cast<int>(std::nearbyint(values[0] + track.velocity[0] * gap)),
        static_cast<int>(std::nearbyint(values[1] + track.velocity[1] * gap)),
        static_cast<int>(std::nearbyint(values[2] + track.velocity[2] * gap)),
        static_cast<int>(std::nearbyint(values[3] + track.velocity[3] * gap)),
    };
    return predicted.valid() ? predicted : track.bbox;
  }

  std::optional<float> associationScore(const Track& track,
                                        const Stage1Detection& detection) const {
    const Box predicted = predictedBox(track, detection.frameIndex);
    const float predictedOverlap = bboxIou(predicted, detection.bbox);
    const float previousOverlap = bboxIou(track.bbox, detection.bbox);
    const auto [currentX, currentY, currentWidth, currentHeight] =
        boxGeometry(detection.bbox);
    const auto [predictedX, predictedY, predictedWidth, predictedHeight] =
        boxGeometry(predicted);
    const float widthRatio =
        std::min(currentWidth, predictedWidth) / std::max(currentWidth, predictedWidth);
    const float heightRatio = std::min(currentHeight, predictedHeight) /
                              std::max(currentHeight, predictedHeight);
    if (widthRatio < 0.5F || heightRatio < 0.5F) {
      return std::nullopt;
    }
    const float centerDistance =
        std::hypot(currentX - predictedX, currentY - predictedY);
    const float diagonal = std::max(std::hypot(currentWidth, currentHeight),
                                    std::hypot(predictedWidth, predictedHeight));
    const int frameGap = std::max(1, detection.frameIndex - track.lastSeenFrame);
    const float distanceLimit = std::max(24.0F, diagonal * 0.45F) *
                                std::min(2.0F, std::sqrt(static_cast<float>(frameGap)));
    if (std::max(predictedOverlap, previousOverlap) < iouThreshold_ &&
        centerDistance > distanceLimit) {
      return std::nullopt;
    }
    const float normalizedDistance = centerDistance / std::max(diagonal, 1.0F);
    return predictedOverlap * 3.0F + previousOverlap +
           std::max(0.0F, 1.0F - normalizedDistance) +
           0.25F * (widthRatio + heightRatio);
  }

  std::shared_ptr<Track> newTrack(const Stage1Detection& detection) {
    auto track = std::make_shared<Track>();
    track->trackId = nextTrackId_++;
    track->majorClass = detection.majorClass;
    track->bbox = detection.bbox;
    track->lastSeenFrame = detection.frameIndex;
    if (smoothingMethod_ != "none") {
      const auto threshold = thresholds_.at(detection.majorClass);
      track->smoother = std::make_unique<TemporalDecisionSmoother>(
          smoothingMethod_, smoothingWindow_, switchConfirmations_,
          threshold.first, threshold.second);
    }
    tracks_[track->trackId] = track;
    return track;
  }

  void updateTrack(const std::shared_ptr<Track>& track,
                   const Stage1Detection& detection) {
    const int frameGap = std::max(1, detection.frameIndex - track->lastSeenFrame);
    if (frameGap > staleJobGapFrames_ && track->inFlight) {
      ++track->jobToken;
      track->inFlight = false;
    }
    const auto current = boxValues(detection.bbox);
    const auto previous = boxValues(track->bbox);
    for (std::size_t index = 0; index < current.size(); ++index) {
      const float observed = (current[index] - previous[index]) / frameGap;
      track->velocity[index] =
          track->hits <= 1 ? observed
                           : 0.70F * observed + 0.30F * track->velocity[index];
    }
    track->bbox = detection.bbox;
    track->lastSeenFrame = detection.frameIndex;
    ++track->hits;
  }

  std::vector<std::shared_ptr<Track>> associateTracks(
      const std::vector<Stage1Detection>& detections) {
    struct Pair {
      float score = 0.0F;
      std::size_t detectionIndex = 0;
      std::shared_ptr<Track> track;
    };
    std::vector<Pair> pairs;
    for (std::size_t detectionIndex = 0; detectionIndex < detections.size();
         ++detectionIndex) {
      for (const auto& [trackId, track] : tracks_) {
        if (usedTrackIds_.count(trackId) != 0 ||
            track->majorClass != detections[detectionIndex].majorClass) {
          continue;
        }
        const auto score = associationScore(*track, detections[detectionIndex]);
        if (score) {
          pairs.push_back({*score, detectionIndex, track});
        }
      }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& left, const Pair& right) {
      return left.score > right.score;
    });
    std::set<std::size_t> assignedDetections;
    std::set<int> assignedTracks;
    std::map<std::size_t, std::shared_ptr<Track>> matches;
    for (const auto& pair : pairs) {
      if (assignedDetections.count(pair.detectionIndex) != 0 ||
          assignedTracks.count(pair.track->trackId) != 0) {
        continue;
      }
      assignedDetections.insert(pair.detectionIndex);
      assignedTracks.insert(pair.track->trackId);
      matches[pair.detectionIndex] = pair.track;
    }
    std::vector<std::shared_ptr<Track>> resolved;
    for (std::size_t index = 0; index < detections.size(); ++index) {
      auto found = matches.find(index);
      std::shared_ptr<Track> track;
      if (found == matches.end()) {
        track = newTrack(detections[index]);
      } else {
        track = found->second;
        updateTrack(track, detections[index]);
      }
      usedTrackIds_.insert(track->trackId);
      resolved.push_back(std::move(track));
    }
    return resolved;
  }

  void submit(const std::shared_ptr<Track>& track,
              const Stage1Detection& detection,
              const RecognizeFunction& recognize,
              const std::optional<std::string>& cropPath) {
    if (closed_ || track->inFlight) {
      return;
    }
    ++track->jobToken;
    const int token = track->jobToken;
    track->inFlight = true;
    const int trackId = track->trackId;
    ThreadPool* pool = detection.majorClass == "f" ? facePool_.get() : knifePool_.get();
    pool->submit([this, trackId, token, detection, recognize, cropPath] {
      const auto started = Clock::now();
      Stage2Decision decision;
      try {
        decision = recognize(detection);
      } catch (const std::exception& error) {
        decision.module = detection.majorClass == "f" ? "face" : "knife";
        decision.status = "error";
        decision.metadata = {
            {"error", "RuntimeError: " + std::string(error.what())}};
      }
      const double inferenceMs =
          std::chrono::duration<double, std::milli>(Clock::now() - started).count();
      std::lock_guard<std::mutex> guard(mutex_);
      const auto found = tracks_.find(trackId);
      if (found == tracks_.end() || found->second->jobToken != token) {
        idleCondition_.notify_all();
        return;
      }
      auto& current = found->second;
      current->inFlight = false;
      current->decision = current->smoother
                              ? current->smoother->update(decision)
                              : std::optional<Stage2Decision>(decision);
      current->resultFrame = detection.frameIndex;
      current->resultDetectionId = detection.detectionId;
      current->resultDetection = detection;
      current->resultCropPath = cropPath;
      current->inferenceMs = inferenceMs;
      ++current->revision;
      idleCondition_.notify_all();
    });
  }

  Stage2Resolution resolveTrack(const std::shared_ptr<Track>& track,
                                const Stage1Detection& detection,
                                const RecognizeFunction& recognize,
                                const PrepareJob& prepareJob) {
    const std::optional<int> resultAge =
        track->resultFrame
            ? std::optional<int>(detection.frameIndex - *track->resultFrame)
            : std::nullopt;
    const int refreshInterval =
        track->decision && track->decision->status == "matched" &&
                track->decision->predictedClass
            ? stableRefreshFrames_
            : refreshFrames_;
    const bool needsRefresh = !track->decision || !resultAge ||
                              *resultAge >= refreshInterval;
    if (needsRefresh && !track->inFlight) {
      std::optional<std::string> cropPath;
      if (prepareJob) {
        cropPath = prepareJob();
      }
      submit(track, detection, recognize, cropPath);
    }
    if (!track->decision) {
      Stage2Decision processing;
      processing.module = detection.majorClass == "f" ? "face" : "knife";
      processing.status = "processing";
      return {std::move(processing), "processing", 0.0, track->trackId,
              std::nullopt, detection.detectionId, std::nullopt};
    }
    const bool fresh = track->revision > track->emittedRevision;
    if (fresh) {
      track->emittedRevision = track->revision;
    }
    return {*track->decision,
            fresh ? "fresh" : "cached",
            fresh ? track->inferenceMs : 0.0,
            track->trackId,
            resultAge,
            track->resultDetectionId,
            track->resultCropPath};
  }

  int refreshFrames_ = 15;
  int stableRefreshFrames_ = 30;
  float iouThreshold_ = 0.55F;
  bool asynchronous_ = true;
  int workersPerClass_ = 2;
  std::string smoothingMethod_;
  int smoothingWindow_ = 3;
  int switchConfirmations_ = 2;
  int maxIdleFrames_ = 30;
  int staleJobGapFrames_ = 5;
  std::map<std::string, std::pair<float, float>> thresholds_;
  std::unique_ptr<ThreadPool> facePool_;
  std::unique_ptr<ThreadPool> knifePool_;
  std::map<int, std::shared_ptr<Track>> tracks_;
  int nextTrackId_ = 1;
  std::optional<int> activeFrame_;
  std::set<int> usedTrackIds_;
  bool closed_ = false;
  std::mutex mutex_;
  std::condition_variable idleCondition_;
};

AsyncStage2Scheduler::AsyncStage2Scheduler(
    int refreshFrames, int stableRefreshFrames, float iouThreshold,
    bool asynchronous, int workersPerClass, std::string smoothingMethod,
    int smoothingWindow, int switchConfirmations, float faceThreshold,
    float faceMinMargin, float knifeThreshold, float knifeMinMargin)
    : impl_(std::make_unique<Impl>(
          refreshFrames, stableRefreshFrames, iouThreshold, asynchronous,
          workersPerClass, std::move(smoothingMethod), smoothingWindow,
          switchConfirmations, faceThreshold, faceMinMargin, knifeThreshold,
          knifeMinMargin)) {}

AsyncStage2Scheduler::~AsyncStage2Scheduler() { close(); }

std::vector<Stage2Resolution> AsyncStage2Scheduler::resolveFrame(
    const std::vector<Stage1Detection>& detections,
    const RecognizeFunction& recognize,
    const std::vector<PrepareJob>& prepareJobs) {
  return impl_->resolveFrame(detections, recognize, prepareJobs);
}

bool AsyncStage2Scheduler::waitForIdle(std::chrono::milliseconds timeout) {
  return impl_->waitForIdle(timeout);
}

std::vector<FinalStage2Result> AsyncStage2Scheduler::finalizeResults() {
  return impl_->finalizeResults();
}

void AsyncStage2Scheduler::close() {
  if (impl_) {
    impl_->close();
  }
}

bool AsyncStage2Scheduler::asynchronous() const {
  return impl_->asynchronous();
}

}  // namespace insight_cup
