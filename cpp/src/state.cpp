#include "insight_cup/state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "insight_cup/logging.hpp"

namespace insight_cup {

RuntimeState::RuntimeState(bool retainEvents, std::size_t maxEvents)
    : maxEvents_(maxEvents), retainEvents_(retainEvents) {
  state_ = {
      {"schema", "recognition_runtime_state.v1"},
      {"status", "starting"},
      {"source", nullptr},
      {"source_info", json::object()},
      {"session_id", nullptr},
      {"session_dir", nullptr},
      {"started_at", localTimeIso(false)},
      {"ended_at", nullptr},
      {"error", nullptr},
      {"frame_index", nullptr},
      {"processed_frames", 0},
      {"fps", 0.0},
      {"last_timing_ms", {{"yolo", 0.0}, {"stage2", 0.0}, {"total", 0.0}}},
      {"detections_in_frame", 0},
      {"major_counts", {{"b0", 0}, {"f", 0}, {"k", 0}}},
      {"status_counts", json::object()},
      {"modules",
       {{"yolo", {{"status", "waiting"}, {"detail", ""}}},
        {"face", {{"status", "waiting"}, {"detail", ""}}},
        {"knife", {{"status", "waiting"}, {"detail", ""}}}}},
      {"has_frame", false},
  };
}

void RuntimeState::changed() {
  ++version_;
  condition_.notify_all();
}

void RuntimeState::configureSession(const std::string& source,
                                    const std::string& sessionId,
                                    const fs::path& sessionDirectory) {
  std::lock_guard<std::mutex> guard(mutex_);
  state_["source"] = source;
  state_["session_id"] = sessionId;
  state_["session_dir"] = sessionDirectory.string();
  changed();
}

void RuntimeState::setSourceInfo(const json& values) {
  std::lock_guard<std::mutex> guard(mutex_);
  state_["source_info"].update(values);
  changed();
}

void RuntimeState::setStatus(const std::string& status,
                             const std::string& error) {
  std::lock_guard<std::mutex> guard(mutex_);
  state_["status"] = status;
  state_["error"] = error.empty() ? json(nullptr) : json(error);
  if (status == "completed" || status == "stopped" || status == "error") {
    state_["ended_at"] = localTimeIso(false);
  }
  changed();
}

void RuntimeState::setModule(const std::string& name,
                             const std::string& status,
                             const std::string& detail) {
  std::lock_guard<std::mutex> guard(mutex_);
  state_["modules"][name] = {{"status", status}, {"detail", detail}};
  changed();
}

void RuntimeState::recordEvent(const json& event) {
  std::lock_guard<std::mutex> guard(mutex_);
  const std::string status = event.at("stage2").at("status").get<std::string>();
  ++statusCounts_[status];
  state_["status_counts"] = statusCounts_;
  if (retainEvents_) {
    events_.push_back(event);
    while (events_.size() > maxEvents_) {
      events_.pop_front();
    }
  }
  changed();
}

void RuntimeState::recordDetection(const std::string& majorClass) {
  if (majorClass != "b0" && majorClass != "f" && majorClass != "k") {
    throw std::runtime_error("Unsupported major class: " + majorClass);
  }
  std::lock_guard<std::mutex> guard(mutex_);
  ++majorCounts_[majorClass];
  state_["major_counts"] = {
      {"b0", majorCounts_["b0"]},
      {"f", majorCounts_["f"]},
      {"k", majorCounts_["k"]},
  };
  changed();
}

void RuntimeState::publishFrame(const std::vector<unsigned char>& jpeg,
                                int frameIndex, int processedFrames,
                                double fps, const json& timingMs,
                                int detectionCount) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!jpeg.empty()) {
    frameJpeg_ = jpeg;
    ++frameVersion_;
  }
  state_["frame_index"] = frameIndex;
  state_["processed_frames"] = processedFrames;
  state_["fps"] = std::round(fps * 100.0) / 100.0;
  state_["last_timing_ms"] = timingMs;
  state_["detections_in_frame"] = detectionCount;
  state_["has_frame"] = !frameJpeg_.empty();
  changed();
}

json RuntimeState::snapshot() const {
  std::lock_guard<std::mutex> guard(mutex_);
  json result = state_;
  result["version"] = version_;
  result["paused"] = paused_;
  result["stop_requested"] = stopRequested_;
  return result;
}

json RuntimeState::recentEvents(std::size_t limit) const {
  std::lock_guard<std::mutex> guard(mutex_);
  limit = std::max<std::size_t>(1, std::min(limit, maxEvents_));
  json result = json::array();
  std::size_t count = 0;
  for (auto iterator = events_.rbegin(); iterator != events_.rend() && count < limit;
       ++iterator, ++count) {
    result.push_back(*iterator);
  }
  return result;
}

std::vector<unsigned char> RuntimeState::latestFrame() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return frameJpeg_;
}

std::tuple<std::uint64_t, std::vector<unsigned char>, std::string>
RuntimeState::waitForFrame(std::uint64_t afterVersion,
                           std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  condition_.wait_for(lock, timeout, [&] {
    const std::string status = state_["status"].get<std::string>();
    return frameVersion_ > afterVersion || status == "completed" ||
           status == "stopped" || status == "error";
  });
  return {frameVersion_, frameJpeg_, state_["status"].get<std::string>()};
}

std::pair<bool, std::string> RuntimeState::control(const std::string& action) {
  std::lock_guard<std::mutex> guard(mutex_);
  const std::string status = state_["status"].get<std::string>();
  if (action == "pause") {
    if (status != "running") {
      return {false, "cannot pause while status is " + status};
    }
    paused_ = true;
    state_["status"] = "paused";
  } else if (action == "resume") {
    if (status != "paused") {
      return {false, "cannot resume while status is " + status};
    }
    paused_ = false;
    state_["status"] = "running";
  } else if (action == "stop") {
    if (status == "completed" || status == "stopped" || status == "error") {
      return {false, "runtime is already " + status};
    }
    stopRequested_ = true;
    paused_ = false;
    state_["status"] = "stopping";
  } else {
    return {false, "unsupported action: " + action};
  }
  changed();
  return {true, action};
}

bool RuntimeState::waitUntilRunnable() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (paused_ && !stopRequested_) {
    condition_.wait_for(lock, std::chrono::milliseconds(500));
  }
  return !stopRequested_;
}

bool RuntimeState::stopRequested() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return stopRequested_;
}

}  // namespace insight_cup
