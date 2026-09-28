#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

namespace insight_cup {

namespace fs = std::filesystem;
using json = nlohmann::json;

class RuntimeState {
 public:
  explicit RuntimeState(bool retainEvents, std::size_t maxEvents = 200);

  void configureSession(const std::string& source, const std::string& sessionId,
                        const fs::path& sessionDirectory);
  void setSourceInfo(const json& values);
  void setStatus(const std::string& status,
                 const std::string& error = std::string());
  void setModule(const std::string& name, const std::string& status,
                 const std::string& detail = std::string());
  void recordEvent(const json& event);
  void recordDetection(const std::string& majorClass);
  void publishFrame(const std::vector<unsigned char>& jpeg, int frameIndex,
                    int processedFrames, double fps, const json& timingMs,
                    int detectionCount);

  json snapshot() const;
  json recentEvents(std::size_t limit) const;
  std::vector<unsigned char> latestFrame() const;
  std::tuple<std::uint64_t, std::vector<unsigned char>, std::string>
  waitForFrame(std::uint64_t afterVersion,
               std::chrono::milliseconds timeout);
  std::pair<bool, std::string> control(const std::string& action);
  bool waitUntilRunnable();
  bool stopRequested() const;

 private:
  void changed();

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<json> events_;
  std::size_t maxEvents_ = 200;
  bool retainEvents_ = true;
  std::vector<unsigned char> frameJpeg_;
  std::uint64_t frameVersion_ = 0;
  bool stopRequested_ = false;
  bool paused_ = false;
  std::uint64_t version_ = 0;
  json state_;
  std::map<std::string, std::uint64_t> majorCounts_;
  std::map<std::string, std::uint64_t> statusCounts_;
};

}  // namespace insight_cup
