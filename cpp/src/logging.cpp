#include "insight_cup/logging.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace insight_cup {

namespace {

std::string clockTime() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t value = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
  localtime_r(&value, &local);
  std::ostringstream output;
  output << std::put_time(&local, "%H:%M:%S");
  return output.str();
}

std::string numberText(const json& value) {
  if (value.is_null() || !value.is_number()) {
    return "-";
  }
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << value.get<double>();
  return output.str();
}

std::string stringOr(const json& value, const std::string& fallback) {
  return value.is_string() ? value.get<std::string>() : fallback;
}

}  // namespace

std::string localTimeIso(bool milliseconds) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t value = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
  localtime_r(&value, &local);
  std::ostringstream output;
  output << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
  if (milliseconds) {
    const auto epochMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now.time_since_epoch())
                             .count();
    output << '.' << std::setfill('0') << std::setw(3) << (epochMs % 1000);
  }
  output << std::put_time(&local, "%z");
  std::string result = output.str();
  if (result.size() >= 5) {
    result.insert(result.size() - 2, ":");
  }
  return result;
}

void writeJson(const fs::path& path, const json& payload) {
  fs::create_directories(path.parent_path());
  const fs::path temporary = path.parent_path() / ("." + path.filename().string() + ".tmp");
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("Could not write JSON: " + temporary.string());
    }
    output << payload.dump(2) << '\n';
  }
  fs::rename(temporary, path);
}

void Logger::debug(const std::string& message) const {
  if (verbose_) {
    log("DEBUG", message);
  }
}

void Logger::info(const std::string& message) const { log("INFO", message); }

void Logger::warning(const std::string& message) const {
  log("WARNING", message);
}

void Logger::error(const std::string& message) const { log("ERROR", message); }

void Logger::log(const char* level, const std::string& message) const {
  std::lock_guard<std::mutex> guard(mutex_);
  std::cerr << clockTime() << " | " << std::left << std::setw(7) << level
            << " | " << message << std::endl;
}

EventJournal::EventJournal(const fs::path& path, Logger& logger)
    : logger_(logger) {
  fs::create_directories(path.parent_path());
  stream_.open(path, std::ios::binary | std::ios::app);
  if (!stream_) {
    throw std::runtime_error("Could not open event journal: " + path.string());
  }
}

EventJournal::~EventJournal() { close(); }

void EventJournal::write(const json& event) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    stream_ << event.dump() << '\n';
    stream_.flush();
  }
  const auto& stage2 = event.at("stage2");
  std::string result = stringOr(stage2.value("predicted_class", json(nullptr)), "");
  if (result.empty()) {
    result = stringOr(stage2.value("candidate_class", json(nullptr)), "");
  }
  if (result.empty()) {
    result = stage2.value("status", "-");
  }
  std::ostringstream line;
  line << "RESULT frame=" << event.at("frame").at("index")
       << " id=" << event.at("id").get<std::string>()
       << " label=" << result
       << " probability=" << numberText(stage2.value("score", json(nullptr)))
       << " second="
       << stringOr(stage2.value("second_best_class", json(nullptr)), "-")
       << " second_probability="
       << numberText(stage2.value("second_best_score", json(nullptr)));
  logger_.info(line.str());
}

void EventJournal::close() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (stream_.is_open()) {
    stream_.close();
  }
}

}  // namespace insight_cup
