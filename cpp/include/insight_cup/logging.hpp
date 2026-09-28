#pragma once

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

namespace insight_cup {

namespace fs = std::filesystem;
using json = nlohmann::json;

std::string localTimeIso(bool milliseconds = false);
void writeJson(const fs::path& path, const json& payload);

class Logger {
 public:
  explicit Logger(bool verbose = false) : verbose_(verbose) {}

  void debug(const std::string& message) const;
  void info(const std::string& message) const;
  void warning(const std::string& message) const;
  void error(const std::string& message) const;

 private:
  void log(const char* level, const std::string& message) const;

  bool verbose_ = false;
  mutable std::mutex mutex_;
};

class EventJournal {
 public:
  EventJournal(const fs::path& path, Logger& logger);
  ~EventJournal();

  EventJournal(const EventJournal&) = delete;
  EventJournal& operator=(const EventJournal&) = delete;

  void write(const json& event);
  void close();

 private:
  std::ofstream stream_;
  Logger& logger_;
  std::mutex mutex_;
};

}  // namespace insight_cup
