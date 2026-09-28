#pragma once

#include <memory>
#include <string>

#include "insight_cup/logging.hpp"
#include "insight_cup/state.hpp"

namespace insight_cup {

class DebugServer {
 public:
  DebugServer(std::string host, int port, RuntimeState& state, Logger& logger);
  ~DebugServer();

  DebugServer(const DebugServer&) = delete;
  DebugServer& operator=(const DebugServer&) = delete;

  bool listen();
  void stop();
  int port() const { return port_; }
  const std::string& host() const { return host_; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  std::string host_;
  int port_ = 0;
};

}  // namespace insight_cup
