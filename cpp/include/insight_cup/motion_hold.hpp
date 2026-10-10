#pragma once

#include <memory>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>

namespace insight_cup {

class MotionHoldClient {
 public:
  MotionHoldClient();
  ~MotionHoldClient();

  MotionHoldClient(const MotionHoldClient&) = delete;
  MotionHoldClient& operator=(const MotionHoldClient&) = delete;

  bool available() const;
  bool request(bool hold, const std::string& source,
               const std::string& reason);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace insight_cup
