#pragma once

#include <memory>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "insight_cup/safe_obstacle.hpp"

namespace insight_cup {

class SafeTopic {
 public:
  SafeTopic();
  ~SafeTopic();

  SafeTopic(const SafeTopic&) = delete;
  SafeTopic& operator=(const SafeTopic&) = delete;

  void publish(const SafeObstacleState& state);

 private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spinThread_;
};

}  // namespace insight_cup
