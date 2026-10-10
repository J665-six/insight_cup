#pragma once

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "insight_cup/types.hpp"

namespace insight_cup {

class VisionTopic {
 public:
  VisionTopic();
  ~VisionTopic();

  VisionTopic(const VisionTopic&) = delete;
  VisionTopic& operator=(const VisionTopic&) = delete;

  void publish(const std::vector<std::string>& labels);

 private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spinThread_;
};

class InsightTopic {
 public:
  InsightTopic();
  ~InsightTopic();

  InsightTopic(const InsightTopic&) = delete;
  InsightTopic& operator=(const InsightTopic&) = delete;

  void publish(const json& payload);

 private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread spinThread_;
};

}  // namespace insight_cup
