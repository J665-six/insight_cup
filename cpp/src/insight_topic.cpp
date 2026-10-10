#include "insight_cup/insight_topic.hpp"

#include <utility>

namespace insight_cup {

VisionTopic::VisionTopic() {
  const auto context = rclcpp::contexts::get_global_default_context();
  if (!context->is_valid()) {
    rclcpp::init(0, nullptr);
  }
  node_ = std::make_shared<rclcpp::Node>("insight_cup_recognition");
  publisher_ = node_->create_publisher<std_msgs::msg::String>("/vision", 10);
  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spinThread_ = std::thread([this] { executor_->spin(); });
  RCLCPP_INFO(node_->get_logger(),
              "Publishing recognition debug state on /vision");
}

VisionTopic::~VisionTopic() {
  if (executor_) {
    executor_->cancel();
  }
  if (spinThread_.joinable()) {
    spinThread_.join();
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
}

void VisionTopic::publish(const std::vector<std::string>& labels) {
  std_msgs::msg::String message;
  for (const auto& label : labels) {
    if (!message.data.empty()) {
      message.data.push_back(',');
    }
    message.data += label;
  }
  publisher_->publish(std::move(message));
}

InsightTopic::InsightTopic() {
  const auto context = rclcpp::contexts::get_global_default_context();
  if (!context->is_valid()) {
    rclcpp::init(0, nullptr);
  }
  node_ = std::make_shared<rclcpp::Node>("insight_cup_final");
  publisher_ = node_->create_publisher<std_msgs::msg::String>("/insight", 10);
  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spinThread_ = std::thread([this] { executor_->spin(); });
  RCLCPP_INFO(node_->get_logger(),
              "Publishing authoritative combined state on /insight");
}

InsightTopic::~InsightTopic() {
  if (executor_) {
    executor_->cancel();
  }
  if (spinThread_.joinable()) {
    spinThread_.join();
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
}

void InsightTopic::publish(const json& payload) {
  std_msgs::msg::String message;
  message.data = payload.dump(2);
  publisher_->publish(std::move(message));
}

}  // namespace insight_cup
