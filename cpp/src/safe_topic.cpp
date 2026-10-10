#include "insight_cup/safe_topic.hpp"

#include <utility>

namespace insight_cup {

SafeTopic::SafeTopic() {
  const auto context = rclcpp::contexts::get_global_default_context();
  if (!context->is_valid()) {
    rclcpp::init(0, nullptr);
  }
  node_ = std::make_shared<rclcpp::Node>("insight_cup_safety");
  publisher_ = node_->create_publisher<std_msgs::msg::String>("/safe", 10);
  executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spinThread_ = std::thread([this] { executor_->spin(); });
  RCLCPP_INFO(node_->get_logger(),
              "Publishing obstacle safety state on /safe");
}

SafeTopic::~SafeTopic() {
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

void SafeTopic::publish(const SafeObstacleState& state) {
  std_msgs::msg::String message;
  // Keep each ROS String sample on one line; YAML echo escapes embedded
  // newlines in multi-line JSON as literal "\\n" sequences.
  message.data = state.toJson().dump();
  publisher_->publish(std::move(message));
}

}  // namespace insight_cup
