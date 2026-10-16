#include "insight_cup/safe_topic.hpp"

#include <utility>

namespace insight_cup {

namespace {

constexpr int kNavigationObstacleMinDistanceCm = 40;
constexpr int kNavigationObstacleMaxDistanceCm = 50;

SafeObstacleState stateForNavigation(const SafeObstacleState& state) {
  if (!state.obstacle ||
      (state.distanceCm >= kNavigationObstacleMinDistanceCm &&
       state.distanceCm <= kNavigationObstacleMaxDistanceCm)) {
    return state;
  }

  SafeObstacleState filtered = state;
  filtered.obstacle = false;
  filtered.status = "clear";
  filtered.distanceCm = 0;
  filtered.distanceM = 0.0;
  filtered.stableSeconds = 0.0;
  filtered.stableFrames = 0;
  filtered.confidence = 0.0F;
  filtered.bbox = {};
  return filtered;
}

}  // namespace

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
  message.data = stateForNavigation(state).toJson().dump();
  publisher_->publish(std::move(message));
}

}  // namespace insight_cup
