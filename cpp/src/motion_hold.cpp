#include "insight_cup/motion_hold.hpp"

#include <chrono>
#include <future>
#include <type_traits>

namespace {

template <typename T, typename = void>
struct HasSuccessField : std::false_type {};

template <typename T>
struct HasSuccessField<T, std::void_t<decltype(std::declval<T>().success)>>
    : std::true_type {};

}  // namespace

#if INSIGHT_CUP_HAS_MOTION_HOLD
#include <mission_control_interfaces/srv/set_motion_hold.hpp>
#endif

namespace insight_cup {

struct MotionHoldClient::Impl {
#if INSIGHT_CUP_HAS_MOTION_HOLD
  rclcpp::Node::SharedPtr node;
  rclcpp::Client<mission_control_interfaces::srv::SetMotionHold>::SharedPtr client;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor;
  std::thread spinThread;
#endif
};

MotionHoldClient::MotionHoldClient() : impl_(std::make_unique<Impl>()) {
#if INSIGHT_CUP_HAS_MOTION_HOLD
  const auto context = rclcpp::contexts::get_global_default_context();
  if (!context->is_valid()) {
    rclcpp::init(0, nullptr);
  }
  impl_->node = std::make_shared<rclcpp::Node>("insight_cup_motion_hold");
  impl_->client = impl_->node->create_client<
      mission_control_interfaces::srv::SetMotionHold>(
      "/navigation_mission/set_motion_hold");
  impl_->executor =
      std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
  impl_->executor->add_node(impl_->node);
  impl_->spinThread = std::thread([this] { impl_->executor->spin(); });
  RCLCPP_INFO(impl_->node->get_logger(),
              "Motion hold client ready: /navigation_mission/set_motion_hold");
#endif
}

MotionHoldClient::~MotionHoldClient() {
#if INSIGHT_CUP_HAS_MOTION_HOLD
  if (impl_->executor) {
    impl_->executor->cancel();
  }
  if (impl_->spinThread.joinable()) {
    impl_->spinThread.join();
  }
#endif
}

bool MotionHoldClient::available() const {
#if INSIGHT_CUP_HAS_MOTION_HOLD
  return impl_->client && impl_->client->wait_for_service(std::chrono::milliseconds(0));
#else
  return false;
#endif
}

bool MotionHoldClient::request(bool hold, const std::string& source,
                               const std::string& reason) {
#if INSIGHT_CUP_HAS_MOTION_HOLD
  if (!impl_->client->wait_for_service(std::chrono::milliseconds(500))) {
    RCLCPP_ERROR(impl_->node->get_logger(),
                 "Motion hold service unavailable hold=%s source=%s",
                 hold ? "true" : "false", source.c_str());
    return false;
  }
  auto request = std::make_shared<
      mission_control_interfaces::srv::SetMotionHold::Request>();
  request->source = source;
  request->hold = hold;
  request->reason = reason;
  auto future = impl_->client->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
    RCLCPP_ERROR(impl_->node->get_logger(),
                 "Motion hold service timeout hold=%s source=%s",
                 hold ? "true" : "false", source.c_str());
    return false;
  }
  const auto response = future.get();
  bool success = true;
  if constexpr (HasSuccessField<
                    mission_control_interfaces::srv::SetMotionHold::Response>::value) {
    success = response->success;
  }
  RCLCPP_INFO(impl_->node->get_logger(),
              "Motion hold request completed hold=%s source=%s success=%s reason=%s",
              hold ? "true" : "false", source.c_str(),
              success ? "true" : "false", reason.c_str());
  return success;
#else
  (void)hold;
  (void)source;
  (void)reason;
  return false;
#endif
}

}  // namespace insight_cup
