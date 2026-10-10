#include <cmath>
#include <iostream>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "insight_cup/stereo_range.hpp"

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void requireNear(float actual, float expected, float tolerance,
                 const char *message) {
  require(std::abs(actual - expected) <= tolerance, message);
}

rs2_intrinsics testIntrinsics() {
  rs2_intrinsics intrinsics{};
  intrinsics.width = 100;
  intrinsics.height = 100;
  intrinsics.ppx = 50.0F;
  intrinsics.ppy = 50.0F;
  intrinsics.fx = 100.0F;
  intrinsics.fy = 100.0F;
  intrinsics.model = RS2_DISTORTION_NONE;
  return intrinsics;
}

void testMedianDepthAndThreeDimensionalRange() {
  cv::Mat depth(100, 100, CV_16UC1, cv::Scalar(2000));
  depth.at<std::uint16_t>(50, 50) = 8000;
  const auto result = insight_cup::measureStereoRange(
      depth, cv::Rect(40, 40, 20, 20), 0.001F, testIntrinsics());
  require(result.valid, "valid ROI returns a measurement");
  requireNear(result.zMeters, 2.0F, 0.001F, "median rejects a depth outlier");
  requireNear(result.xMeters, 0.0F, 0.001F,
              "center pixel projects to camera center");
  requireNear(result.yMeters, 0.0F, 0.001F,
              "center pixel projects to camera center vertically");
  requireNear(result.rangeMeters, 2.0F, 0.001F,
              "center obstacle range matches optical depth");
  require(result.validSamples > 5, "sample count is reported");
}

void testOffAxisBoxIsDeprojected() {
  cv::Mat depth(100, 100, CV_16UC1, cv::Scalar(2000));
  const auto result = insight_cup::measureStereoRange(
      depth, cv::Rect(60, 40, 20, 20), 0.001F, testIntrinsics());
  require(result.valid, "off-axis box returns a measurement");
  requireNear(result.xMeters, 0.4F, 0.001F,
              "pixel location is converted to lateral position");
  requireNear(result.rangeMeters, std::sqrt(4.16F), 0.001F,
              "off-axis range accounts for lateral position");
}

void testInvalidDepthAndOutOfFrameBoxesAreRejected() {
  cv::Mat depth = cv::Mat::zeros(100, 100, CV_16UC1);
  const auto noDepth = insight_cup::measureStereoRange(
      depth, cv::Rect(40, 40, 20, 20), 0.001F, testIntrinsics());
  require(!noDepth.valid, "zero depth is not a valid measurement");
  require(noDepth.reason == "insufficient_valid_depth_samples",
          "failure reason is useful to a caller");

  const auto outside = insight_cup::measureStereoRange(
      depth, cv::Rect(120, 120, 20, 20), 0.001F, testIntrinsics());
  require(!outside.valid, "box outside the depth image is rejected");
  require(outside.reason == "box_outside_depth_frame",
          "out-of-frame failure is reported");
}

void testLowDepthCoverageIsNotReportedAsAValidRange() {
  cv::Mat depth = cv::Mat::zeros(100, 100, CV_16UC1);
  depth.at<std::uint16_t>(50, 50) = 2000;
  const auto result = insight_cup::measureStereoRange(
      depth, cv::Rect(40, 40, 20, 20), 0.001F, testIntrinsics(), 0.2F, 10.0F,
      25, 1, 0.05F);
  require(!result.valid, "sparse depth coverage is rejected");
  require(result.reason == "low_depth_coverage",
          "low coverage is distinguished from a fully invalid frame");
}

} // namespace

int main() {
  try {
    testMedianDepthAndThreeDimensionalRange();
    testOffAxisBoxIsDeprojected();
    testInvalidDepthAndOutOfFrameBoxesAreRejected();
    testLowDepthCoverageIsNotReportedAsAValidRange();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
