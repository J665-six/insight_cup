#include "insight_cup/stereo_range.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace insight_cup {

StereoRangeMeasurement
measureStereoRange(const cv::Mat &alignedDepth, const cv::Rect &detectionBox,
                   float depthUnitsInMeters, const rs2_intrinsics &intrinsics,
                   float minDepthMeters, float maxDepthMeters,
                   int maxSamplesPerAxis, std::size_t minValidSamples,
                   float minValidRatio) {
  StereoRangeMeasurement result;
  if (alignedDepth.empty() || alignedDepth.type() != CV_16UC1) {
    result.reason = "depth_frame_must_be_z16";
    return result;
  }
  if (depthUnitsInMeters <= 0.0F || !std::isfinite(depthUnitsInMeters)) {
    result.reason = "invalid_depth_scale";
    return result;
  }
  if (intrinsics.fx <= 0.0F || intrinsics.fy <= 0.0F ||
      !std::isfinite(intrinsics.fx) || !std::isfinite(intrinsics.fy)) {
    result.reason = "invalid_camera_intrinsics";
    return result;
  }
  if (maxSamplesPerAxis <= 0 || minValidSamples == 0 ||
      !std::isfinite(minDepthMeters) || !std::isfinite(maxDepthMeters) ||
      minDepthMeters < 0.0F || maxDepthMeters <= minDepthMeters ||
      !std::isfinite(minValidRatio) || minValidRatio < 0.0F ||
      minValidRatio > 1.0F) {
    result.reason = "invalid_measurement_settings";
    return result;
  }

  const cv::Rect imageBounds(0, 0, alignedDepth.cols, alignedDepth.rows);
  const cv::Rect clipped = detectionBox & imageBounds;
  if (clipped.width <= 0 || clipped.height <= 0) {
    result.reason = "box_outside_depth_frame";
    return result;
  }

  // Ignore the border of the detection, where boxes commonly include
  // background. The center remains the projection point of the detection.
  const int insetX = static_cast<int>(std::floor(clipped.width * 0.2));
  const int insetY = static_cast<int>(std::floor(clipped.height * 0.2));
  const cv::Rect sampleArea(clipped.x + insetX, clipped.y + insetY,
                            std::max(1, clipped.width - 2 * insetX),
                            std::max(1, clipped.height - 2 * insetY));
  const int stepX = std::max(
      1, static_cast<int>(std::ceil(static_cast<float>(sampleArea.width) /
                                    maxSamplesPerAxis)));
  const int stepY = std::max(
      1, static_cast<int>(std::ceil(static_cast<float>(sampleArea.height) /
                                    maxSamplesPerAxis)));

  std::vector<float> depths;
  depths.reserve(static_cast<std::size_t>(maxSamplesPerAxis) *
                 static_cast<std::size_t>(maxSamplesPerAxis));
  for (int y = sampleArea.y; y < sampleArea.y + sampleArea.height; y += stepY) {
    const auto *row = alignedDepth.ptr<std::uint16_t>(y);
    for (int x = sampleArea.x; x < sampleArea.x + sampleArea.width;
         x += stepX) {
      ++result.totalSamples;
      const float meters = row[x] * depthUnitsInMeters;
      if (std::isfinite(meters) && meters >= minDepthMeters &&
          meters <= maxDepthMeters) {
        depths.push_back(meters);
      }
    }
  }
  result.validSamples = depths.size();
  result.validRatio = result.totalSamples == 0
                          ? 0.0F
                          : static_cast<float>(result.validSamples) /
                                static_cast<float>(result.totalSamples);
  if (depths.size() < minValidSamples || result.validRatio < minValidRatio) {
    result.reason = depths.size() < minValidSamples
                        ? "insufficient_valid_depth_samples"
                        : "low_depth_coverage";
    return result;
  }

  const std::size_t middle = depths.size() / 2;
  std::nth_element(depths.begin(), depths.begin() + middle, depths.end());
  float medianDepth = depths[middle];
  if (depths.size() % 2 == 0) {
    const float upper = medianDepth;
    std::nth_element(depths.begin(), depths.begin() + middle - 1, depths.end());
    medianDepth = (depths[middle - 1] + upper) * 0.5F;
  }

  const float pixel[2] = {
      (static_cast<float>(clipped.x) + clipped.width * 0.5F),
      (static_cast<float>(clipped.y) + clipped.height * 0.5F)};
  float point[3] = {0.0F, 0.0F, 0.0F};
  rs2_deproject_pixel_to_point(point, &intrinsics, pixel, medianDepth);
  result.xMeters = point[0];
  result.yMeters = point[1];
  result.zMeters = point[2];
  result.rangeMeters = std::sqrt(point[0] * point[0] + point[1] * point[1] +
                                 point[2] * point[2]);
  if (!std::isfinite(result.rangeMeters)) {
    result.reason = "invalid_deprojected_point";
    return result;
  }
  result.valid = true;
  result.reason = "ok";
  return result;
}

} // namespace insight_cup
