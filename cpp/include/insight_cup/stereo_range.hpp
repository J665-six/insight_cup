#pragma once

#include <cstddef>
#include <string>

#include <librealsense2/rsutil.h>
#include <opencv2/core.hpp>

namespace insight_cup {

// Distances are camera-relative. x is right, y is down, and z is forward.
struct StereoRangeMeasurement {
  bool valid = false;
  std::string reason = "unavailable";
  float xMeters = 0.0F;
  float yMeters = 0.0F;
  float zMeters = 0.0F;
  float rangeMeters = 0.0F;
  std::size_t validSamples = 0;
  std::size_t totalSamples = 0;
  float validRatio = 0.0F;
};

// Measures the median aligned Z16 depth in the central part of a detection
// box, then deprojects its center with the calibrated RealSense intrinsics.
StereoRangeMeasurement
measureStereoRange(const cv::Mat &alignedDepth, const cv::Rect &detectionBox,
                   float depthUnitsInMeters, const rs2_intrinsics &intrinsics,
                   float minDepthMeters = 0.2F, float maxDepthMeters = 10.0F,
                   int maxSamplesPerAxis = 25, std::size_t minValidSamples = 5,
                   float minValidRatio = 0.05F);

} // namespace insight_cup
