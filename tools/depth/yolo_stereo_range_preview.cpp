#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "insight_cup/logging.hpp"
#include "insight_cup/models.hpp"
#include "insight_cup/stereo_range.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::optional<std::string> serial;
  insight_cup::fs::path weights =
      insight_cup::fs::path(INSIGHT_CUP_PROJECT_ROOT) / "models/yolo/best.onnx";
  std::string provider = "auto";
  float confidence = 0.25F;
  float iou = 0.45F;
  int imageSize = 640;
  int confirmations = 2;
  int width = 1280;
  int height = 720;
  int fps = 30;
  bool showRawLabel = true;
};

volatile std::sig_atomic_t stopped = 0;
void handleSignal(int) { stopped = 1; }

template <typename Value>
Value numberValue(const std::string &text, const std::string &name) {
  Value value{};
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
    throw std::runtime_error("Invalid " + name + ": " + text);
  }
  return value;
}

void printHelp() {
  std::cout
      << "YOLO stable-box + D455 stereo range preview\n\n"
      << "Usage: insight-cup-yolo-range-preview [options]\n\n"
      << "  --weights PATH       YOLO ONNX/RKNN model (default "
         "models/yolo/best.onnx)\n"
      << "  --provider NAME      auto, cpu, or rknn (default auto)\n"
      << "  --serial SERIAL      select a RealSense device by serial number\n"
      << "  --conf VALUE         YOLO confidence threshold (default 0.25)\n"
      << "  --iou VALUE          YOLO NMS threshold (default 0.45)\n"
      << "  --imgsz PIXELS       YOLO input size (default 640)\n"
      << "  --confirmations N    detections needed before range is shown "
         "(default 2)\n"
      << "  --width PIXELS       color width (default 1280)\n"
      << "  --height PIXELS      color height (default 720)\n"
      << "  --fps FPS            camera frame rate (default 30)\n"
      << "  --major-label-only   hide raw YOLO subclass label\n"
      << "  --help               show this help\n\n"
      << "This preview runs stage-1 YOLO only. Press q or Esc to quit.\n";
}

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto next = [&]() -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option);
      }
      return argv[++index];
    };
    if (option == "--help" || option == "-h") {
      printHelp();
      std::exit(0);
    }
    if (option == "--weights") {
      options.weights = next();
    } else if (option == "--provider") {
      options.provider = next();
    } else if (option == "--serial") {
      options.serial = next();
    } else if (option == "--conf") {
      options.confidence = numberValue<float>(next(), "confidence");
    } else if (option == "--iou") {
      options.iou = numberValue<float>(next(), "IoU threshold");
    } else if (option == "--imgsz") {
      options.imageSize = numberValue<int>(next(), "image size");
    } else if (option == "--confirmations") {
      options.confirmations = numberValue<int>(next(), "confirmations");
    } else if (option == "--width") {
      options.width = numberValue<int>(next(), "width");
    } else if (option == "--height") {
      options.height = numberValue<int>(next(), "height");
    } else if (option == "--fps") {
      options.fps = numberValue<int>(next(), "fps");
    } else if (option == "--major-label-only") {
      options.showRawLabel = false;
    } else {
      throw std::runtime_error("Unknown option: " + option);
    }
  }
  if (options.confidence <= 0.0F || options.confidence >= 1.0F ||
      options.iou <= 0.0F || options.iou >= 1.0F || options.imageSize <= 0 ||
      options.confirmations <= 0 || options.width <= 0 || options.height <= 0 ||
      options.fps <= 0) {
    throw std::runtime_error("Numeric options are outside their valid range");
  }
  if (options.provider != "auto" && options.provider != "cpu" &&
      options.provider != "rknn") {
    throw std::runtime_error("--provider must be auto, cpu, or rknn");
  }
  return options;
}

struct PreviewDetection {
  insight_cup::Box box;
  std::string majorClass;
  std::string rawClass;
  float confidence = 0.0F;
  int trackId = -1;
  int hits = 1;
  bool stable = false;
};

struct Stage1Track {
  int id = -1;
  insight_cup::DetectedRegion region;
  std::array<float, 4> velocity = {0.0F, 0.0F, 0.0F, 0.0F};
  int lastYoloFrame = 0;
  int hits = 1;
  int missedFrames = 0;
};

std::array<float, 4> boxValues(const insight_cup::Box &box) {
  return {static_cast<float>(box.x1), static_cast<float>(box.y1),
          static_cast<float>(box.x2), static_cast<float>(box.y2)};
}

insight_cup::Box predictedBox(const Stage1Track &track, int frameGap,
                              const cv::Size &frameSize) {
  const auto values = boxValues(track.region.handoff.bbox);
  insight_cup::Box box;
  box.x1 = static_cast<int>(
      std::nearbyint(values[0] + track.velocity[0] * frameGap));
  box.y1 = static_cast<int>(
      std::nearbyint(values[1] + track.velocity[1] * frameGap));
  box.x2 = static_cast<int>(
      std::nearbyint(values[2] + track.velocity[2] * frameGap));
  box.y2 = static_cast<int>(
      std::nearbyint(values[3] + track.velocity[3] * frameGap));
  box.x1 = std::clamp(box.x1, 0, std::max(0, frameSize.width - 1));
  box.y1 = std::clamp(box.y1, 0, std::max(0, frameSize.height - 1));
  box.x2 = std::clamp(box.x2, 0, frameSize.width);
  box.y2 = std::clamp(box.y2, 0, frameSize.height);
  return box;
}

std::optional<float> associationScore(const Stage1Track &track,
                                      const insight_cup::DetectedRegion &region,
                                      int frameIndex,
                                      const cv::Size &frameSize) {
  if (track.region.handoff.majorClass != region.handoff.majorClass) {
    return std::nullopt;
  }
  const int frameGap = std::max(1, frameIndex - track.lastYoloFrame);
  const insight_cup::Box predicted = predictedBox(track, frameGap, frameSize);
  const auto &previous = track.region.handoff.bbox;
  const auto &current = region.handoff.bbox;
  const float overlap = insight_cup::bboxIou(predicted, current);
  const float previousOverlap = insight_cup::bboxIou(previous, current);
  const float currentWidth = std::max(1, current.width());
  const float currentHeight = std::max(1, current.height());
  const float predictedWidth = std::max(1, predicted.width());
  const float predictedHeight = std::max(1, predicted.height());
  const float widthRatio = std::min(currentWidth, predictedWidth) /
                           std::max(currentWidth, predictedWidth);
  const float heightRatio = std::min(currentHeight, predictedHeight) /
                            std::max(currentHeight, predictedHeight);
  if (widthRatio < 0.5F || heightRatio < 0.5F) {
    return std::nullopt;
  }
  const float centerX = (current.x1 + current.x2) * 0.5F;
  const float centerY = (current.y1 + current.y2) * 0.5F;
  const float predictedX = (predicted.x1 + predicted.x2) * 0.5F;
  const float predictedY = (predicted.y1 + predicted.y2) * 0.5F;
  const float centerDistance =
      std::hypot(centerX - predictedX, centerY - predictedY);
  const float diagonal = std::max(std::hypot(currentWidth, currentHeight),
                                  std::hypot(predictedWidth, predictedHeight));
  const float distanceLimit =
      std::max(24.0F, diagonal * 0.50F) *
      std::min(2.0F, std::sqrt(static_cast<float>(frameGap)));
  if (std::max(overlap, previousOverlap) < 0.30F &&
      centerDistance > distanceLimit) {
    return std::nullopt;
  }
  const float normalizedDistance = centerDistance / std::max(diagonal, 1.0F);
  return overlap * 3.0F + previousOverlap +
         std::max(0.0F, 1.0F - normalizedDistance) +
         0.25F * (widthRatio + heightRatio);
}

class StableStage1Tracker {
public:
  explicit StableStage1Tracker(int confirmations)
      : confirmations_(confirmations) {}

  std::vector<PreviewDetection> update(const insight_cup::DetectionFrame &frame,
                                       const cv::Size &frameSize,
                                       int frameIndex) {
    struct Pair {
      float score = 0.0F;
      std::size_t regionIndex = 0;
      std::size_t trackIndex = 0;
    };
    std::vector<Pair> pairs;
    for (std::size_t regionIndex = 0; regionIndex < frame.regions.size();
         ++regionIndex) {
      const auto &region = frame.regions[regionIndex];
      if (region.handoff.majorClass != "b0" &&
          region.handoff.majorClass != "f" &&
          region.handoff.majorClass != "k") {
        continue;
      }
      for (std::size_t trackIndex = 0; trackIndex < tracks_.size();
           ++trackIndex) {
        const auto score = associationScore(tracks_[trackIndex], region,
                                            frameIndex, frameSize);
        if (score) {
          pairs.push_back({*score, regionIndex, trackIndex});
        }
      }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair &left, const Pair &right) {
                return left.score > right.score;
              });

    std::vector<bool> matchedRegions(frame.regions.size(), false);
    std::vector<bool> matchedTracks(tracks_.size(), false);
    std::vector<PreviewDetection> output;
    auto emit = [&](const insight_cup::DetectedRegion &region, int trackId,
                    int hits) {
      output.push_back(
          {region.handoff.bbox, region.handoff.majorClass,
           region.debug.rawClassName, region.handoff.confidence, trackId, hits,
           region.handoff.majorClass == "b0" || hits >= confirmations_});
    };
    for (const auto &pair : pairs) {
      if (matchedRegions[pair.regionIndex] || matchedTracks[pair.trackIndex]) {
        continue;
      }
      matchedRegions[pair.regionIndex] = true;
      matchedTracks[pair.trackIndex] = true;
      auto &track = tracks_[pair.trackIndex];
      auto region = frame.regions[pair.regionIndex];
      const int frameGap = std::max(1, frameIndex - track.lastYoloFrame);
      const auto current = boxValues(region.handoff.bbox);
      const auto previous = boxValues(track.region.handoff.bbox);
      for (std::size_t coordinate = 0; coordinate < current.size();
           ++coordinate) {
        const float observed =
            (current[coordinate] - previous[coordinate]) / frameGap;
        track.velocity[coordinate] =
            track.hits <= 1
                ? observed
                : 0.70F * observed + 0.30F * track.velocity[coordinate];
      }
      track.region = region;
      track.lastYoloFrame = frameIndex;
      track.missedFrames = 0;
      ++track.hits;
      emit(region, track.id, track.hits);
    }

    for (std::size_t index = 0; index < frame.regions.size(); ++index) {
      if (matchedRegions[index]) {
        continue;
      }
      const auto &region = frame.regions[index];
      if (region.handoff.majorClass != "b0" &&
          region.handoff.majorClass != "f" &&
          region.handoff.majorClass != "k") {
        continue;
      }
      Stage1Track track;
      track.id = nextTrackId_++;
      track.region = region;
      track.lastYoloFrame = frameIndex;
      tracks_.push_back(track);
      emit(region, track.id, track.hits);
    }

    std::vector<Stage1Track> retained;
    retained.reserve(tracks_.size());
    for (std::size_t index = 0; index < tracks_.size(); ++index) {
      auto track = std::move(tracks_[index]);
      if (index < matchedTracks.size() && !matchedTracks[index] &&
          track.lastYoloFrame != frameIndex) {
        ++track.missedFrames;
      }
      if (track.missedFrames <= 1) {
        retained.push_back(std::move(track));
      }
    }
    tracks_ = std::move(retained);
    return output;
  }

private:
  int confirmations_ = 2;
  int nextTrackId_ = 1;
  std::vector<Stage1Track> tracks_;
};

cv::Scalar classColor(const std::string &majorClass, bool stable) {
  if (!stable) {
    return {160, 160, 160};
  }
  return majorClass == "f" ? cv::Scalar(40, 210, 90) : cv::Scalar(230, 110, 55);
}

std::string labelText(const PreviewDetection &detection,
                      const insight_cup::StereoRangeMeasurement &measurement,
                      int confirmations, bool showRawLabel) {
  std::ostringstream label;
  if (!detection.stable) {
    label << "confirming " << detection.hits << '/' << confirmations;
  } else {
    if (showRawLabel && !detection.rawClass.empty()) {
      label << detection.rawClass;
    } else {
      label << detection.majorClass;
    }
    label << " " << std::fixed << std::setprecision(2) << detection.confidence;
    if (measurement.valid) {
      label << " | R " << measurement.rangeMeters << " m  Z "
            << measurement.zMeters << " m";
    } else {
      label << " | depth unavailable";
    }
  }
  return label.str();
}

void drawLabel(cv::Mat &image, const cv::Rect &box, const std::string &label,
               const cv::Scalar &color) {
  int baseline = 0;
  const cv::Size size =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.58, 2, &baseline);
  const int top = std::max(size.height + baseline + 4, box.y);
  cv::rectangle(
      image,
      cv::Rect(box.x, top - size.height - baseline - 6,
               std::min(size.width + 8, std::max(1, image.cols - box.x)),
               size.height + baseline + 8),
      color, cv::FILLED);
  cv::putText(image, label, cv::Point(box.x + 4, top - 5),
              cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 255, 255), 2,
              cv::LINE_AA);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    std::signal(SIGINT, handleSignal);

    insight_cup::RuntimeConfig detectorConfig;
    detectorConfig.weights = options.weights;
    detectorConfig.provider = options.provider;
    detectorConfig.confidence = options.confidence;
    detectorConfig.iou = options.iou;
    detectorConfig.imageSize = options.imageSize;
    detectorConfig.shadowLog = false;
    insight_cup::Logger logger;
    insight_cup::YoloDetector detector(detectorConfig);
    const auto model = detector.metadata();
    logger.info("YOLO preview model=" + model.value("weights", std::string()) +
                " backend=" + model.value("backend", std::string()));

    rs2::context context;
    const rs2::device_list devices = context.query_devices();
    if (devices.size() == 0) {
      throw std::runtime_error("No RealSense camera found");
    }
    if (options.serial) {
      bool found = false;
      for (auto &&device : devices) {
        if (device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) == *options.serial) {
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::runtime_error("RealSense serial not found: " +
                                 *options.serial);
      }
    }

    rs2::pipeline pipeline(context);
    rs2::config cameraConfig;
    if (options.serial) {
      cameraConfig.enable_device(*options.serial);
    }
    cameraConfig.enable_stream(RS2_STREAM_COLOR, options.width, options.height,
                               RS2_FORMAT_BGR8, options.fps);
    cameraConfig.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16,
                               options.fps);
    const rs2::pipeline_profile profile = pipeline.start(cameraConfig);
    const rs2::device device = profile.get_device();
    logger.info("Camera=" + std::string(device.get_info(RS2_CAMERA_INFO_NAME)) +
                " serial=" + device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));

    rs2::align alignDepthToColor(RS2_STREAM_COLOR);
    StableStage1Tracker tracker(options.confirmations);
    constexpr char windowName[] = "YOLO + D455 stereo range preview";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    auto previousFrame = Clock::now();
    double fps = 0.0;
    int frameIndex = 0;
    while (!stopped) {
      const rs2::frameset aligned =
          alignDepthToColor.process(pipeline.wait_for_frames(5000));
      const rs2::video_frame color = aligned.get_color_frame();
      const rs2::depth_frame depth = aligned.get_depth_frame();
      if (!color || !depth) {
        continue;
      }
      cv::Mat colorFrame(color.get_height(), color.get_width(), CV_8UC3,
                         const_cast<void *>(color.get_data()),
                         color.get_stride_in_bytes());
      cv::Mat depthFrame(depth.get_height(), depth.get_width(), CV_16UC1,
                         const_cast<void *>(depth.get_data()),
                         depth.get_stride_in_bytes());
      const auto intrinsics =
          depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
      const auto detections = detector.detect(colorFrame, frameIndex);
      const auto tracked =
          tracker.update(detections, colorFrame.size(), frameIndex);
      cv::Mat display = colorFrame.clone();
      for (const auto &detection : tracked) {
        const cv::Rect box =
            cv::Rect(detection.box.x1, detection.box.y1, detection.box.width(),
                     detection.box.height()) &
            cv::Rect(0, 0, colorFrame.cols, colorFrame.rows);
        if (box.area() <= 0) {
          continue;
        }
        insight_cup::StereoRangeMeasurement measurement;
        if (detection.stable) {
          measurement = insight_cup::measureStereoRange(
              depthFrame, box, depth.get_units(), intrinsics);
        }
        const cv::Scalar color =
            classColor(detection.majorClass, detection.stable);
        cv::rectangle(display, box, color, detection.stable ? 2 : 1);
        drawLabel(display, box,
                  labelText(detection, measurement, options.confirmations,
                            options.showRawLabel),
                  color);
      }
      const auto now = Clock::now();
      const double interval =
          std::chrono::duration<double>(now - previousFrame).count();
      if (interval > 0.0) {
        const double instantFps = 1.0 / interval;
        fps = fps <= 0.0 ? instantFps : 0.85 * fps + 0.15 * instantFps;
      }
      previousFrame = now;
      std::ostringstream status;
      status << "Stage 1 YOLO only | " << std::fixed << std::setprecision(1)
             << fps << " FPS | YOLO " << detections.inferenceMs << " ms | "
             << "stable after " << options.confirmations << " hits | q: quit";
      cv::putText(display, status.str(), cv::Point(12, 28),
                  cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2,
                  cv::LINE_AA);
      cv::imshow(windowName, display);
      const int key = cv::waitKey(1) & 0xff;
      if (key == 'q' || key == 27) {
        break;
      }
      ++frameIndex;
    }
    pipeline.stop();
    cv::destroyWindow(windowName);
    return 0;
  } catch (const rs2::error &error) {
    std::cerr << "RealSense error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
