#include "insight_cup/config.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace insight_cup {

namespace {

fs::path expandPath(const std::string& text) {
  std::string value = text;
  if (!value.empty() && value[0] == '~' &&
      (value.size() == 1 || value[1] == '/')) {
    if (const char* home = std::getenv("HOME")) {
      value = std::string(home) + value.substr(1);
    }
  }
  return fs::absolute(fs::path(value)).lexically_normal();
}

std::pair<std::string, std::optional<std::string>> splitArgument(
    const std::string& argument) {
  const std::size_t position = argument.find('=');
  if (position == std::string::npos) {
    return {argument, std::nullopt};
  }
  return {argument.substr(0, position), argument.substr(position + 1)};
}

int integerValue(const std::string& option, const std::string& value) {
  std::size_t consumed = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error(option + " requires an integer, got: " + value);
  }
  if (consumed != value.size()) {
    throw std::runtime_error(option + " requires an integer, got: " + value);
  }
  return parsed;
}

float floatValue(const std::string& option, const std::string& value) {
  std::size_t consumed = 0;
  float parsed = 0.0F;
  try {
    parsed = std::stof(value, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error(option + " requires a number, got: " + value);
  }
  if (consumed != value.size()) {
    throw std::runtime_error(option + " requires a number, got: " + value);
  }
  return parsed;
}

void requireChoice(const std::string& option, const std::string& value,
                   const std::unordered_set<std::string>& choices) {
  if (choices.count(value) == 0) {
    std::ostringstream message;
    message << option << " has unsupported value: " << value;
    throw std::runtime_error(message.str());
  }
}

void requireRange(const std::string& name, float value, float minimum,
                  float maximum, bool excludeMinimum = false) {
  const bool valid = excludeMinimum ? value > minimum : value >= minimum;
  if (!valid || value > maximum) {
    throw std::runtime_error(name + " is outside its supported range");
  }
}

}  // namespace

bool isRealSenseSource(const std::string& value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char character) { return std::tolower(character); });
  return normalized == "realsense" || normalized == "d455" ||
         normalized == "depth-camera" || normalized == "depth_camera";
}

bool isVideoSource(const std::string& value) {
  if (isRealSenseSource(value) ||
      (!value.empty() &&
       std::all_of(value.begin(), value.end(), [](unsigned char character) {
         return std::isdigit(character);
       }))) {
    return false;
  }
  static const std::unordered_set<std::string> extensions = {
      ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"};
  std::string extension = fs::path(value).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char character) { return std::tolower(character); });
  return extensions.count(extension) != 0;
}

void RuntimeConfig::validate() const {
  if (!fs::is_regular_file(weights)) {
    throw std::runtime_error("YOLO weights not found: " + weights.string());
  }
  if (weights.extension() != ".onnx") {
    throw std::runtime_error(
        "The native C++ runtime requires exported YOLO .onnx weights");
  }
  if (!fs::is_regular_file(faceGallery)) {
    throw std::runtime_error("Face gallery not found: " + faceGallery.string());
  }
  const fs::path faceModelDir = faceModelRoot / "models" / faceModelName;
  for (const auto& filename : {"det_10g.onnx", "w600k_r50.onnx"}) {
    if (!fs::is_regular_file(faceModelDir / filename)) {
      throw std::runtime_error("Face model not found: " +
                               (faceModelDir / filename).string());
    }
  }
  if (knifeMode == "classification") {
    if (!fs::is_regular_file(knifeModelDir / "inference.onnx") ||
        !fs::is_regular_file(knifeModelDir / "labels.txt")) {
      throw std::runtime_error("Knife classification model is incomplete: " +
                               knifeModelDir.string());
    }
  } else if (knifeMode == "retrieval") {
    if (!fs::is_regular_file(knifeRetrievalModel)) {
      throw std::runtime_error("Knife retrieval model not found: " +
                               knifeRetrievalModel.string());
    }
    if (!fs::is_regular_file(knifeGallery)) {
      throw std::runtime_error("Knife gallery not found: " +
                               knifeGallery.string());
    }
  } else {
    throw std::runtime_error("Knife mode must be classification or retrieval");
  }
  requireRange("YOLO confidence", confidence, 0.0F, 1.0F, true);
  requireRange("YOLO IoU", iou, 0.0F, 1.0F, true);
  requireRange("Face threshold", faceThreshold, 0.0F, 1.0F);
  requireRange("Face margin", faceMinMargin, 0.0F, 1.0F);
  requireRange("Face detector threshold", faceDetectionThreshold, 0.0F, 1.0F,
               true);
  requireRange("Knife threshold", knifeThreshold, 0.0F, 1.0F);
  requireRange("Knife margin", knifeMinMargin, 0.0F, 1.0F);
  requireRange("Knife retrieval threshold", knifeRetrievalThreshold, 0.0F,
               1.0F);
  requireRange("Knife retrieval margin", knifeRetrievalMinMargin, 0.0F, 1.0F);
  requireRange("Stage-2 cache IoU", stage2CacheIou, 0.0F, 1.0F, true);
  if (imageSize <= 0 || faceDetectionSize <= 0 || faceUpsampleMinSide < 0 ||
      yoloCpuThreads <= 0 || faceCpuThreads <= 0 ||
      knifeCpuThreads <= 0 || stage2RefreshFrames <= 0 ||
      stage2StableRefreshFrames < stage2RefreshFrames || stage2Workers <= 0 ||
      stage2SmoothingWindow <= 0 || stage2SwitchConfirmations <= 0 ||
      cameraWidth <= 0 || cameraHeight <= 0 || cameraFps <= 0 ||
      videoStride <= 0 || (maxFrames && *maxFrames <= 0) || port < 1 ||
      port > 65535 || jpegQuality < 1 || jpegQuality > 100 ||
      previewWidth <= 0 || frameLogEvery <= 0) {
    throw std::runtime_error("One or more positive runtime settings are invalid");
  }
  if (faceRotationRetryDegrees < 0.0F || faceRotationRetryDegrees > 45.0F) {
    throw std::runtime_error("Face rotation retry must be between 0 and 45 degrees");
  }
  requireChoice("--provider", provider, {"auto", "cpu", "cuda"});
  requireChoice("--stage2-smoothing", stage2Smoothing,
                {"none", "mean", "vote"});
}

json RuntimeConfig::toJson() const {
  return {
      {"source", source},
      {"weights", weights.string()},
      {"output_root", outputRoot.string()},
      {"confidence", confidence},
      {"iou", iou},
      {"image_size", imageSize},
      {"yolo_device", yoloDevice},
      {"yolo_cpu_threads", yoloCpuThreads},
      {"provider", provider},
      {"face_gallery", faceGallery.string()},
      {"face_model_root", faceModelRoot.string()},
      {"face_model_name", faceModelName},
      {"face_threshold", faceThreshold},
      {"face_min_margin", faceMinMargin},
      {"face_detection_threshold", faceDetectionThreshold},
      {"face_detection_size", faceDetectionSize},
      {"face_upsample_min_side", faceUpsampleMinSide},
      {"face_cpu_threads", faceCpuThreads},
      {"face_rotation_retry_degrees", faceRotationRetryDegrees},
      {"knife_model_dir", knifeModelDir.string()},
      {"knife_mode", knifeMode},
      {"knife_retrieval_model", knifeRetrievalModel.string()},
      {"knife_gallery", knifeGallery.string()},
      {"knife_threshold", knifeThreshold},
      {"knife_min_margin", knifeMinMargin},
      {"knife_retrieval_threshold", knifeRetrievalThreshold},
      {"knife_retrieval_min_margin", knifeRetrievalMinMargin},
      {"knife_cpu_threads", knifeCpuThreads},
      {"stage2_refresh_frames", stage2RefreshFrames},
      {"stage2_stable_refresh_frames", stage2StableRefreshFrames},
      {"stage2_cache_iou", stage2CacheIou},
      {"stage2_workers", stage2Workers},
      {"stage2_smoothing", stage2Smoothing},
      {"stage2_smoothing_window", stage2SmoothingWindow},
      {"stage2_switch_confirmations", stage2SwitchConfirmations},
      {"realsense_serial",
       realsenseSerial ? json(*realsenseSerial) : json(nullptr)},
      {"camera_width", cameraWidth},
      {"camera_height", cameraHeight},
      {"camera_fps", cameraFps},
      {"video_stride", videoStride},
      {"pace_video", paceVideo},
      {"max_frames", maxFrames ? json(*maxFrames) : json(nullptr)},
      {"save_video", saveVideo},
      {"save_crops", saveCrops},
      {"show_raw_label", showRawLabel},
      {"publish_preview", publishPreview},
      {"jpeg_quality", jpegQuality},
      {"preview_width", previewWidth},
      {"frame_log_every", frameLogEvery},
      {"runtime", "native_cpp17"},
  };
}

ParsedCommandLine parseCommandLine(int argc, char** argv) {
  ParsedCommandLine parsed;
  RuntimeConfig& config = parsed.config;

  for (int index = 1; index < argc; ++index) {
    auto [option, inlineValue] = splitArgument(argv[index]);
    auto next = [&]() -> std::string {
      if (inlineValue) {
        std::string value = *inlineValue;
        inlineValue.reset();
        return value;
      }
      if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option);
      }
      return argv[++index];
    };

    if (option == "-h" || option == "--help") {
      parsed.showHelp = true;
    } else if (option == "--source") {
      config.source = next();
    } else if (option == "--weights") {
      config.weights = expandPath(next());
    } else if (option == "--output-root") {
      config.outputRoot = expandPath(next());
    } else if (option == "--session-id") {
      config.sessionId = next();
    } else if (option == "--conf") {
      config.confidence = floatValue(option, next());
    } else if (option == "--iou") {
      config.iou = floatValue(option, next());
    } else if (option == "--imgsz") {
      config.imageSize = integerValue(option, next());
    } else if (option == "--device") {
      config.yoloDevice = next();
    } else if (option == "--yolo-cpu-threads") {
      config.yoloCpuThreads = integerValue(option, next());
    } else if (option == "--provider") {
      config.provider = next();
    } else if (option == "--face-gallery") {
      config.faceGallery = expandPath(next());
    } else if (option == "--face-model-root") {
      config.faceModelRoot = expandPath(next());
    } else if (option == "--face-model-name") {
      config.faceModelName = next();
    } else if (option == "--face-threshold") {
      config.faceThreshold = floatValue(option, next());
    } else if (option == "--face-min-margin") {
      config.faceMinMargin = floatValue(option, next());
    } else if (option == "--face-det-thresh") {
      config.faceDetectionThreshold = floatValue(option, next());
    } else if (option == "--face-det-size") {
      config.faceDetectionSize = integerValue(option, next());
    } else if (option == "--face-upsample-min-side") {
      config.faceUpsampleMinSide = integerValue(option, next());
    } else if (option == "--face-cpu-threads") {
      config.faceCpuThreads = integerValue(option, next());
    } else if (option == "--face-rotation-retry-degrees") {
      config.faceRotationRetryDegrees = floatValue(option, next());
    } else if (option == "--knife-model-dir") {
      config.knifeModelDir = expandPath(next());
    } else if (option == "--knife-mode") {
      config.knifeMode = next();
    } else if (option == "--knife-retrieval-model") {
      config.knifeRetrievalModel = expandPath(next());
    } else if (option == "--knife-gallery") {
      config.knifeGallery = expandPath(next());
    } else if (option == "--knife-threshold") {
      config.knifeThreshold = floatValue(option, next());
    } else if (option == "--knife-min-margin") {
      config.knifeMinMargin = floatValue(option, next());
    } else if (option == "--knife-retrieval-threshold") {
      config.knifeRetrievalThreshold = floatValue(option, next());
    } else if (option == "--knife-retrieval-min-margin") {
      config.knifeRetrievalMinMargin = floatValue(option, next());
    } else if (option == "--knife-cpu-threads") {
      config.knifeCpuThreads = integerValue(option, next());
    } else if (option == "--stage2-refresh-frames") {
      config.stage2RefreshFrames = integerValue(option, next());
    } else if (option == "--stage2-stable-refresh-frames") {
      config.stage2StableRefreshFrames = integerValue(option, next());
    } else if (option == "--stage2-cache-iou") {
      config.stage2CacheIou = floatValue(option, next());
    } else if (option == "--stage2-workers") {
      config.stage2Workers = integerValue(option, next());
    } else if (option == "--stage2-smoothing") {
      config.stage2Smoothing = next();
    } else if (option == "--stage2-smoothing-window") {
      config.stage2SmoothingWindow = integerValue(option, next());
    } else if (option == "--stage2-switch-confirmations") {
      config.stage2SwitchConfirmations = integerValue(option, next());
    } else if (option == "--realsense-serial") {
      config.realsenseSerial = next();
    } else if (option == "--camera-width") {
      config.cameraWidth = integerValue(option, next());
    } else if (option == "--camera-height") {
      config.cameraHeight = integerValue(option, next());
    } else if (option == "--camera-fps") {
      config.cameraFps = integerValue(option, next());
    } else if (option == "--vid-stride") {
      config.videoStride = integerValue(option, next());
    } else if (option == "--max-frames") {
      config.maxFrames = integerValue(option, next());
    } else if (option == "--host") {
      config.host = next();
    } else if (option == "--port") {
      config.port = integerValue(option, next());
    } else if (option == "--jpeg-quality") {
      config.jpegQuality = integerValue(option, next());
    } else if (option == "--preview-width") {
      config.previewWidth = integerValue(option, next());
    } else if (option == "--frame-log-every") {
      config.frameLogEvery = integerValue(option, next());
    } else if (option == "--ui") {
      config.noUi = false;
    } else if (option == "--no-ui") {
      config.noUi = true;
    } else if (option == "--exit-on-complete") {
      config.exitOnComplete = true;
    } else if (option == "--open-browser") {
      config.openBrowser = true;
    } else if (option == "--no-open-browser") {
      config.openBrowser = false;
    } else if (option == "--no-save-video") {
      config.saveVideo = false;
    } else if (option == "--no-save-crops") {
      config.saveCrops = false;
    } else if (option == "--show-raw-label") {
      config.showRawLabel = true;
    } else if (option == "--unthrottled-video") {
      config.paceVideo = false;
    } else if (option == "--verbose") {
      config.verbose = true;
    } else {
      throw std::runtime_error("Unknown option: " + option);
    }
  }

  if (!isRealSenseSource(config.source) &&
      !std::all_of(config.source.begin(), config.source.end(),
                   [](unsigned char character) { return std::isdigit(character); })) {
    const fs::path candidate = expandPath(config.source);
    if (fs::exists(candidate)) {
      config.source = candidate.string();
    }
  }
  config.publishPreview = !config.noUi;
  return parsed;
}

std::string commandLineHelp() {
  return R"HELP(insight-cup - native C++ YOLO, InsightFace, and PaddleClas runtime

Usage:
  ./start.sh [options]

Source and output:
  --source VALUE                 realsense, camera index, video, image, or directory
  --session-id NAME              output session name
  --output-root PATH             session output root
  --max-frames N                 stop after N processed frames
  --vid-stride N                 process one frame in every N source frames
  --unthrottled-video            process videos without source-FPS pacing

Models:
  --weights PATH                 YOLO ONNX model
  --knife-mode MODE              classification or retrieval
  --provider MODE                auto or cpu (this build is CPU-only)
  --conf VALUE                   YOLO confidence threshold (default 0.25)
  --iou VALUE                    YOLO NMS IoU (default 0.45)

Runtime:
  --ui | --no-ui                 enable or disable the debug web UI
  --host HOST --port PORT        debug UI address (default 127.0.0.1:7860)
  --no-open-browser              do not launch the browser
  --no-save-video                skip annotated video output
  --no-save-crops                skip stage-2 crop output
  --verbose                      print per-frame diagnostics
  -h, --help                     show this help

The remaining Python-compatible threshold, thread, camera, and temporal options
are also accepted. See README.md for the complete command reference.
)HELP";
}

}  // namespace insight_cup
