#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace insight_cup {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct RuntimeConfig {
  std::string profile = "custom";
  std::string source = "realsense";
  fs::path weights = fs::path(INSIGHT_CUP_PROJECT_ROOT) / "models/yolo/best.onnx";
  fs::path outputRoot = fs::path(INSIGHT_CUP_PROJECT_ROOT) / "outputs/runtime";
  float confidence = 0.25F;
  float iou = 0.45F;
  int imageSize = 640;
  std::string yoloDevice = "auto";
  int yoloCpuThreads = 1;
  int yoloFrameStride = 1;
  int yoloDenseForceThreshold = 0;
  std::string provider = "auto";
  fs::path rknnModelDir = fs::path(INSIGHT_CUP_PROJECT_ROOT) /
                          "models/rknn/rk3588_fp16";

  fs::path faceGallery = fs::path(INSIGHT_CUP_PROJECT_ROOT) /
                         "data/face_gallery/cpp/trainv5_gallery_r50.icg";
  fs::path faceModelRoot = fs::path(INSIGHT_CUP_PROJECT_ROOT) / "models/face";
  std::string faceModelName = "face_only_r50";
  float faceThreshold = 0.40F;
  float faceMinMargin = 0.03F;
  float faceDetectionThreshold = 0.18F;
  int faceDetectionSize = 320;
  int faceUpsampleMinSide = 320;
  int faceCpuThreads = 1;
  float faceRotationRetryDegrees = 0.0F;

  fs::path knifeModelDir = fs::path(INSIGHT_CUP_PROJECT_ROOT) /
                           "models/knife/pplcnetv2_base_knife10";
  std::string knifeMode = "classification";
  fs::path knifeRetrievalModel = fs::path(INSIGHT_CUP_PROJECT_ROOT) /
                                 "models/knife/ppshitu_v2_retrieval/inference.onnx";
  fs::path knifeGallery = fs::path(INSIGHT_CUP_PROJECT_ROOT) /
                          "data/knife_gallery/cpp/trainv5_ppshitu_v2.icg";
  float knifeThreshold = 0.50F;
  float knifeMinMargin = 0.15F;
  float knifeMinBoxAreaRatio = 0.0F;
  float knifeRetrievalThreshold = 0.50F;
  float knifeRetrievalMinMargin = 0.02F;
  fs::path knifeTraditionalTemplateDir =
      fs::path(INSIGHT_CUP_PROJECT_ROOT) /
      "data/knife_dataset/trainv5_grouped/images/train";
  float knifeTraditionalThreshold = 0.30F;
  float knifeTraditionalMinMargin = 0.0F;
  int knifeTraditionalMaxTemplatesPerClass = 240;
  int knifeCpuThreads = 1;

  int stage2RefreshFrames = 15;
  int stage2StableRefreshFrames = 30;
  float stage2CacheIou = 0.55F;
  // One worker per recognition class keeps CPU contention bounded while the
  // desktop is being used over NoMachine. Increase explicitly when headless.
  int stage2Workers = 1;
  std::string stage2Smoothing = "vote";
  int stage2SmoothingWindow = 3;
  int stage2SwitchConfirmations = 2;
  int inspectionStableFrames = 10;
  int stage1Confirmations = 2;

  // Stage-1 YOLO uses the explicit #1A polygon for obstacles and #7A for
  // person/knife. Stage 2 uses separate calibrated rectangle-area floors for
  // person and knife boxes (defined in runtime.cpp).
  bool stage1RoiEnabled = true;
  fs::path stage1RoiConfig = fs::path(INSIGHT_CUP_PROJECT_ROOT) /
                             "outputs/distance/stage1_roi.json";
  float stage2AreaTolerance = 0.30F;

  // Obstacle safety monitor. It reads the polygon calibration saved by the
  // distance tool and publishes the current front obstacle state on /safe.
  bool safeEnabled = true;
  fs::path safeZoneConfig = fs::path(INSIGHT_CUP_PROJECT_ROOT) /
                            "outputs/distance/obstacle_distance_new.json";
  // Kept for manifest/API compatibility; obstacle confirmation is immediate.
  float safeStableSeconds = 2.0F;
  float safeConfidence = 0.25F;

  std::optional<std::string> realsenseSerial;
  int cameraWidth = 1280;
  int cameraHeight = 720;
  // The D455 stream is shared with navigation. Keep the capture rate at the
  // navigation-safe 15 FPS unless a deployment explicitly requests another
  // rate.
  int cameraFps = 15;
  int videoStride = 1;
  bool paceVideo = true;
  std::optional<int> maxFrames;

  std::string host = "127.0.0.1";
  int port = 7860;
  bool noUi = false;
  bool startPaused = false;
  bool exitOnComplete = false;
  bool openBrowser = true;
  bool saveVideo = true;
  bool saveCrops = true;
  bool shadowLog = false;
  bool showRawLabel = false;
  bool publishPreview = true;
  int jpegQuality = 82;
  int previewWidth = 960;
  int frameLogEvery = 30;
  bool verbose = false;
  std::optional<std::string> sessionId;

  void validate() const;
  json toJson() const;
};

struct ParsedCommandLine {
  RuntimeConfig config;
  bool showHelp = false;
};

ParsedCommandLine parseCommandLine(int argc, char** argv);
std::string commandLineHelp();
std::string displayPath(const fs::path& path);
bool isRealSenseSource(const std::string& value);
bool isVideoSource(const std::string& value);

}  // namespace insight_cup
