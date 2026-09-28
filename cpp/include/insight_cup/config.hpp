#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace insight_cup {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct RuntimeConfig {
  std::string source = "realsense";
  fs::path weights = "/home/j/trainv5/runs/yolo11n_trainv5_continue_latest-2/weights/best.onnx";
  fs::path outputRoot = fs::path(INSIGHT_CUP_PROJECT_ROOT) / "outputs/runtime";
  float confidence = 0.25F;
  float iou = 0.45F;
  int imageSize = 640;
  std::string yoloDevice = "auto";
  int yoloCpuThreads = 1;
  std::string provider = "auto";

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
  float knifeRetrievalThreshold = 0.50F;
  float knifeRetrievalMinMargin = 0.02F;
  int knifeCpuThreads = 1;

  int stage2RefreshFrames = 15;
  int stage2StableRefreshFrames = 30;
  float stage2CacheIou = 0.55F;
  int stage2Workers = 2;
  std::string stage2Smoothing = "vote";
  int stage2SmoothingWindow = 3;
  int stage2SwitchConfirmations = 2;

  std::optional<std::string> realsenseSerial;
  int cameraWidth = 1280;
  int cameraHeight = 720;
  int cameraFps = 30;
  int videoStride = 1;
  bool paceVideo = true;
  std::optional<int> maxFrames;

  std::string host = "127.0.0.1";
  int port = 7860;
  bool noUi = false;
  bool exitOnComplete = false;
  bool openBrowser = true;
  bool saveVideo = true;
  bool saveCrops = true;
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
bool isRealSenseSource(const std::string& value);
bool isVideoSource(const std::string& value);

}  // namespace insight_cup
