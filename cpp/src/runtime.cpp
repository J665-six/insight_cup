#include "insight_cup/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <openssl/evp.h>

#if INSIGHT_CUP_HAS_REALSENSE
#include <librealsense2/rs.hpp>
#endif

namespace insight_cup {

namespace {

using Clock = std::chrono::steady_clock;

const std::set<std::string> kImageExtensions = {
    ".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"};

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) { return std::tolower(character); });
  return value;
}

bool isImagePath(const fs::path& path) {
  return kImageExtensions.count(lower(path.extension().string())) != 0;
}

bool isCameraIndexSource(const std::string& source) {
  return !source.empty() &&
         std::all_of(source.begin(), source.end(), [](unsigned char value) {
           return std::isdigit(value);
         });
}

bool looksLikeLocalPath(const std::string& source) {
  if (source.empty()) {
    return false;
  }
  const fs::path path(source);
  return path.is_absolute() || source.find('/') != std::string::npos ||
         source.find('\\') != std::string::npos || isImagePath(path) ||
         isVideoSource(source);
}

void validateLocalSourcePath(const std::string& source) {
  if (isRealSenseSource(source) || isCameraIndexSource(source) ||
      !looksLikeLocalPath(source)) {
    return;
  }
  std::error_code error;
  const bool exists = fs::exists(source, error);
  if (error) {
    throw std::runtime_error("Could not inspect source path: " +
                             displayPath(source) + " (" + error.message() +
                             ")");
  }
  if (!exists) {
    throw std::runtime_error("Source path does not exist: " +
                             displayPath(source));
  }
}

json displayPathsInJson(const json& value) {
  if (value.is_string()) {
    const std::string text = value.get<std::string>();
    const fs::path path(text);
    return path.is_absolute() ? json(displayPath(path)) : value;
  }
  if (value.is_array()) {
    json result = json::array();
    for (const auto& item : value) {
      result.push_back(displayPathsInJson(item));
    }
    return result;
  }
  if (value.is_object()) {
    json result = json::object();
    for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
      result[iterator.key()] = displayPathsInJson(iterator.value());
    }
    return result;
  }
  return value;
}

double elapsedMs(const Clock::time_point& started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

std::string fixed(double value, int precision = 1) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(precision) << value;
  return output.str();
}

double median(std::deque<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::vector<double> sorted(values.begin(), values.end());
  const std::size_t middle = sorted.size() / 2;
  std::nth_element(sorted.begin(), sorted.begin() + middle, sorted.end());
  if (sorted.size() % 2 != 0) {
    return sorted[middle];
  }
  const double upper = sorted[middle];
  std::nth_element(sorted.begin(), sorted.begin() + middle - 1, sorted.end());
  return (sorted[middle - 1] + upper) * 0.5;
}

std::string defaultSessionName() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t value = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
  localtime_r(&value, &local);
  std::ostringstream output;
  output << std::put_time(&local, "%Y%m%d_%H%M%S");
  return output.str();
}

cv::Scalar classColor(const std::string& majorClass) {
  if (majorClass == "b0") {
    return {45, 156, 225};
  }
  if (majorClass == "f") {
    return {52, 166, 92};
  }
  return {67, 75, 218};
}

bool isAcceptedDecision(const Stage2Decision& decision) {
  return decision.predictedClass.has_value() &&
         (decision.status == "matched" || decision.status == "passthrough");
}

std::array<float, 4> boxValues(const Box& box) {
  return {static_cast<float>(box.x1), static_cast<float>(box.y1),
          static_cast<float>(box.x2), static_cast<float>(box.y2)};
}

Box clampBoxToFrame(const Box& box, const cv::Size& frameSize) {
  return {
      std::clamp(box.x1, 0, std::max(frameSize.width - 1, 0)),
      std::clamp(box.y1, 0, std::max(frameSize.height - 1, 0)),
      std::clamp(box.x2, 0, frameSize.width),
      std::clamp(box.y2, 0, frameSize.height),
  };
}

Box predictedBox(const Box& box, const std::array<float, 4>& velocity,
                 int frameGap, const cv::Size& frameSize) {
  const auto values = boxValues(box);
  Box predicted{
      static_cast<int>(std::nearbyint(values[0] + velocity[0] * frameGap)),
      static_cast<int>(std::nearbyint(values[1] + velocity[1] * frameGap)),
      static_cast<int>(std::nearbyint(values[2] + velocity[2] * frameGap)),
      static_cast<int>(std::nearbyint(values[3] + velocity[3] * frameGap)),
  };
  predicted = clampBoxToFrame(predicted, frameSize);
  return predicted.valid() ? predicted : box;
}

float newTrackMinConfidence(const std::string& majorClass) {
  if (majorClass == "f") {
    return 0.45F;
  }
  if (majorClass == "k") {
    return 0.35F;
  }
  return 0.0F;
}

int trackableTargetCount(const DetectionFrame& detectionFrame) {
  return static_cast<int>(
      std::count_if(detectionFrame.regions.begin(), detectionFrame.regions.end(),
                    [](const DetectedRegion& region) {
                      return region.handoff.majorClass != "b0";
                    }));
}

std::string sha256File(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Could not open shadow input for hashing: " +
                             displayPath(path));
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("Could not initialize SHA-256 for shadow input");
  }

  std::array<char, 64 * 1024> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = stream.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(),
                         static_cast<std::size_t>(count)) != 1) {
      throw std::runtime_error("Could not hash shadow input: " +
                               displayPath(path));
    }
  }
  if (!stream.eof()) {
    throw std::runtime_error("Could not read shadow input for hashing: " +
                             displayPath(path));
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digestSize = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digestSize) != 1) {
    throw std::runtime_error("Could not finish SHA-256 for shadow input");
  }
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digestSize; ++index) {
    encoded << std::setw(2) << static_cast<unsigned int>(digest[index]);
  }
  return encoded.str();
}

std::string captureId() {
  std::random_device entropy;
  std::ostringstream value;
  value << std::hex << std::setfill('0')
        << std::setw(16)
        << std::chrono::high_resolution_clock::now().time_since_epoch().count();
  for (int index = 0; index < 4; ++index) {
    value << std::setw(8) << entropy();
  }
  return value.str();
}

json shadowInputProvenance(const RuntimeConfig& config) {
  const fs::path source(config.source);
  std::error_code error;
  const bool cameraIndex =
      !config.source.empty() &&
      std::all_of(config.source.begin(), config.source.end(),
                  [](unsigned char value) { return std::isdigit(value); });
  if (!isRealSenseSource(config.source) && !cameraIndex &&
      fs::is_regular_file(source, error) && !error) {
    return {{"kind", "file"},
            {"sha256", sha256File(source)},
            {"video_stride", config.videoStride}};
  }
  return {{"kind", "capture"},
          {"capture_id", captureId()},
          {"video_stride", config.videoStride}};
}

}  // namespace

fs::path createSessionDirectory(const fs::path& outputRoot,
                                const std::optional<std::string>& sessionId) {
  const fs::path root = fs::absolute(outputRoot).lexically_normal();
  fs::create_directories(root);
  std::string normalized = sessionId ? *sessionId : defaultSessionName();
  normalized = std::regex_replace(normalized, std::regex("[^A-Za-z0-9._-]+"),
                                  "_");
  while (!normalized.empty() &&
         (normalized.front() == '.' || normalized.front() == '_' ||
          normalized.front() == '-')) {
    normalized.erase(normalized.begin());
  }
  while (!normalized.empty() &&
         (normalized.back() == '.' || normalized.back() == '_' ||
          normalized.back() == '-')) {
    normalized.pop_back();
  }
  if (normalized.empty()) {
    throw std::runtime_error("Session ID contains no usable characters");
  }
  fs::path candidate = root / normalized;
  int suffix = 1;
  while (fs::exists(candidate)) {
    std::ostringstream name;
    name << normalized << '_' << std::setfill('0') << std::setw(2) << suffix++;
    candidate = root / name.str();
  }
  fs::create_directories(candidate);
  return candidate;
}

RecognitionRuntime::RecognitionRuntime(RuntimeConfig config,
                                       fs::path sessionDirectory,
                                       RuntimeState& state,
                                       EventJournal& journal, Logger& logger)
    : config_(std::move(config)),
      sessionDirectory_(std::move(sessionDirectory)),
      state_(state),
      journal_(journal),
      logger_(logger) {
  // Keep OpenCV from creating a second thread pool on the small ARM board.
  // ONNX sessions are already explicitly bounded by the per-model settings.
  int cvThreads = 1;
  if (const char* value = std::getenv("INSIGHT_CUP_OPENCV_THREADS")) {
    try {
      cvThreads = std::max(1, std::stoi(value));
    } catch (...) {
      cvThreads = 1;
    }
  }
  cv::setNumThreads(cvThreads);
  if (config_.shadowLog) {
    shadowInput_ = shadowInputProvenance(config_);
    const fs::path path = sessionDirectory_ / "shadow.jsonl";
    shadowStream_.open(path, std::ios::binary | std::ios::trunc);
    if (!shadowStream_) {
      throw std::runtime_error("Could not open shadow journal: " +
                               displayPath(path));
    }
  }
}

RecognitionRuntime::~RecognitionRuntime() {
  if (scheduler_) {
    scheduler_->close();
  }
  if (shadowStream_.is_open()) {
    shadowStream_.close();
  }
}

void RecognitionRuntime::run() {
  writeManifest();
  try {
    config_.validate();
    validateLocalSourcePath(config_.source);
    state_.setStatus("loading");
    loadModels();
    writeManifest();
    if (state_.stopRequested()) {
      state_.setStatus("stopped");
    } else {
      state_.setStatus("running");
      if (config_.startPaused) {
        const auto [accepted, message] = state_.control("pause");
        if (!accepted) {
          throw std::runtime_error("Could not start runtime paused: " + message);
        }
      }
      runSource();
      drainStage2();
      state_.setStatus(state_.stopRequested() ? "stopped" : "completed");
    }
  } catch (const std::exception& error) {
    const std::string message = "RuntimeError: " + std::string(error.what());
    logger_.error("Runtime failed: " + message);
    state_.setStatus("error", message);
  }
  if (scheduler_) {
    scheduler_->close();
  }
  writeSummary();
  journal_.close();
  if (shadowStream_.is_open()) {
    shadowStream_.close();
  }
}

void RecognitionRuntime::loadModels() {
  state_.setModule("yolo", "loading", displayPath(config_.weights));
  logger_.info("LOAD module=yolo model=" + displayPath(config_.weights));
  auto started = Clock::now();
  detector_ = std::make_unique<YoloDetector>(config_);
  double loadMs = elapsedMs(started);
  modelMetadata_["yolo"] = detector_->metadata();
  const std::string yoloDevice = modelMetadata_["yolo"].value("device", "unknown");
  const std::string yoloBackend =
      modelMetadata_["yolo"].value("backend", "unknown");
  state_.setModule("yolo", "ready",
                   yoloDevice + " / " + fixed(loadMs, 0) + " ms");
  logger_.info("READY module=yolo backend=" + yoloBackend +
               " device=" + yoloDevice + " load_ms=" + fixed(loadMs));

  state_.setModule("face", "loading", displayPath(config_.faceGallery));
  logger_.info("LOAD module=face gallery=" + displayPath(config_.faceGallery) +
               " model=" + config_.faceModelName);
  started = Clock::now();
  face_ = std::make_shared<FaceRecognizer>(config_);
  loadMs = elapsedMs(started);
  modelMetadata_["face"] = face_->metadata();
  const std::string faceDevice = modelMetadata_["face"].value("device", "unknown");
  const std::string faceBackend =
      modelMetadata_["face"].value("backend", "unknown");
  state_.setModule("face", "ready",
                   faceDevice + " / " + std::to_string(face_->identityCount()) +
                       " identities / " +
                       fixed(loadMs, 0) + " ms");
  logger_.info("READY module=face backend=" + faceBackend +
               " device=" + faceDevice + " identities=" +
               std::to_string(face_->identityCount()) + " references=" +
               std::to_string(face_->referenceCount()) + " load_ms=" +
               fixed(loadMs));

  const fs::path knifeSource = config_.knifeMode == "classification"
                                   ? config_.knifeModelDir
                                   : config_.knifeRetrievalModel;
  state_.setModule("knife", "loading", displayPath(knifeSource));
  logger_.info("LOAD module=knife mode=" + config_.knifeMode +
               " model=" + displayPath(knifeSource));
  started = Clock::now();
  knife_ = std::make_shared<KnifeRecognizer>(config_);
  loadMs = elapsedMs(started);
  modelMetadata_["knife"] = knife_->metadata();
  const std::string knifeDevice =
      modelMetadata_["knife"].value("device", "unknown");
  const std::string knifeBackend =
      modelMetadata_["knife"].value("backend", "unknown");
  state_.setModule("knife", "ready",
                   knifeDevice + " / " + knife_->mode() + " / " +
                       std::to_string(knife_->classCount()) + " classes / " +
                       fixed(loadMs, 0) + " ms");
  logger_.info("READY module=knife backend=" + knifeBackend +
               " device=" + knifeDevice + " mode=" + knife_->mode() + " classes=" +
               std::to_string(knife_->classCount()) + " load_ms=" +
               fixed(loadMs));

  router_ = std::make_unique<RecognitionRouter>(face_, knife_, logger_);
  bool cameraIndex = !config_.source.empty() &&
                     std::all_of(config_.source.begin(), config_.source.end(),
                                 [](unsigned char value) { return std::isdigit(value); });
  const bool asynchronous = isRealSenseSource(config_.source) || cameraIndex ||
                            isVideoSource(config_.source);
  scheduler_ = std::make_unique<AsyncStage2Scheduler>(
      config_.stage2RefreshFrames, config_.stage2StableRefreshFrames,
      config_.stage2CacheIou, asynchronous, config_.stage2Workers,
      config_.stage2Smoothing, config_.stage2SmoothingWindow,
      config_.stage2SwitchConfirmations, config_.faceThreshold,
      config_.faceMinMargin, knife_->threshold(), knife_->minMargin());
  modelMetadata_["temporal"] = {
      {"method", config_.stage2Smoothing},
      {"window_size", config_.stage2SmoothingWindow},
      {"switch_confirmations", config_.stage2SwitchConfirmations},
      {"refresh_frames", config_.stage2RefreshFrames},
      {"stable_refresh_frames", config_.stage2StableRefreshFrames},
      {"scope", "per_track_id"},
  };
  logger_.info("READY module=temporal method=" + config_.stage2Smoothing +
               " window=" + std::to_string(config_.stage2SmoothingWindow) +
               " switch_confirmations=" +
               std::to_string(config_.stage2SwitchConfirmations) +
               " refresh_frames=" +
               std::to_string(config_.stage2RefreshFrames) +
               " stable_refresh_frames=" +
               std::to_string(config_.stage2StableRefreshFrames));
}

void RecognitionRuntime::runSource() {
  if (isRealSenseSource(config_.source)) {
    runRealSense();
    return;
  }
  const fs::path source(config_.source);
  if (fs::is_directory(source) ||
      (fs::is_regular_file(source) && isImagePath(source))) {
    runImages(source);
    return;
  }
  runCapture();
}

void RecognitionRuntime::runRealSense() {
#if INSIGHT_CUP_HAS_REALSENSE
  rs2::context context;
  const rs2::device_list devices = context.query_devices();
  if (devices.size() == 0) {
    throw std::runtime_error("No Intel RealSense camera was detected");
  }
  if (config_.realsenseSerial) {
    bool found = false;
    for (auto&& device : devices) {
      if (device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) ==
          *config_.realsenseSerial) {
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("Requested RealSense serial was not found: " +
                               *config_.realsenseSerial);
    }
  }
  rs2::pipeline pipeline(context);
  rs2::config cameraConfig;
  if (config_.realsenseSerial) {
    cameraConfig.enable_device(*config_.realsenseSerial);
  }
  cameraConfig.enable_stream(RS2_STREAM_COLOR, config_.cameraWidth,
                             config_.cameraHeight, RS2_FORMAT_BGR8,
                             config_.cameraFps);
  rs2::pipeline_profile profile;
  try {
    profile = pipeline.start(cameraConfig);
  } catch (const rs2::error& error) {
    throw std::runtime_error("Could not start RealSense color stream: " +
                             std::string(error.what()));
  }
  const rs2::device device = profile.get_device();
  const std::string serial = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
  const std::string model = device.get_info(RS2_CAMERA_INFO_NAME);
  const std::string firmware = device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION);
  state_.setSourceInfo({
      {"type", "realsense"},
      {"model", model},
      {"serial", serial},
      {"firmware", firmware},
      {"stream", "color"},
      {"format", "bgr8"},
      {"width", config_.cameraWidth},
      {"height", config_.cameraHeight},
      {"fps", config_.cameraFps},
      {"total_frames", 0},
      {"video_stride", config_.videoStride},
  });
  logger_.info("SOURCE type=realsense model=" + model + " serial=" + serial +
               " stream=color size=" + std::to_string(config_.cameraWidth) +
               "x" + std::to_string(config_.cameraHeight) + " fps=" +
               std::to_string(config_.cameraFps));

  cv::VideoWriter writer;
  int frameIndex = 0;
  int processedIndex = 0;
  try {
    while (state_.waitUntilRunnable()) {
      rs2::frameset frames = pipeline.wait_for_frames(5000);
      rs2::video_frame color = frames.get_color_frame();
      if (!color) {
        throw std::runtime_error("RealSense frameset has no color frame");
      }
      const int currentIndex = frameIndex++;
      if (currentIndex % config_.videoStride != 0) {
        continue;
      }
      cv::Mat wrapped(color.get_height(), color.get_width(), CV_8UC3,
                      const_cast<void*>(color.get_data()),
                      static_cast<std::size_t>(color.get_stride_in_bytes()));
      if (!writer.isOpened() && config_.saveVideo) {
        const fs::path output = sessionDirectory_ / "annotated.mp4";
        writer.open(output.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    std::max(1.0, static_cast<double>(config_.cameraFps) /
                                      config_.videoStride),
                    wrapped.size());
        if (!writer.isOpened()) {
          throw std::runtime_error("Could not write video: " +
                                   displayPath(output));
        }
        mediaOutputs_.push_back(displayPath(output));
      }
      // All model work and refresh-time crop copies finish before color leaves
      // this scope, so the RealSense buffer can be consumed without a full-frame copy.
      cv::Mat annotated = processFrame(
          wrapped, currentIndex, processedIndex, "realsense:" + serial,
          std::isfinite(color.get_timestamp())
              ? std::optional<double>(color.get_timestamp())
              : std::nullopt);
      if (writer.isOpened()) {
        writer.write(annotated);
      }
      ++processedIndex;
      if (config_.maxFrames && processedIndex >= *config_.maxFrames) {
        break;
      }
    }
  } catch (...) {
    pipeline.stop();
    writer.release();
    throw;
  }
  pipeline.stop();
  writer.release();
  if (processedIndex == 0 && !state_.stopRequested()) {
    throw std::runtime_error("RealSense camera produced no color frames");
  }
#else
  throw std::runtime_error(
      "This C++ build does not include Intel RealSense support");
#endif
}

void RecognitionRuntime::runImages(const fs::path& source) {
  std::vector<fs::path> images;
  if (fs::is_directory(source)) {
    for (const auto& entry : fs::recursive_directory_iterator(source)) {
      if (entry.is_regular_file() && isImagePath(entry.path())) {
        images.push_back(entry.path());
      }
    }
    std::sort(images.begin(), images.end());
  } else {
    images.push_back(source);
  }
  if (config_.maxFrames && images.size() > static_cast<std::size_t>(*config_.maxFrames)) {
    images.resize(*config_.maxFrames);
  }
  if (images.empty()) {
    throw std::runtime_error("No supported images found: " + displayPath(source));
  }
  state_.setSourceInfo({{"type", "images"}, {"total_frames", images.size()}});
  const fs::path outputDirectory = sessionDirectory_ / "annotated_images";
  if (config_.saveVideo) {
    fs::create_directories(outputDirectory);
  }
  int processedIndex = 0;
  for (std::size_t frameIndex = 0; frameIndex < images.size(); ++frameIndex) {
    if (!state_.waitUntilRunnable()) {
      break;
    }
    cv::Mat frame = cv::imread(images[frameIndex].string());
    if (frame.empty()) {
      logger_.warning("SKIP source=" + displayPath(images[frameIndex]) +
                      " reason=read_error");
      continue;
    }
    cv::Mat annotated = processFrame(frame, static_cast<int>(frameIndex),
                                     processedIndex,
                                     displayPath(images[frameIndex]),
                                     std::nullopt);
    if (config_.saveVideo) {
      if (annotated.empty()) {
        throw std::runtime_error("Annotated image output was not generated");
      }
      std::ostringstream filename;
      filename << std::setfill('0') << std::setw(6) << frameIndex << '_'
               << images[frameIndex].filename().string();
      const fs::path destination = outputDirectory / filename.str();
      if (!cv::imwrite(destination.string(), annotated)) {
        throw std::runtime_error("Could not write annotated image: " +
                                 displayPath(destination));
      }
      mediaOutputs_.push_back(displayPath(destination));
    }
    ++processedIndex;
  }
}

void RecognitionRuntime::runCapture() {
  const bool cameraIndex = isCameraIndexSource(config_.source);
  cv::VideoCapture capture;
  if (cameraIndex) {
    capture.open(std::stoi(config_.source));
  } else {
    capture.open(config_.source);
  }
  if (!capture.isOpened()) {
    throw std::runtime_error("Could not open source: " +
                             displayPath(config_.source));
  }
  double sourceFps = capture.get(cv::CAP_PROP_FPS);
  if (!std::isfinite(sourceFps) || sourceFps < 1.0 || sourceFps > 240.0) {
    sourceFps = 30.0;
  }
  const int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
  const bool paceVideo = !cameraIndex && config_.paceVideo;
  state_.setSourceInfo({
      {"type", cameraIndex ? "camera" : "video"},
      {"fps", std::round(sourceFps * 1000.0) / 1000.0},
      {"total_frames", std::max(totalFrames, 0)},
      {"video_stride", config_.videoStride},
      {"paced", paceVideo},
  });

  cv::VideoWriter writer;
  int rawIndex = 0;
  int processedIndex = 0;
  auto playbackStarted = Clock::now();
  while (state_.waitUntilRunnable()) {
    cv::Mat frame;
    if (!capture.read(frame)) {
      break;
    }
    if (rawIndex % config_.videoStride != 0) {
      ++rawIndex;
      continue;
    }
    if (paceVideo && rawIndex > 0) {
      const auto target = playbackStarted +
                          std::chrono::duration_cast<Clock::duration>(
                              std::chrono::duration<double>(rawIndex / sourceFps));
      const auto now = Clock::now();
      if (target > now) {
        std::this_thread::sleep_until(target);
      } else if (now - target > std::chrono::seconds(1)) {
        playbackStarted = now - std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(rawIndex /
                                                                  sourceFps));
      }
    }
    if (!writer.isOpened() && config_.saveVideo) {
      const fs::path output = sessionDirectory_ / "annotated.mp4";
      writer.open(output.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                  std::max(1.0, sourceFps / config_.videoStride), frame.size());
      if (!writer.isOpened()) {
        throw std::runtime_error("Could not write video: " + displayPath(output));
      }
      mediaOutputs_.push_back(displayPath(output));
    }
    const double positionMs = capture.get(cv::CAP_PROP_POS_MSEC);
    cv::Mat annotated = processFrame(
        frame, rawIndex, processedIndex, displayPath(config_.source),
        std::isfinite(positionMs) ? std::optional<double>(positionMs)
                                  : std::nullopt);
    if (writer.isOpened()) {
      writer.write(annotated);
    }
    ++rawIndex;
    ++processedIndex;
    if (config_.maxFrames && processedIndex >= *config_.maxFrames) {
      break;
    }
  }
  capture.release();
  writer.release();
  if (processedIndex == 0 && !state_.stopRequested()) {
    throw std::runtime_error("Source produced no frames: " +
                             displayPath(config_.source));
  }
}

std::optional<float> RecognitionRuntime::stage1AssociationScore(
    const Stage1Track& track, const DetectedRegion& region,
    const cv::Size& frameSize) const {
  if (track.region.handoff.majorClass != region.handoff.majorClass) {
    return std::nullopt;
  }
  const int frameGap =
      std::max(1, region.handoff.frameIndex - track.lastYoloFrame);
  const Box predicted =
      predictedBox(track.region.handoff.bbox, track.velocity, frameGap,
                   frameSize);
  const Box& current = region.handoff.bbox;
  const float overlap = bboxIou(predicted, current);
  const float previousOverlap = bboxIou(track.region.handoff.bbox, current);
  const float currentWidth = std::max(1, current.width());
  const float currentHeight = std::max(1, current.height());
  const float predictedWidth = std::max(1, predicted.width());
  const float predictedHeight = std::max(1, predicted.height());
  const float widthRatio =
      std::min(currentWidth, predictedWidth) /
      std::max(currentWidth, predictedWidth);
  const float heightRatio = std::min(currentHeight, predictedHeight) /
                            std::max(currentHeight, predictedHeight);
  if (widthRatio < 0.5F || heightRatio < 0.5F) {
    return std::nullopt;
  }
  const float currentX = (current.x1 + current.x2) * 0.5F;
  const float currentY = (current.y1 + current.y2) * 0.5F;
  const float predictedX = (predicted.x1 + predicted.x2) * 0.5F;
  const float predictedY = (predicted.y1 + predicted.y2) * 0.5F;
  const float centerDistance =
      std::hypot(currentX - predictedX, currentY - predictedY);
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

DetectionFrame RecognitionRuntime::trackYoloDetections(
    DetectionFrame detectionFrame, const cv::Size& frameSize, int frameIndex) {
  if (config_.yoloFrameStride <= 1) {
    return detectionFrame;
  }

  struct Pair {
    float score = 0.0F;
    std::size_t regionIndex = 0;
    std::size_t trackIndex = 0;
  };
  std::vector<Pair> pairs;
  for (std::size_t regionIndex = 0; regionIndex < detectionFrame.regions.size();
       ++regionIndex) {
    const auto& region = detectionFrame.regions[regionIndex];
    if (region.handoff.majorClass == "b0") {
      continue;
    }
    for (std::size_t trackIndex = 0; trackIndex < stage1Tracks_.size();
         ++trackIndex) {
      const auto score =
          stage1AssociationScore(stage1Tracks_[trackIndex], region, frameSize);
      if (score) {
        pairs.push_back({*score, regionIndex, trackIndex});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair& left, const Pair& right) {
    return left.score > right.score;
  });

  std::set<std::size_t> matchedRegions;
  std::set<std::size_t> matchedTracks;
  DetectionFrame filtered;
  filtered.inferenceMs = detectionFrame.inferenceMs;
  filtered.ignoredClasses = std::move(detectionFrame.ignoredClasses);
  for (const auto& pair : pairs) {
    if (matchedRegions.count(pair.regionIndex) != 0 ||
        matchedTracks.count(pair.trackIndex) != 0) {
      continue;
    }
    matchedRegions.insert(pair.regionIndex);
    matchedTracks.insert(pair.trackIndex);
    auto& track = stage1Tracks_[pair.trackIndex];
    DetectedRegion region = detectionFrame.regions[pair.regionIndex];
    const int frameGap = std::max(1, frameIndex - track.lastYoloFrame);
    const auto current = boxValues(region.handoff.bbox);
    const auto previous = boxValues(track.region.handoff.bbox);
    for (std::size_t index = 0; index < current.size(); ++index) {
      const float observed = (current[index] - previous[index]) / frameGap;
      track.velocity[index] =
          track.hits <= 1 ? observed : 0.70F * observed + 0.30F * track.velocity[index];
    }
    track.region = region;
    track.lastYoloFrame = frameIndex;
    track.missedYoloFrames = 0;
    ++track.hits;
    filtered.regions.push_back(std::move(region));
  }

  for (std::size_t regionIndex = 0; regionIndex < detectionFrame.regions.size();
       ++regionIndex) {
    if (matchedRegions.count(regionIndex) != 0) {
      continue;
    }
    auto region = detectionFrame.regions[regionIndex];
    if (region.handoff.majorClass == "b0") {
      filtered.regions.push_back(std::move(region));
      continue;
    }
    if (region.handoff.confidence <
        newTrackMinConfidence(region.handoff.majorClass)) {
      continue;
    }
    Stage1Track track;
    track.trackId = nextStage1TrackId_++;
    track.region = region;
    track.lastYoloFrame = frameIndex;
    stage1Tracks_.push_back(std::move(track));
    filtered.regions.push_back(std::move(region));
  }

  std::vector<Stage1Track> retained;
  retained.reserve(stage1Tracks_.size());
  for (std::size_t trackIndex = 0; trackIndex < stage1Tracks_.size();
       ++trackIndex) {
    auto track = std::move(stage1Tracks_[trackIndex]);
    if (matchedTracks.count(trackIndex) == 0 && track.lastYoloFrame != frameIndex) {
      ++track.missedYoloFrames;
    }
    if (track.missedYoloFrames <= 1) {
      retained.push_back(std::move(track));
    }
  }
  stage1Tracks_ = std::move(retained);
  return filtered;
}

DetectionFrame RecognitionRuntime::predictTrackedDetections(
    const cv::Size& frameSize, int frameIndex) const {
  DetectionFrame detectionFrame;
  int detectionIndex = 0;
  for (const auto& track : stage1Tracks_) {
    const int frameGap = frameIndex - track.lastYoloFrame;
    if (frameGap <= 0 || frameGap > std::max(1, config_.yoloFrameStride)) {
      continue;
    }
    DetectedRegion region = track.region;
    region.handoff.frameIndex = frameIndex;
    region.handoff.bbox =
        predictedBox(track.region.handoff.bbox, track.velocity, frameGap,
                     frameSize);
    if (!region.handoff.bbox.valid()) {
      continue;
    }
    region.handoff.confidence =
        std::max(0.01F, region.handoff.confidence * 0.98F);
    std::ostringstream id;
    id << "frame" << std::setfill('0') << std::setw(6) << frameIndex << ":trk"
       << std::setw(3) << detectionIndex++;
    region.handoff.detectionId = id.str();
    region.handoff.crop.release();
    region.shadowClassScores.reset();
    detectionFrame.regions.push_back(std::move(region));
  }
  return detectionFrame;
}

DetectionFrame RecognitionRuntime::stage1Frame(const cv::Mat& frame,
                                               int frameIndex,
                                               int processedIndex) {
  if (config_.yoloFrameStride <= 1) {
    return detector_->detect(frame, frameIndex);
  }
  const bool hasLiveTracks = std::any_of(
      stage1Tracks_.begin(), stage1Tracks_.end(), [&](const Stage1Track& track) {
        return frameIndex - track.lastYoloFrame <= config_.yoloFrameStride;
      });
  const bool strideDue =
      !lastFullYoloProcessedIndex_ ||
      processedIndex - *lastFullYoloProcessedIndex_ >= config_.yoloFrameStride;
  const bool denseScene =
      config_.yoloDenseForceThreshold > 0 &&
      lastFullYoloTargetCount_ >= config_.yoloDenseForceThreshold;
  const bool runYolo =
      forceNextYoloFrame_ || !hasLiveTracks || strideDue || denseScene;
  if (runYolo) {
    DetectionFrame detectionFrame = detector_->detect(frame, frameIndex);
    lastFullYoloProcessedIndex_ = processedIndex;
    forceNextYoloFrame_ = false;
    DetectionFrame tracked =
        trackYoloDetections(std::move(detectionFrame), frame.size(), frameIndex);
    lastFullYoloTargetCount_ = trackableTargetCount(tracked);
    return tracked;
  }
  DetectionFrame predicted = predictTrackedDetections(frame.size(), frameIndex);
  if (predicted.regions.empty()) {
    forceNextYoloFrame_ = true;
  }
  return predicted;
}

void RecognitionRuntime::requestYoloOnUnstableResult(
    const Stage2Resolution& resolution) {
  if (config_.yoloFrameStride <= 1 || resolution.decision.module == "passthrough") {
    return;
  }
  const bool stable = resolution.decision.status == "matched" &&
                      resolution.decision.predictedClass.has_value();
  if (resolution.hasNewResult() && !stable) {
    forceNextYoloFrame_ = true;
  }
}

cv::Mat RecognitionRuntime::processFrame(
    const cv::Mat& frame, int frameIndex, int processedIndex,
    const std::string& sourceLabel, std::optional<double> sourceTimeMs) {
  if (!detector_ || !router_ || !scheduler_) {
    throw std::runtime_error("Models are not loaded");
  }
  const bool competition = config_.profile == "competition";
  const bool needsAnnotatedFrame = config_.publishPreview || config_.saveVideo;
  Clock::time_point frameStarted{};
  if (!competition) {
    frameStarted = Clock::now();
  }
  DetectionFrame detectionFrame = stage1Frame(frame, frameIndex, processedIndex);
  cv::Mat annotated;
  if (needsAnnotatedFrame) {
    annotated = frame.clone();
  }
  double stage2TotalMs = 0.0;
  int recognitionCount = 0;
  for (const auto& rawName : detectionFrame.ignoredClasses) {
    logger_.warning("DROP frame=" + std::to_string(frameIndex) +
                    " raw_class=" + rawName + " reason=unsupported_class");
  }
  std::vector<PrepareJob> prepareJobs;
  prepareJobs.reserve(detectionFrame.regions.size());
  for (const auto& region : detectionFrame.regions) {
    state_.recordDetection(region.handoff.majorClass);
    prepareJobs.emplace_back([this, frame, detection = region.handoff]() {
      PreparedStage2Input prepared;
      prepared.crop =
          detection.majorClass == "k"
              ? squareContextCrop(frame, detection.bbox, 1.15F)
              : squareContextCrop(frame, detection.bbox, 1.30F);
      prepared.cropPath = saveCrop(detection, prepared.crop);
      return prepared;
    });
  }
  const auto resolutions = scheduler_->resolveFrame(
      detectionFrame.regions,
      [this](const Stage1Detection& detection) {
        return router_->route(detection);
      },
      prepareJobs);
  for (std::size_t index = 0; index < detectionFrame.regions.size(); ++index) {
    const auto& region = detectionFrame.regions[index];
    const auto& resolution = resolutions[index];
    requestYoloOnUnstableResult(resolution);
    if (!competition) {
      stage2TotalMs += resolution.inferenceMs;
    }
    if (resolution.hasNewResult()) {
      const json event = buildEvent(region, resolution, frame.size(), processedIndex,
                                    sourceLabel, sourceTimeMs,
                                    detectionFrame.inferenceMs);
      journal_.write(event);
      state_.recordEvent(event);
      writeShadowEvent(resolution);
    }
    if (isAcceptedDecision(resolution.decision)) {
      if (!annotated.empty()) {
        drawResult(annotated, region, resolution.decision);
      }
      if (!competition) {
        ++recognitionCount;
      }
    }
  }

  lastEventFrameSize_ = frame.size();
  lastProcessedIndex_ = processedIndex;
  lastSourceLabel_ = sourceLabel;
  lastSourceTimeMs_ = sourceTimeMs;

  if (competition) {
    state_.recordHeadlessFrame(frameIndex, processedIndex + 1);
    return annotated;
  }

  std::vector<unsigned char> encoded;
  if (config_.publishPreview) {
    cv::Mat preview = annotated;
    if (preview.cols > config_.previewWidth) {
      const int previewHeight = std::max(
          1, static_cast<int>(std::nearbyint(preview.rows *
                                             static_cast<double>(config_.previewWidth) /
                                             preview.cols)));
      cv::resize(annotated, preview,
                 cv::Size(config_.previewWidth, previewHeight), 0.0, 0.0,
                 cv::INTER_AREA);
    }
    if (!cv::imencode(".jpg", preview, encoded,
                      {cv::IMWRITE_JPEG_QUALITY, config_.jpegQuality})) {
      throw std::runtime_error("Could not encode the latest debug frame");
    }
  }
  const auto frameCompleted = Clock::now();
  const double totalMs = std::chrono::duration<double, std::milli>(
                             frameCompleted - frameStarted)
                             .count();
  if (lastFrameCompletedAt_) {
    frameIntervalsMs_.push_back(std::max(
        0.001, std::chrono::duration<double, std::milli>(frameCompleted -
                                                         *lastFrameCompletedAt_)
                   .count()));
    while (frameIntervalsMs_.size() > 30) {
      frameIntervalsMs_.pop_front();
    }
  }
  lastFrameCompletedAt_ = frameCompleted;
  const double intervalMedian = median(frameIntervalsMs_);
  const double fps = intervalMedian > 0.0 ? 1000.0 / intervalMedian : 0.0;
  const json timing = {
      {"yolo", std::round(detectionFrame.inferenceMs * 100.0) / 100.0},
      {"stage2", std::round(stage2TotalMs * 100.0) / 100.0},
      {"total", std::round(totalMs * 100.0) / 100.0},
  };
  state_.publishFrame(encoded, frameIndex, processedIndex + 1, fps, timing,
                      static_cast<int>(detectionFrame.regions.size()),
                      recognitionCount);
  if (config_.verbose &&
      (processedIndex + 1) % config_.frameLogEvery == 0) {
    logger_.debug("FRAME frame=" + std::to_string(frameIndex) +
                  " processed=" + std::to_string(processedIndex + 1) +
                  " detections=" +
                  std::to_string(detectionFrame.regions.size()) +
                  " yolo_ms=" + fixed(detectionFrame.inferenceMs) +
                  " stage2_ms=" + fixed(stage2TotalMs) +
                  " total_ms=" + fixed(totalMs) + " fps=" + fixed(fps, 2));
  }
  return annotated;
}

void RecognitionRuntime::drainStage2() {
  if (!scheduler_ || !scheduler_->asynchronous()) {
    return;
  }
  const auto started = Clock::now();
  const bool idle = scheduler_->waitForIdle(std::chrono::seconds(120));
  const auto results = idle ? scheduler_->finalizeResults()
                            : std::vector<FinalStage2Result>{};
  if (!idle) {
    logger_.warning("STAGE2 drain_timeout pending_results_not_written=true");
  }
  for (const auto& result : results) {
    writeFinalStage2Event(result);
  }
  logger_.info("STAGE2 drain_ms=" + fixed(elapsedMs(started)) +
               " completed=" + std::to_string(results.size()) +
               " idle=" + (idle ? "true" : "false"));
}

void RecognitionRuntime::writeFinalStage2Event(
    const FinalStage2Result& result) {
  if (lastEventFrameSize_.width <= 0 || lastEventFrameSize_.height <= 0 ||
      !lastProcessedIndex_ || !lastSourceLabel_) {
    return;
  }
  if (!result.resolution.evidence) {
    throw std::runtime_error(
        "Final stage-2 result is missing its source evidence");
  }
  const auto& evidence = *result.resolution.evidence;
  const DetectedRegion region{evidence.source, evidence.debug, nullptr};
  const json event = buildEvent(region, result.resolution, lastEventFrameSize_,
                                *lastProcessedIndex_, *lastSourceLabel_,
                                lastSourceTimeMs_, 0.0);
  journal_.write(event);
  state_.recordEvent(event);
  writeShadowEvent(result.resolution);
}

std::optional<std::string> RecognitionRuntime::saveCrop(
    const Stage1Detection& detection, const cv::Mat& crop) {
  if (!config_.saveCrops || detection.majorClass == "b0") {
    return std::nullopt;
  }
  std::string filename = detection.detectionId;
  std::replace(filename.begin(), filename.end(), ':', '_');
  filename += ".jpg";
  const fs::path relative = fs::path("crops") / detection.majorClass / filename;
  const fs::path destination = sessionDirectory_ / relative;
  fs::create_directories(destination.parent_path());
  if (!cv::imwrite(destination.string(), crop)) {
    logger_.warning("Could not save crop: " + displayPath(destination));
    return std::nullopt;
  }
  return relative.generic_string();
}

json RecognitionRuntime::buildEvent(
    const DetectedRegion& region, const Stage2Resolution& resolution,
    const cv::Size& frameSize, int processedIndex,
    const std::string& sourceLabel, std::optional<double> sourceTimeMs,
    double yoloMs) const {
  const auto& detection = region.handoff;
  const auto& decision = resolution.decision;
  json stage2 = decision.toJson();
  stage2["accepted"] = isAcceptedDecision(decision);
  return {
      {"schema", "recognition_event.v1"},
      {"id", detection.detectionId},
      {"timestamp", localTimeIso(true)},
      {"frame",
       {{"index", detection.frameIndex},
        {"processed_index", processedIndex},
        {"source", sourceLabel},
        {"source_time_ms",
         sourceTimeMs ? json(*sourceTimeMs) : json(nullptr)},
        {"image_shape", {frameSize.height, frameSize.width}}}},
      {"stage1", detection.handoffJson(resolution.sourceCropPath)},
      {"route",
       {{"module", decision.module},
        {"execution", resolution.mode},
        {"track_id",
         resolution.trackId ? json(*resolution.trackId) : json(nullptr)},
        {"cache_age_frames",
         resolution.cacheAgeFrames ? json(*resolution.cacheAgeFrames)
                                   : json(nullptr)},
        {"stage2_source_detection_id",
         resolution.sourceDetectionId ? json(*resolution.sourceDetectionId)
                                      : json(nullptr)},
        {"input_contract",
         {"id", "major_class", "confidence", "bbox_xyxy", "crop"}}}},
      {"stage2", std::move(stage2)},
      {"timing_ms",
       {{"yolo_frame", std::round(yoloMs * 1000.0) / 1000.0},
        {"stage2", std::round(resolution.inferenceMs * 1000.0) / 1000.0},
        {"detection_total",
         std::round((yoloMs + resolution.inferenceMs) * 1000.0) / 1000.0}}},
      {"debug", region.debug.toJson()},
  };
}

json RecognitionRuntime::buildShadowEvent(
    const Stage2Resolution& resolution) const {
  if (!resolution.evidence || !resolution.evidence->shadow) {
    throw std::runtime_error(
        "Fresh stage-2 result is missing source-aligned shadow evidence");
  }
  const Stage2Evidence& evidence = *resolution.evidence;
  const Stage2ShadowPayload& shadow = *evidence.shadow;
  const Stage1Detection& source = evidence.source;
  if (!shadow.yoloClassScores || shadow.yoloClassScores->empty()) {
    throw std::runtime_error(
        "Source detection is missing YOLO scores required by shadow logging");
  }

  json rawStage2 = shadow.rawDecision.toJson();
  rawStage2["class_scores"] = shadow.rawDecision.classScores;
  json decision = resolution.decision.toJson();
  decision["class_scores"] = resolution.decision.classScores;
  return {
      {"schema", "recognition_shadow.v1"},
      {"timestamp", localTimeIso(true)},
      {"input", shadowInput_},
      {"source",
       {{"id", source.detectionId},
        {"frame_index", source.frameIndex},
        {"major_class", source.majorClass},
        {"confidence", source.confidence},
        {"bbox_xyxy", source.bbox.toJson()},
        {"raw_yolo_class_id", evidence.debug.rawClassId},
        {"raw_yolo_class", evidence.debug.rawClassName},
        {"class_scores", *shadow.yoloClassScores}}},
      {"stage2_raw", std::move(rawStage2)},
      {"decision", std::move(decision)},
      {"route",
       {{"track_id",
         resolution.trackId ? json(*resolution.trackId) : json(nullptr)},
        {"execution", resolution.mode}}},
      {"timing_ms", {{"stage2", resolution.inferenceMs}}},
  };
}

void RecognitionRuntime::writeShadowEvent(
    const Stage2Resolution& resolution) {
  if (!config_.shadowLog || !resolution.evidence ||
      !resolution.evidence->shadow ||
      resolution.evidence->source.majorClass == "b0") {
    return;
  }
  shadowStream_ << buildShadowEvent(resolution).dump() << '\n';
  ++shadowRecordCount_;
  if (shadowRecordCount_ % 128 == 0) {
    shadowStream_.flush();
  }
  if (!shadowStream_) {
    throw std::runtime_error("Could not write source-aligned shadow evidence");
  }
}

void RecognitionRuntime::drawResult(cv::Mat& image,
                                    const DetectedRegion& region,
                                    const Stage2Decision& decision) const {
  const Box& box = region.handoff.bbox;
  const cv::Scalar color = classColor(region.handoff.majorClass);
  cv::rectangle(image, cv::Point(box.x1, box.y1),
                cv::Point(box.x2 - 1, box.y2 - 1), color, 2);
  std::string result;
  if (decision.status == "matched" || decision.status == "passthrough") {
    result = decision.predictedClass.value_or(region.handoff.majorClass);
  } else {
    result = decision.status;
  }
  std::ostringstream label;
  label << region.handoff.majorClass << '>' << result;
  if (decision.score) {
    label << ' ' << std::fixed << std::setprecision(2) << *decision.score;
  }
  if (config_.showRawLabel) {
    label << " [" << region.debug.rawClassName << ']';
  }
  const int font = cv::FONT_HERSHEY_SIMPLEX;
  const double scale = 0.55;
  const int thickness = 2;
  int baseline = 0;
  const cv::Size textSize =
      cv::getTextSize(label.str(), font, scale, thickness, &baseline);
  const int left = std::clamp(box.x1, 0,
                              std::max(image.cols - textSize.width - 10, 0));
  int bottom = std::max(textSize.height + baseline + 7, box.y1);
  bottom = std::min(bottom, image.rows - 1);
  const int top = std::max(0, bottom - textSize.height - baseline - 7);
  const int right = std::min(image.cols - 1, left + textSize.width + 10);
  cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color,
                cv::FILLED);
  cv::putText(image, label.str(),
              cv::Point(left + 5,
                        std::max(textSize.height + 1, bottom - baseline - 4)),
              font, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

void RecognitionRuntime::writeManifest() const {
  const json payload = {
      {"schema", "recognition_session.v1"},
      {"session_id", sessionDirectory_.filename().string()},
      {"created_at", state_.snapshot()["started_at"]},
      {"source", displayPath(config_.source)},
      {"session_dir", displayPath(sessionDirectory_)},
      {"config", config_.toJson()},
      {"models", displayPathsInJson(modelMetadata_)},
      {"data_flow",
       {{{"from", "source"}, {"to", "yolo"}, {"payload", "BGR frame"}},
        {{"from", "yolo"},
         {"to", "stage2 scheduler"},
         {"payload", {"id", "major_class", "confidence", "bbox_xyxy"}},
         {"excluded", {"raw_yolo_class", "raw_yolo_class_id"}}},
        {{"condition", "major_class == f or k and stage2 refresh is due"},
         {"from", "stage2 scheduler"},
         {"to", "router"},
         {"payload", {"id", "major_class", "confidence", "bbox_xyxy", "crop"}}},
        {{"condition", "major_class == b0"},
         {"from", "stage2 scheduler"},
         {"to", "router"},
         {"payload", {"id", "major_class", "confidence", "bbox_xyxy"}}},
        {{"condition", "major_class == f"},
         {"from", "router"},
         {"to", "InsightFace face recognition"}},
        {{"condition", "major_class == k"},
         {"from", "router"},
         {"to", "PaddleClas knife recognition"}},
        {{"condition", "major_class == b0"},
         {"from", "router"},
         {"to", "passthrough"}}}},
      {"outputs",
       {{"events", displayPath(sessionDirectory_ / "events.jsonl")},
        {"shadow",
         config_.shadowLog
             ? json(displayPath(sessionDirectory_ / "shadow.jsonl"))
             : json(nullptr)},
        {"summary", displayPath(sessionDirectory_ / "summary.json")},
        {"annotated_video",
         config_.saveVideo
             ? json(displayPath(sessionDirectory_ / "annotated.mp4"))
             : json(nullptr)},
        {"crops",
         config_.saveCrops ? json(displayPath(sessionDirectory_ / "crops"))
                           : json(nullptr)}}},
      {"implementation", "C++17"},
  };
  writeJson(sessionDirectory_ / "manifest.json", payload);
}

void RecognitionRuntime::writeSummary() const {
  const json snapshot = state_.snapshot();
  const json payload = {
      {"schema", "recognition_session_summary.v1"},
      {"session_id", sessionDirectory_.filename().string()},
      {"status", snapshot["status"]},
      {"source", snapshot["source"]},
      {"started_at", snapshot["started_at"]},
      {"ended_at", snapshot["ended_at"]},
      {"error", snapshot["error"]},
      {"processed_frames", snapshot["processed_frames"]},
      {"major_counts", snapshot["major_counts"]},
      {"status_counts", snapshot["status_counts"]},
      {"media_outputs", mediaOutputs_},
      {"events_jsonl", displayPath(sessionDirectory_ / "events.jsonl")},
      {"shadow_jsonl",
       config_.shadowLog
           ? json(displayPath(sessionDirectory_ / "shadow.jsonl"))
           : json(nullptr)},
      {"shadow_records", shadowRecordCount_},
      {"implementation", "C++17"},
  };
  writeJson(sessionDirectory_ / "summary.json", payload);
  logger_.info("END status=" + snapshot["status"].get<std::string>() +
               " frames=" + std::to_string(snapshot["processed_frames"].get<int>()) +
               " output=" + displayPath(sessionDirectory_));
}

}  // namespace insight_cup
