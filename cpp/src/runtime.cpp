#include "insight_cup/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
  cv::setNumThreads(4);
}

RecognitionRuntime::~RecognitionRuntime() {
  if (scheduler_) {
    scheduler_->close();
  }
}

void RecognitionRuntime::run() {
  writeManifest();
  try {
    config_.validate();
    state_.setStatus("loading");
    loadModels();
    writeManifest();
    if (state_.stopRequested()) {
      state_.setStatus("stopped");
    } else {
      state_.setStatus("running");
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
}

void RecognitionRuntime::loadModels() {
  state_.setModule("yolo", "loading", config_.weights.string());
  logger_.info("LOAD module=yolo model=" + config_.weights.string());
  auto started = Clock::now();
  detector_ = std::make_unique<YoloDetector>(config_);
  double loadMs = elapsedMs(started);
  modelMetadata_["yolo"] = detector_->metadata();
  state_.setModule("yolo", "ready", fixed(loadMs, 0) + " ms");
  logger_.info("READY module=yolo load_ms=" + fixed(loadMs));

  state_.setModule("face", "loading", config_.faceGallery.string());
  logger_.info("LOAD module=face gallery=" + config_.faceGallery.string() +
               " model=" + config_.faceModelName);
  started = Clock::now();
  face_ = std::make_shared<FaceRecognizer>(config_);
  loadMs = elapsedMs(started);
  modelMetadata_["face"] = face_->metadata();
  state_.setModule("face", "ready",
                   std::to_string(face_->identityCount()) + " identities / " +
                       fixed(loadMs, 0) + " ms");
  logger_.info("READY module=face identities=" +
               std::to_string(face_->identityCount()) + " references=" +
               std::to_string(face_->referenceCount()) + " load_ms=" +
               fixed(loadMs));

  const fs::path knifeSource = config_.knifeMode == "classification"
                                   ? config_.knifeModelDir
                                   : config_.knifeRetrievalModel;
  state_.setModule("knife", "loading", knifeSource.string());
  logger_.info("LOAD module=knife mode=" + config_.knifeMode +
               " model=" + knifeSource.string());
  started = Clock::now();
  knife_ = std::make_shared<KnifeRecognizer>(config_);
  loadMs = elapsedMs(started);
  modelMetadata_["knife"] = knife_->metadata();
  state_.setModule("knife", "ready",
                   knife_->mode() + " / " +
                       std::to_string(knife_->classCount()) + " classes / " +
                       fixed(loadMs, 0) + " ms");
  logger_.info("READY module=knife mode=" + knife_->mode() + " classes=" +
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
      cv::Mat frame = wrapped.clone();
      if (!writer.isOpened() && config_.saveVideo) {
        const fs::path output = sessionDirectory_ / "annotated.mp4";
        writer.open(output.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    std::max(1.0, static_cast<double>(config_.cameraFps) /
                                      config_.videoStride),
                    frame.size());
        if (!writer.isOpened()) {
          throw std::runtime_error("Could not write video: " + output.string());
        }
        mediaOutputs_.push_back(output.string());
      }
      cv::Mat annotated = processFrame(
          frame, currentIndex, processedIndex, "realsense:" + serial,
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
    throw std::runtime_error("No supported images found: " + source.string());
  }
  state_.setSourceInfo({{"type", "images"}, {"total_frames", images.size()}});
  const fs::path outputDirectory = sessionDirectory_ / "annotated_images";
  fs::create_directories(outputDirectory);
  int processedIndex = 0;
  for (std::size_t frameIndex = 0; frameIndex < images.size(); ++frameIndex) {
    if (!state_.waitUntilRunnable()) {
      break;
    }
    cv::Mat frame = cv::imread(images[frameIndex].string());
    if (frame.empty()) {
      logger_.warning("SKIP source=" + images[frameIndex].string() +
                      " reason=read_error");
      continue;
    }
    cv::Mat annotated = processFrame(frame, static_cast<int>(frameIndex),
                                     processedIndex, images[frameIndex].string(),
                                     std::nullopt);
    std::ostringstream filename;
    filename << std::setfill('0') << std::setw(6) << frameIndex << '_'
             << images[frameIndex].filename().string();
    const fs::path destination = outputDirectory / filename.str();
    if (!cv::imwrite(destination.string(), annotated)) {
      throw std::runtime_error("Could not write annotated image: " +
                               destination.string());
    }
    mediaOutputs_.push_back(destination.string());
    ++processedIndex;
  }
}

void RecognitionRuntime::runCapture() {
  const bool cameraIndex = !config_.source.empty() &&
                           std::all_of(config_.source.begin(), config_.source.end(),
                                       [](unsigned char value) {
                                         return std::isdigit(value);
                                       });
  cv::VideoCapture capture;
  if (cameraIndex) {
    capture.open(std::stoi(config_.source));
  } else {
    capture.open(config_.source);
  }
  if (!capture.isOpened()) {
    throw std::runtime_error("Could not open source: " + config_.source);
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
        throw std::runtime_error("Could not write video: " + output.string());
      }
      mediaOutputs_.push_back(output.string());
    }
    const double positionMs = capture.get(cv::CAP_PROP_POS_MSEC);
    cv::Mat annotated = processFrame(
        frame, rawIndex, processedIndex, config_.source,
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
    throw std::runtime_error("Source produced no frames: " + config_.source);
  }
}

cv::Mat RecognitionRuntime::processFrame(
    const cv::Mat& frame, int frameIndex, int processedIndex,
    const std::string& sourceLabel, std::optional<double> sourceTimeMs) {
  if (!detector_ || !router_ || !scheduler_) {
    throw std::runtime_error("Models are not loaded");
  }
  const auto frameStarted = Clock::now();
  DetectionFrame detectionFrame = detector_->detect(frame, frameIndex);
  cv::Mat annotated = frame.clone();
  double stage2TotalMs = 0.0;
  for (const auto& rawName : detectionFrame.ignoredClasses) {
    logger_.warning("DROP frame=" + std::to_string(frameIndex) +
                    " raw_class=" + rawName + " reason=unsupported_class");
  }
  std::vector<Stage1Detection> detections;
  std::vector<PrepareJob> prepareJobs;
  detections.reserve(detectionFrame.regions.size());
  prepareJobs.reserve(detectionFrame.regions.size());
  for (const auto& region : detectionFrame.regions) {
    state_.recordDetection(region.handoff.majorClass);
    detections.push_back(region.handoff);
    prepareJobs.emplace_back([this, region]() { return saveCrop(region); });
  }
  const auto resolutions = scheduler_->resolveFrame(
      detections,
      [this](const Stage1Detection& detection) {
        return router_->route(detection);
      },
      prepareJobs);
  for (std::size_t index = 0; index < detectionFrame.regions.size(); ++index) {
    const auto& region = detectionFrame.regions[index];
    const auto& resolution = resolutions[index];
    if (resolution.trackId) {
      trackDebug_[*resolution.trackId] = region.debug;
    }
    stage2TotalMs += resolution.inferenceMs;
    if (resolution.hasNewResult()) {
      const json event = buildEvent(region, resolution, frame, processedIndex,
                                    sourceLabel, sourceTimeMs,
                                    detectionFrame.inferenceMs);
      journal_.write(event);
      state_.recordEvent(event);
    }
    drawResult(annotated, region, resolution.decision);
  }

  lastEventFrame_ = frame;
  lastProcessedIndex_ = processedIndex;
  lastSourceLabel_ = sourceLabel;
  lastSourceTimeMs_ = sourceTimeMs;

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
                      static_cast<int>(detectionFrame.regions.size()));
  if ((processedIndex + 1) % config_.frameLogEvery == 0) {
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
  if (lastEventFrame_.empty() || !lastProcessedIndex_ || !lastSourceLabel_) {
    return;
  }
  YoloDebugInfo debug{-1, result.detection.majorClass};
  if (result.resolution.trackId) {
    const auto found = trackDebug_.find(*result.resolution.trackId);
    if (found != trackDebug_.end()) {
      debug = found->second;
    }
  }
  const DetectedRegion region{result.detection, debug};
  const json event = buildEvent(region, result.resolution, lastEventFrame_,
                                *lastProcessedIndex_, *lastSourceLabel_,
                                lastSourceTimeMs_, 0.0);
  journal_.write(event);
  state_.recordEvent(event);
}

std::optional<std::string> RecognitionRuntime::saveCrop(
    const DetectedRegion& region) {
  if (!config_.saveCrops || region.handoff.majorClass == "b0") {
    return std::nullopt;
  }
  std::string filename = region.handoff.detectionId;
  std::replace(filename.begin(), filename.end(), ':', '_');
  filename += ".jpg";
  const fs::path relative = fs::path("crops") / region.handoff.majorClass / filename;
  const fs::path destination = sessionDirectory_ / relative;
  fs::create_directories(destination.parent_path());
  if (!cv::imwrite(destination.string(), region.handoff.crop)) {
    logger_.warning("Could not save crop: " + destination.string());
    return std::nullopt;
  }
  return relative.generic_string();
}

json RecognitionRuntime::buildEvent(
    const DetectedRegion& region, const Stage2Resolution& resolution,
    const cv::Mat& frame, int processedIndex,
    const std::string& sourceLabel, std::optional<double> sourceTimeMs,
    double yoloMs) const {
  const auto& detection = region.handoff;
  const auto& decision = resolution.decision;
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
        {"image_shape", {frame.rows, frame.cols}}}},
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
      {"stage2", decision.toJson()},
      {"timing_ms",
       {{"yolo_frame", std::round(yoloMs * 1000.0) / 1000.0},
        {"stage2", std::round(resolution.inferenceMs * 1000.0) / 1000.0},
        {"detection_total",
         std::round((yoloMs + resolution.inferenceMs) * 1000.0) / 1000.0}}},
      {"debug", region.debug.toJson()},
  };
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
      {"source", config_.source},
      {"session_dir", sessionDirectory_.string()},
      {"config", config_.toJson()},
      {"models", modelMetadata_},
      {"data_flow",
       {{{"from", "source"}, {"to", "yolo"}, {"payload", "BGR frame"}},
        {{"from", "yolo"},
         {"to", "router"},
         {"payload", {"id", "major_class", "confidence", "bbox_xyxy", "crop"}},
         {"excluded", {"raw_yolo_class", "raw_yolo_class_id"}}},
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
       {{"events", (sessionDirectory_ / "events.jsonl").string()},
        {"summary", (sessionDirectory_ / "summary.json").string()},
        {"annotated_video", (sessionDirectory_ / "annotated.mp4").string()},
        {"crops", (sessionDirectory_ / "crops").string()}}},
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
      {"events_jsonl", (sessionDirectory_ / "events.jsonl").string()},
      {"implementation", "C++17"},
  };
  writeJson(sessionDirectory_ / "summary.json", payload);
  logger_.info("END status=" + snapshot["status"].get<std::string>() +
               " frames=" + std::to_string(snapshot["processed_frames"].get<int>()) +
               " output=" + sessionDirectory_.string());
}

}  // namespace insight_cup
