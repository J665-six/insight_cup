#include "insight_cup/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
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
#if INSIGHT_CUP_HAS_ROS2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#endif

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

bool isTemporalSource(const std::string& source) {
  return isRealSenseSource(source) || isRosSource(source) ||
         isCameraIndexSource(source) || isVideoSource(source);
}

// Calibrated YOLO rectangle-area floors for the D455's original 1280x720
// stream, rescaled to the current 848x480 stream.  These remain separate
// because knife boxes are substantially smaller than person boxes.
constexpr double kStage2PersonMinAreaPx = 5035.0;
constexpr double kStage2KnifeMinAreaPx = 1428.0;

double stage2MinAreaPx(const std::string& majorClass) {
  if (majorClass == "f") {
    return kStage2PersonMinAreaPx;
  }
  if (majorClass == "k") {
    return kStage2KnifeMinAreaPx;
  }
  return 0.0;
}

int requiredStage1Confirmations(const RuntimeConfig& config) {
  return isTemporalSource(config.source) ? config.stage1Confirmations : 1;
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
  if (isRealSenseSource(source) || isRosSource(source) ||
      isCameraIndexSource(source) || !looksLikeLocalPath(source)) {
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

std::vector<cv::Point2f> calibrationPolygon(const json& value) {
  std::vector<cv::Point2f> polygon;
  if (!value.is_array()) {
    return polygon;
  }
  for (const auto& point : value) {
    if (point.is_array() && point.size() >= 2 && point[0].is_number() &&
        point[1].is_number()) {
      polygon.emplace_back(point[0].get<float>(), point[1].get<float>());
    }
  }
  return polygon.size() >= 3 ? polygon : std::vector<cv::Point2f>();
}

std::vector<cv::Point2f> calibrationRecordPolygon(const json& record) {
  auto polygon = calibrationPolygon(record.value("polygon", json()));
  if (!polygon.empty()) {
    return polygon;
  }
  json bbox = record.value("bbox", record.value("bbox_xyxy", json()));
  if (!bbox.is_array() && record.value("x1", json()).is_number() &&
      record.value("y1", json()).is_number() &&
      record.value("x2", json()).is_number() &&
      record.value("y2", json()).is_number()) {
    bbox = {record.at("x1"), record.at("y1"), record.at("x2"),
            record.at("y2")};
  }
  if (!bbox.is_array() || bbox.size() < 4 || !bbox[0].is_number() ||
      !bbox[1].is_number() || !bbox[2].is_number() ||
      !bbox[3].is_number()) {
    return {};
  }
  const float x1 = bbox[0].get<float>();
  const float y1 = bbox[1].get<float>();
  const float x2 = bbox[2].get<float>();
  const float y2 = bbox[3].get<float>();
  if (!(x2 > x1 && y2 > y1)) {
    return {};
  }
  return {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
}

double calibrationArea(const std::vector<cv::Point2f>& polygon) {
  return polygon.size() >= 3 ? std::abs(cv::contourArea(polygon)) : 0.0;
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
                 double frameGap, const cv::Size& frameSize) {
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
                                       EventJournal& journal, Logger& logger,
                                       std::function<void(
                                           const std::vector<std::string>&)>
                                           visionPublisher,
                                       std::function<void(const json&)>
                                           insightPublisher,
                                       std::function<void(
                                           const SafeObstacleState&)>
                                           safePublisher,
                                       std::function<bool(
                                           bool, const std::string&)>
                                           motionHoldRequester)
    : config_(std::move(config)),
      sessionDirectory_(std::move(sessionDirectory)),
      state_(state),
      journal_(journal),
      logger_(logger),
      visionPublisher_(std::move(visionPublisher)),
      insightPublisher_(std::move(insightPublisher)),
      safePublisher_(std::move(safePublisher)),
      motionHoldRequester_(std::move(motionHoldRequester)) {
  sourceFps_ = std::max(1, config_.cameraFps);
  // Keep the stage-1 motion model on the configured source-rate reference
  // from the first frame onward.
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

void RecognitionRuntime::loadStage1Roi() {
  stage1Roi_.clear();
  obstacleRoiForDisplay_.clear();
  stage1RoiConfigPath_.clear();
  stage2ReferenceAreaPx_ = 0.0;
  stage2PersonReferenceAreaPx_ = 0.0;
  stage2KnifeReferenceAreaPx_ = 0.0;
  stage1RoiConfigured_ = false;

  const fs::path requested = config_.stage1RoiConfig;
  std::vector<fs::path> candidates;
  std::error_code error;
  if (!requested.empty() && fs::is_regular_file(requested, error)) {
    candidates.push_back(requested);
  } else if (!requested.empty()) {
    const fs::path directory = requested.parent_path();
    error.clear();
    if (fs::is_directory(directory, error)) {
      std::vector<std::pair<fs::file_time_type, fs::path>> timestamped;
      std::error_code iteratorError;
      for (const auto& entry : fs::directory_iterator(directory, iteratorError)) {
        if (iteratorError) {
          break;
        }
        std::error_code entryError;
        if (!entry.is_regular_file(entryError) || entryError ||
            entry.path().extension() != ".json" ||
            entry.path().filename().string().find("distance_") != 0) {
          continue;
        }
        const auto timestamp = entry.last_write_time(entryError);
        if (!entryError) {
          timestamped.emplace_back(timestamp, entry.path());
        }
      }
      std::sort(timestamped.begin(), timestamped.end(),
                [](const auto& left, const auto& right) {
                  return left.first > right.first;
                });
      for (const auto& item : timestamped) {
        candidates.push_back(item.second);
      }
    }
  }

  for (const auto& candidatePath : candidates) {
    try {
      std::ifstream stream(candidatePath);
      if (!stream) {
        throw std::runtime_error("cannot open file");
      }
      json payload;
      stream >> payload;

      std::vector<cv::Point2f> roi;
      std::vector<double> referenceAreas;
      std::vector<double> personReferenceAreas;
      std::vector<double> knifeReferenceAreas;
      if (payload.value("schema", std::string()) ==
          "insight.stage1_roi.v1") {
        roi = calibrationPolygon(payload.value("roi_polygon", json()));
        if (roi.empty()) {
          roi = calibrationPolygon(payload.value("roi", json()));
        }
        obstacleRoiForDisplay_ =
            calibrationPolygon(payload.value("obstacle_roi", json()));
        const auto regions =
            payload.value("reference_regions", json::array());
        if (regions.is_array()) {
          for (const auto& region : regions) {
            const auto savedArea = region.value("area_px2", json());
            if (!savedArea.is_number() || savedArea.get<double>() <= 0.0) {
              continue;
            }
            const double area = savedArea.get<double>();
            referenceAreas.push_back(area);
            const std::string majorClass =
                region.value("major_class", region.value("kind", ""));
            if (majorClass == "f" || majorClass.rfind("f", 0) == 0) {
              personReferenceAreas.push_back(area);
            } else if (majorClass == "k" || majorClass.rfind("k", 0) == 0) {
              knifeReferenceAreas.push_back(area);
            }
          }
        }
        const auto values = payload.value("reference_areas_px", json::array());
        if (values.is_array() && referenceAreas.empty()) {
          for (const auto& value : values) {
            if (value.is_number() && value.get<double>() > 0.0) {
              referenceAreas.push_back(value.get<double>());
            }
          }
        }
      } else {
        const auto values = payload.value("boxes", json::array());
        if (!values.is_array()) {
          throw std::runtime_error("calibration has no boxes array");
        }
        struct Candidate {
          int id = 0;
          std::vector<cv::Point2f> polygon;
          double areaPx = 0.0;
          std::string majorClass;
        };
        std::vector<Candidate> boxes;
        for (const auto& value : values) {
          const auto polygon = calibrationRecordPolygon(value);
          if (polygon.size() < 3) {
            continue;
          }
          double areaPx = calibrationArea(polygon);
          const json savedArea = value.value("area_px2", json());
          if (savedArea.is_number() && savedArea.get<double>() > 0.0) {
            areaPx = savedArea.get<double>();
          }
          if (areaPx > 0.0) {
            std::string majorClass = value.value("kind", "");
            if (majorClass.empty()) {
              const std::string label = value.value("yolo_label", "");
              if (!label.empty() && (label.front() == 'f' || label.front() == 'k')) {
                majorClass = label.substr(0, 1);
              }
            }
            boxes.push_back({value.value("id", 0), polygon, areaPx,
                             std::move(majorClass)});
          }
        }
        if (boxes.empty()) {
          throw std::runtime_error("calibration contains no polygons");
        }

        auto roiIterator = std::find_if(
            boxes.begin(), boxes.end(),
            [](const Candidate& value) { return value.id == 7; });
        if (roiIterator == boxes.end()) {
          throw std::runtime_error(
              "calibration has no #7A person/knife ROI");
        }
        roi = roiIterator->polygon;
        std::set<int> selectedIds;
        for (int id = 1; id <= 4; ++id) {
          const auto found = std::find_if(
              boxes.begin(), boxes.end(),
              [id](const Candidate& value) { return value.id == id; });
          if (found != boxes.end() && found != roiIterator) {
            referenceAreas.push_back(found->areaPx);
            if (found->majorClass == "f") {
              personReferenceAreas.push_back(found->areaPx);
            } else if (found->majorClass == "k") {
              knifeReferenceAreas.push_back(found->areaPx);
            }
            selectedIds.insert(found->id);
          }
        }
        if (referenceAreas.size() < 4) {
          std::vector<const Candidate*> remaining;
          for (const auto& value : boxes) {
            if (&value != &*roiIterator && selectedIds.count(value.id) == 0) {
              remaining.push_back(&value);
            }
          }
          std::sort(remaining.begin(), remaining.end(),
                    [](const Candidate* left, const Candidate* right) {
                      return left->id < right->id;
                    });
          for (const auto* value : remaining) {
            if (referenceAreas.size() >= 4) {
              break;
            }
            referenceAreas.push_back(value->areaPx);
            if (value->majorClass == "f") {
              personReferenceAreas.push_back(value->areaPx);
            } else if (value->majorClass == "k") {
              knifeReferenceAreas.push_back(value->areaPx);
            }
          }
        }
      }

      if (roi.size() < 3 || referenceAreas.size() < 4) {
        throw std::runtime_error(
            "requires #7A ROI plus four #1..#4 reference regions");
      }
      const double sum =
          std::accumulate(referenceAreas.begin(), referenceAreas.end(), 0.0);
      if (!(sum > 0.0)) {
        throw std::runtime_error("reference pixel areas are invalid");
      }
      stage1Roi_ = std::move(roi);
      stage2ReferenceAreaPx_ = sum / referenceAreas.size();
      if (!personReferenceAreas.empty()) {
        stage2PersonReferenceAreaPx_ =
            std::accumulate(personReferenceAreas.begin(),
                            personReferenceAreas.end(), 0.0) /
            personReferenceAreas.size();
      }
      if (!knifeReferenceAreas.empty()) {
        stage2KnifeReferenceAreaPx_ =
            std::accumulate(knifeReferenceAreas.begin(),
                            knifeReferenceAreas.end(), 0.0) /
            knifeReferenceAreas.size();
      }
      stage1RoiConfigPath_ = candidatePath;
      stage1RoiConfigured_ = true;
      json polygon = json::array();
      for (const auto& point : stage1Roi_) {
        polygon.push_back({point.x, point.y});
      }
      modelMetadata_["stage1_roi"] = {
          {"enabled", config_.stage1RoiEnabled},
          {"configured", true},
          {"display_only", !config_.stage1RoiEnabled},
          {"config_path", displayPath(candidatePath)},
          {"polygon", polygon},
          {"obstacle_roi", [&] {
             json value = json::array();
             for (const auto& point : obstacleRoiForDisplay_) {
               value.push_back({point.x, point.y});
             }
             return value;
           }()},
          {"reference_area_px", stage2ReferenceAreaPx_},
          {"person_reference_area_px", stage2PersonReferenceAreaPx_ > 0.0
                                           ? json(stage2PersonReferenceAreaPx_)
                                           : json(nullptr)},
          {"knife_reference_area_px", stage2KnifeReferenceAreaPx_ > 0.0
                                          ? json(stage2KnifeReferenceAreaPx_)
                                          : json(nullptr)},
          {"reference_regions", referenceAreas.size()},
          {"area_tolerance", config_.stage2AreaTolerance}};
      logger_.info("READY module=stage1_roi path=" +
                   displayPath(candidatePath) + " roi_points=" +
                   std::to_string(stage1Roi_.size()) + " reference_area_px=" +
                   fixed(stage2ReferenceAreaPx_, 2) + " tolerance=" +
                   fixed(config_.stage2AreaTolerance, 2) +
                   " person_min_area_px=" +
                   fixed(stage2PersonReferenceAreaPx_ > 0.0
                             ? stage2PersonReferenceAreaPx_ *
                                   (1.0 - config_.stage2AreaTolerance)
                             : kStage2PersonMinAreaPx,
                         0) +
                   " knife_min_area_px=" +
                   fixed(stage2KnifeReferenceAreaPx_ > 0.0
                             ? stage2KnifeReferenceAreaPx_ *
                                   (1.0 - config_.stage2AreaTolerance)
                             : kStage2KnifeMinAreaPx,
                         0) +
                   (config_.stage1RoiEnabled ? "" :
                                                " (display-only; gating disabled)"));
      return;
    } catch (const std::exception& exception) {
      logger_.warning("STAGE1 ROI config ignored path=" +
                      displayPath(candidatePath) + " error=" +
                      exception.what());
    }
  }

  modelMetadata_["stage1_roi"] = {
      {"enabled", config_.stage1RoiEnabled},
      {"configured", false},
      {"display_only", !config_.stage1RoiEnabled},
      {"config_path", displayPath(requested)},
      {"reference_area_px", nullptr},
      {"area_tolerance", config_.stage2AreaTolerance}};
  if (config_.stage1RoiEnabled) {
    logger_.warning("STAGE1 ROI unavailable; person/knife stage 2 is gated until "
                    "#7A plus reference calibration is available");
  } else {
    logger_.info("STAGE1 ROI disabled; using the full camera frame");
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
  loadStage1Roi();
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

  const fs::path knifeSource =
      config_.knifeMode == "classification"
          ? config_.knifeModelDir
          : config_.knifeMode == "retrieval"
                ? config_.knifeRetrievalModel
                : config_.knifeTraditionalTemplateDir;
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
  const bool asynchronous = isRealSenseSource(config_.source) ||
                            isRosSource(config_.source) || cameraIndex ||
                            isVideoSource(config_.source);
  scheduler_ = std::make_unique<AsyncStage2Scheduler>(
      config_.stage2RefreshFrames, config_.stage2StableRefreshFrames,
      config_.stage2CacheIou, asynchronous, config_.stage2Workers,
      config_.stage2Smoothing, config_.stage2SmoothingWindow,
      config_.stage2SwitchConfirmations, config_.faceThreshold,
      config_.faceMinMargin, knife_->threshold(), knife_->minMargin());
  safeMonitor_ = std::make_unique<SafeObstacleMonitor>(config_, logger_);
  modelMetadata_["safe"] = safeMonitor_->metadata();
  logger_.info("READY module=safe enabled=" +
               std::string(config_.safeEnabled ? "true" : "false") +
               " stable_seconds=" + fixed(config_.safeStableSeconds, 1));
  modelMetadata_["temporal"] = {
      {"stage1_confirmations", config_.stage1Confirmations},
      {"stage1_confirmation_scope", "camera_and_video"},
      {"method", config_.stage2Smoothing},
      {"window_size", config_.stage2SmoothingWindow},
      {"switch_confirmations", config_.stage2SwitchConfirmations},
      {"refresh_frames", config_.stage2RefreshFrames},
      {"stable_refresh_frames", config_.stage2StableRefreshFrames},
      {"scope", "per_track_id"},
  };
  logger_.info("READY module=temporal stage1_confirmations=" +
               std::to_string(config_.stage1Confirmations) +
               " method=" + config_.stage2Smoothing +
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
  if (isRosSource(config_.source)) {
    runRosSubscriber();
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
  sourceFps_ = static_cast<double>(config_.cameraFps);
  scheduler_->setFrameRate(sourceFps_);
  logger_.info("TEMPORAL source_fps=" + fixed(sourceFps_, 2) +
               " refresh_frames=" +
               std::to_string(scheduler_->refreshFrames()) +
               " stable_refresh_frames=" +
               std::to_string(scheduler_->stableRefreshFrames()));

  cv::VideoWriter writer;
  int frameIndex = 0;
  int processedIndex = 0;
  int frameTimeouts = 0;
  bool pipelineRunning = true;
  try {
    while (state_.waitUntilRunnable()) {
      rs2::frameset frames;
      try {
        frames = pipeline.wait_for_frames(1000);
        frameTimeouts = 0;
      } catch (const rs2::error& error) {
        ++frameTimeouts;
        logger_.warning("RealSense frame timeout " +
                        std::to_string(frameTimeouts) + "/3: " +
                        std::string(error.what()));
        if (frameTimeouts >= 3) {
          logger_.warning("RealSense stream stalled; restarting color stream");
          if (pipelineRunning) {
            pipeline.stop();
            pipelineRunning = false;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(250));
          try {
            profile = pipeline.start(cameraConfig);
            pipelineRunning = true;
            frameTimeouts = 0;
            logger_.info("RealSense color stream restarted");
          } catch (const rs2::error& restartError) {
            logger_.warning("RealSense stream restart failed: " +
                            std::string(restartError.what()));
          }
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        continue;
      }
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
    if (pipelineRunning) {
      pipeline.stop();
    }
    writer.release();
    throw;
  }
  if (pipelineRunning) {
    pipeline.stop();
  }
  writer.release();
  if (processedIndex == 0 && !state_.stopRequested()) {
    throw std::runtime_error("RealSense camera produced no color frames");
  }
#else
  throw std::runtime_error(
      "This C++ build does not include Intel RealSense support");
#endif
}

void RecognitionRuntime::runRosSubscriber() {
#if INSIGHT_CUP_HAS_ROS2
  const std::string topic = rosTopicName(config_.source);
  const auto context = rclcpp::contexts::get_global_default_context();
  if (!context->is_valid()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("insight_cup_ros_source");

  struct FrameSlot {
    std::mutex mutex;
    std::condition_variable signal;
    cv::Mat frame;
    std::optional<double> stampMs;
    bool fresh = false;
  };
  FrameSlot slot;

  std::string encoding;
  int64_t frameCounter = 0;
  auto subscription = node->create_subscription<sensor_msgs::msg::Image>(
      topic, rclcpp::SensorDataQoS(),
      [&](sensor_msgs::msg::Image::SharedPtr msg) {
        if (msg->width == 0 || msg->height == 0 || msg->data.empty()) {
          return;
        }
        const std::string enc = msg->encoding;
        if (enc != "rgb8" && enc != "bgr8") {
          logger_.warning("Unsupported ROS image encoding on " + topic +
                          ": " + enc);
          return;
        }
        cv::Mat wrapped(static_cast<int>(msg->height),
                        static_cast<int>(msg->width), CV_8UC3,
                        const_cast<uint8_t*>(msg->data.data()),
                        static_cast<std::size_t>(msg->step));
        cv::Mat converted;
        if (enc == "rgb8") {
          cv::cvtColor(wrapped, converted, cv::COLOR_RGB2BGR);
        } else {
          converted = wrapped.clone();
        }
        // ROI polygons, pixel-area gates, and obstacle calibration are all
        // expressed in the configured camera geometry (848x480 by default).
        // A ROS bag can carry the same 16:9 stream at 1280x720; normalize it
        // at the source boundary so the UI overlay and the formal pipeline
        // consume exactly the same calibrated pixel coordinates.
        if (converted.cols != config_.cameraWidth ||
            converted.rows != config_.cameraHeight) {
          cv::Mat normalized;
          cv::resize(converted, normalized,
                     cv::Size(config_.cameraWidth, config_.cameraHeight), 0.0,
                     0.0, cv::INTER_AREA);
          if (!rosSourceResizeLogged_) {
            logger_.warning(
                "SOURCE ROS frame normalized from " +
                std::to_string(converted.cols) + "x" +
                std::to_string(converted.rows) + " to " +
                std::to_string(config_.cameraWidth) + "x" +
                std::to_string(config_.cameraHeight) +
                " to match pixel calibration and ROI");
            rosSourceResizeLogged_ = true;
          }
          converted = std::move(normalized);
        }
        std::optional<double> stampMs;
        const double seconds =
            msg->header.stamp.sec + 1e-9 * msg->header.stamp.nanosec;
        if (seconds > 0.0) {
          stampMs = 1000.0 * seconds;
        }
        {
          std::lock_guard<std::mutex> lock(slot.mutex);
          encoding = enc;
          slot.frame = std::move(converted);
          slot.stampMs = stampMs;
          slot.fresh = true;
        }
        slot.signal.notify_one();
      });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  logger_.info("SOURCE type=ros topic=/" + topic + " fps=" +
               std::to_string(config_.cameraFps));
  sourceFps_ = static_cast<double>(config_.cameraFps);
  scheduler_->setFrameRate(sourceFps_);

  cv::VideoWriter writer;
  int64_t processedIndex = 0;
  bool writerOpened = false;
  std::unique_lock<std::mutex> lock(slot.mutex);
  try {
    while (state_.waitUntilRunnable()) {
      if (!slot.signal.wait_for(lock, std::chrono::seconds(3),
                                [&]() { return slot.fresh; })) {
        logger_.warning("No frames received on /" + topic +
                        " within 3 s; is the camera publisher running?");
        continue;
      }
      const cv::Mat frame = slot.frame;
      const std::optional<double> stampMs = slot.stampMs;
      slot.fresh = false;
      lock.unlock();

      state_.setSourceInfo({
          {"type", "ros"},
          {"topic", topic},
          {"encoding", encoding},
          {"width", frame.cols},
          {"height", frame.rows},
          {"fps", config_.cameraFps},
          {"total_frames", 0},
          {"video_stride", config_.videoStride},
      });
      if (!writerOpened && config_.saveVideo) {
        const fs::path output = sessionDirectory_ / "annotated.mp4";
        writer.open(output.string(),
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    std::max(1.0, static_cast<double>(config_.cameraFps) /
                                      config_.videoStride),
                    frame.size());
        if (!writer.isOpened()) {
          throw std::runtime_error("Could not write video: " +
                                   displayPath(output));
        }
        writerOpened = true;
        mediaOutputs_.push_back(displayPath(output));
      }
      const int64_t currentIndex = frameCounter++;
      if (currentIndex % config_.videoStride != 0) {
        lock.lock();
        continue;
      }
      const cv::Mat annotated = processFrame(
          frame, static_cast<int>(currentIndex),
          static_cast<int>(processedIndex), "ros:" + topic, stampMs);
      if (writer.isOpened()) {
        writer.write(annotated);
      }
      ++processedIndex;
      if (config_.maxFrames &&
          processedIndex >= static_cast<std::size_t>(*config_.maxFrames)) {
        break;
      }
      lock.lock();
    }
  } catch (...) {
    if (lock.owns_lock()) {
      lock.unlock();
    }
    executor.cancel();
    if (spinner.joinable()) {
      spinner.join();
    }
    writer.release();
    throw;
  }
  if (lock.owns_lock()) {
    lock.unlock();
  }
  executor.cancel();
  if (spinner.joinable()) {
    spinner.join();
  }
  writer.release();
  if (processedIndex == 0 && !state_.stopRequested()) {
    throw std::runtime_error("No frames received on ROS image topic: /" +
                             topic);
  }
#else
  throw std::runtime_error("This C++ build does not include ROS 2 support");
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
    sourceFps = cameraIndex ? static_cast<double>(config_.cameraFps) : 30.0;
  }
  sourceFps_ = sourceFps;
  scheduler_->setFrameRate(sourceFps_);
  logger_.info("TEMPORAL source_fps=" + fixed(sourceFps_, 2) +
               " refresh_frames=" +
               std::to_string(scheduler_->refreshFrames()) +
               " stable_refresh_frames=" +
               std::to_string(scheduler_->stableRefreshFrames()));
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

double RecognitionRuntime::normalizedFrameGap(int frameGap) const {
  return std::max(0.0, static_cast<double>(std::max(0, frameGap)) * 30.0 /
                            std::max(sourceFps_, 1.0));
}

std::optional<float> RecognitionRuntime::stage1AssociationScore(
    const Stage1Track& track, const DetectedRegion& region,
    const cv::Size& frameSize) const {
  if (track.region.handoff.majorClass != region.handoff.majorClass) {
    return std::nullopt;
  }
  const int rawFrameGap =
      std::max(1, region.handoff.frameIndex - track.lastYoloFrame);
  const double frameGap = normalizedFrameGap(rawFrameGap);
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
    const Box previousBox = track.region.handoff.bbox;
    const Box currentBox = region.handoff.bbox;
    const double previousArea = static_cast<double>(previousBox.width()) *
                                static_cast<double>(previousBox.height());
    const double currentArea = static_cast<double>(currentBox.width()) *
                              static_cast<double>(currentBox.height());
    const double areaRatio = previousArea > 0.0
                                 ? currentArea / previousArea
                                 : 0.0;
    const bool stableBox = areaRatio >= 0.65 && areaRatio <= 1.35 &&
                           bboxIou(previousBox, currentBox) >= 0.25F;
    const double frameGap = normalizedFrameGap(
        std::max(1, frameIndex - track.lastYoloFrame));
    const auto current = boxValues(region.handoff.bbox);
    const auto previous = boxValues(track.region.handoff.bbox);
    for (std::size_t index = 0; index < current.size(); ++index) {
      const float observed = static_cast<float>(
          (current[index] - previous[index]) / frameGap);
      track.velocity[index] =
          track.hits <= 1 ? observed : 0.70F * observed + 0.30F * track.velocity[index];
    }
    track.region = region;
    track.lastYoloFrame = frameIndex;
    track.missedYoloFrames = 0;
    ++track.hits;
    track.stableBoxHits = stableBox ? track.stableBoxHits + 1 : 1;
    const int requiredHits = std::max(requiredStage1Confirmations(config_),
                                      region.handoff.majorClass == "k" ? 3 : 1);
    if (track.hits >= requiredHits &&
        (region.handoff.majorClass != "k" || track.stableBoxHits >= 3)) {
      filtered.regions.push_back(std::move(region));
    }
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
    track.stableBoxHits = 1;
    stage1Tracks_.push_back(std::move(track));
    if (requiredStage1Confirmations(config_) <= 1 &&
        region.handoff.majorClass != "k") {
      filtered.regions.push_back(std::move(region));
    }
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
    if (track.hits < requiredStage1Confirmations(config_)) {
      continue;
    }
    if (track.region.handoff.majorClass == "k" && track.stableBoxHits < 3) {
      continue;
    }
    const int frameGap = frameIndex - track.lastYoloFrame;
    if (frameGap <= 0 || frameGap > std::max(1, config_.yoloFrameStride)) {
      continue;
    }
    DetectedRegion region = track.region;
    region.handoff.frameIndex = frameIndex;
    region.handoff.bbox =
        predictedBox(track.region.handoff.bbox, track.velocity,
                     normalizedFrameGap(frameGap), frameSize);
    if (!region.handoff.bbox.valid()) {
      continue;
    }
    const cv::Point2f center(
        0.5F * static_cast<float>(region.handoff.bbox.x1 +
                                  region.handoff.bbox.x2),
        0.5F * static_cast<float>(region.handoff.bbox.y1 +
                                  region.handoff.bbox.y2));
    if (!stage1PointAllowed(region.handoff.majorClass, center)) {
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

DetectionFrame RecognitionRuntime::detectStage1Region(const cv::Mat& frame,
                                                      int frameIndex) {
  if (!config_.stage1RoiEnabled) {
    return detector_->detect(frame, frameIndex);
  }
  const auto* obstacleRoi = configuredObstacleRoi();
  const bool hasPersonKnifeRoi = stage1RoiConfigured_ && stage1Roi_.size() >= 3;
  const bool hasObstacleRoi = obstacleRoi && obstacleRoi->size() >= 3;
  if (!hasPersonKnifeRoi && !hasObstacleRoi) {
    return {};
  }

  cv::Rect bounds;
  bool hasBounds = false;
  const auto includeBounds = [&](const std::vector<cv::Point2f>& polygon) {
    const cv::Rect polygonBounds = cv::boundingRect(polygon);
    if (!hasBounds) {
      bounds = polygonBounds;
      hasBounds = true;
    } else {
      bounds |= polygonBounds;
    }
  };
  if (hasPersonKnifeRoi) {
    includeBounds(stage1Roi_);
  }
  if (hasObstacleRoi) {
    includeBounds(*obstacleRoi);
  }
  if (!hasBounds) {
    return {};
  }
  bounds &= cv::Rect(0, 0, frame.cols, frame.rows);
  if (bounds.width <= 0 || bounds.height <= 0) {
    return {};
  }
  cv::Mat region = frame(bounds).clone();
  cv::Mat mask = cv::Mat::zeros(region.size(), CV_8UC1);
  const auto fillLocalPolygon = [&](const std::vector<cv::Point2f>& polygon) {
    std::vector<cv::Point> localPolygon;
    localPolygon.reserve(polygon.size());
    for (const auto& point : polygon) {
      localPolygon.emplace_back(
          std::clamp(static_cast<int>(std::nearbyint(point.x)) - bounds.x, 0,
                     bounds.width - 1),
          std::clamp(static_cast<int>(std::nearbyint(point.y)) - bounds.y, 0,
                     bounds.height - 1));
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{localPolygon},
                 cv::Scalar(255));
  };
  if (hasPersonKnifeRoi) {
    fillLocalPolygon(stage1Roi_);
  }
  if (hasObstacleRoi) {
    fillLocalPolygon(*obstacleRoi);
  }
  region.setTo(cv::Scalar(114, 114, 114), mask == 0);

  DetectionFrame detected = detector_->detect(region, frameIndex);
  DetectionFrame filtered;
  filtered.inferenceMs = detected.inferenceMs;
  filtered.ignoredClasses = std::move(detected.ignoredClasses);
  for (auto& item : detected.regions) {
    auto& box = item.handoff.bbox;
    box.x1 += bounds.x;
    box.x2 += bounds.x;
    box.y1 += bounds.y;
    box.y2 += bounds.y;
    box.x1 = std::clamp(box.x1, 0, frame.cols);
    box.x2 = std::clamp(box.x2, 0, frame.cols);
    box.y1 = std::clamp(box.y1, 0, frame.rows);
    box.y2 = std::clamp(box.y2, 0, frame.rows);
    if (!box.valid()) {
      continue;
    }
    const cv::Point2f center(
        0.5F * static_cast<float>(box.x1 + box.x2),
        0.5F * static_cast<float>(box.y1 + box.y2));
    if (stage1PointAllowed(item.handoff.majorClass, center)) {
      filtered.regions.push_back(std::move(item));
    }
  }
  return filtered;
}

const std::vector<cv::Point2f>* RecognitionRuntime::configuredObstacleRoi() const {
  if (safeMonitor_ && safeMonitor_->obstacleRoi().size() >= 3) {
    return &safeMonitor_->obstacleRoi();
  }
  if (obstacleRoiForDisplay_.size() >= 3) {
    return &obstacleRoiForDisplay_;
  }
  return nullptr;
}

bool RecognitionRuntime::stage1PointAllowed(
    const std::string& majorClass, const cv::Point2f& center) const {
  if (!config_.stage1RoiEnabled) {
    return true;
  }
  if (majorClass == "b0") {
    const auto* obstacleRoi = configuredObstacleRoi();
    return obstacleRoi &&
           cv::pointPolygonTest(*obstacleRoi, center, false) >= 0.0;
  }
  if (majorClass == "f" || majorClass == "k") {
    return stage1RoiConfigured_ && stage1Roi_.size() >= 3 &&
           cv::pointPolygonTest(stage1Roi_, center, false) >= 0.0;
  }
  return false;
}

bool RecognitionRuntime::stage2AreaEligible(const DetectedRegion& region) const {
  if (region.handoff.majorClass == "b0" || !config_.stage1RoiEnabled) {
    return true;
  }
  if (!stage1RoiConfigured_ || stage2ReferenceAreaPx_ <= 0.0) {
    return false;
  }
  if (region.handoff.majorClass != "f" && region.handoff.majorClass != "k") {
    return false;
  }
  const double area = static_cast<double>(region.handoff.bbox.width()) *
                      static_cast<double>(region.handoff.bbox.height());
  const double reference = region.handoff.majorClass == "f"
                               ? stage2PersonReferenceAreaPx_
                               : stage2KnifeReferenceAreaPx_;
  const double lower = reference > 0.0
                           ? reference * (1.0 - config_.stage2AreaTolerance)
                           : stage2MinAreaPx(region.handoff.majorClass);
  return lower > 0.0 && area >= lower;
}

DetectionFrame RecognitionRuntime::stage1Frame(const cv::Mat& frame,
                                               int frameIndex,
                                               int processedIndex) {
  if (config_.yoloFrameStride <= 1 && !isTemporalSource(config_.source)) {
    return detectStage1Region(frame, frameIndex);
  }
  if (config_.yoloFrameStride <= 1) {
    DetectionFrame detectionFrame = detectStage1Region(frame, frameIndex);
    return trackYoloDetections(std::move(detectionFrame), frame.size(),
                               frameIndex);
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
    DetectionFrame detectionFrame = detectStage1Region(frame, frameIndex);
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
  std::optional<SafeObstacleState> safeState;
  if (safeMonitor_) {
    safeState = safeMonitor_->update(
        detectionFrame, frame.size(), processedIndex, sourceTimeMs);
    if (safePublisher_) {
      safePublisher_(*safeState);
    }
  }
  cv::Mat annotated;
  if (needsAnnotatedFrame) {
    annotated = frame.clone();
    const auto drawRoi = [&](const std::vector<cv::Point2f>& polygon,
                             const cv::Scalar& color,
                             const std::string& label) {
      if (polygon.size() < 3) {
        return;
      }
      std::vector<cv::Point> points;
      points.reserve(polygon.size());
      for (const auto& point : polygon) {
        points.emplace_back(
            std::clamp(static_cast<int>(std::nearbyint(point.x)), 0,
                       std::max(0, annotated.cols - 1)),
            std::clamp(static_cast<int>(std::nearbyint(point.y)), 0,
                       std::max(0, annotated.rows - 1)));
      }
      cv::polylines(annotated, std::vector<std::vector<cv::Point>>{points},
                    true, color, 2, cv::LINE_AA);
      const cv::Point anchor = points.front();
      cv::putText(annotated, label,
                  cv::Point(std::max(8, anchor.x),
                            std::max(24, anchor.y - 8)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
    };
    if (stage1RoiConfigured_) {
      const double personFloor = stage2PersonReferenceAreaPx_ > 0.0
                                     ? stage2PersonReferenceAreaPx_ *
                                           (1.0 - config_.stage2AreaTolerance)
                                     : kStage2PersonMinAreaPx;
      const double knifeFloor = stage2KnifeReferenceAreaPx_ > 0.0
                                    ? stage2KnifeReferenceAreaPx_ *
                                          (1.0 - config_.stage2AreaTolerance)
                                    : kStage2KnifeMinAreaPx;
      drawRoi(stage1Roi_, cv::Scalar(0, 220, 80),
              "#7A PERSON/KNIFE ROI | F>=" + fixed(personFloor, 0) +
                  " K>=" + fixed(knifeFloor, 0) +
                  (config_.stage1RoiEnabled ? " PX2" : " PX2 DISPLAY ONLY"));
    }
    const std::vector<cv::Point2f>* obstacleRoi = &obstacleRoiForDisplay_;
    if (safeMonitor_ && !safeMonitor_->obstacleRoi().empty()) {
      obstacleRoi = &safeMonitor_->obstacleRoi();
    }
    drawRoi(*obstacleRoi, cv::Scalar(0, 80, 255), "#1A OBSTACLE ROI");
    if (safeState) {
      cv::putText(annotated, "SAFE: " + safeState->status,
                  cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                  safeState->obstacle ? cv::Scalar(0, 80, 255)
                                      : cv::Scalar(230, 230, 230),
                  2, cv::LINE_AA);
    }
  }
  double stage2TotalMs = 0.0;
  int recognitionCount = 0;
  for (const auto& rawName : detectionFrame.ignoredClasses) {
    logger_.warning("DROP frame=" + std::to_string(frameIndex) +
                    " raw_class=" + rawName + " reason=unsupported_class");
  }
  DetectionFrame stage2Frame;
  stage2Frame.inferenceMs = detectionFrame.inferenceMs;
  stage2Frame.ignoredClasses = detectionFrame.ignoredClasses;
  stage2Frame.regions.reserve(detectionFrame.regions.size());
  // A person and a knife do not reach the camera at exactly the same pixel
  // scale.  Use the calibrated lower bound as the primary trigger, but keep
  // the companion class in the stage-2 frame when its live box is plausibly
  // sized.  This prevents stopping on a valid person while dropping the
  // slightly smaller knife before recognition and speech can run.
  bool strictPersonReady = false;
  bool strictKnifeReady = false;
  for (const auto& region : detectionFrame.regions) {
    if (region.handoff.majorClass != "f" && region.handoff.majorClass != "k") {
      continue;
    }
    if (stage2AreaEligible(region)) {
      if (region.handoff.majorClass == "f") strictPersonReady = true;
      if (region.handoff.majorClass == "k") strictKnifeReady = true;
    }
  }
  const auto companionAreaEligible = [&](const DetectedRegion& region) {
    if ((region.handoff.majorClass != "f" &&
         region.handoff.majorClass != "k") || !stage1RoiConfigured_) {
      return false;
    }
    const double reference = region.handoff.majorClass == "f"
                                 ? stage2PersonReferenceAreaPx_
                                 : stage2KnifeReferenceAreaPx_;
    if (reference <= 0.0) return false;
    const double area = static_cast<double>(region.handoff.bbox.width()) *
                        static_cast<double>(region.handoff.bbox.height());
    // Broad companion window: the strict class still controls entry, while
    // the second class may be smaller/larger because of pose and occlusion.
    return area >= reference * 0.35 && area <= reference * 2.25;
  };
  for (const auto& region : detectionFrame.regions) {
    state_.recordDetection(region.handoff.majorClass);
    const bool strictReady = stage2AreaEligible(region);
    const bool companionReady = (region.handoff.majorClass == "f" &&
                                 strictKnifeReady) ||
                                (region.handoff.majorClass == "k" &&
                                 strictPersonReady);
    if (strictReady || (companionReady && companionAreaEligible(region))) {
      stage2Frame.regions.push_back(region);
    } else {
      if (!annotated.empty()) {
        const Box& box = region.handoff.bbox;
        const cv::Scalar gateColor(0, 215, 255);
        cv::rectangle(annotated, cv::Point(box.x1, box.y1),
                      cv::Point(box.x2 - 1, box.y2 - 1), gateColor, 2);
        const double area = static_cast<double>(box.width()) * box.height();
        std::ostringstream gateLabel;
        gateLabel << region.handoff.majorClass << ">stage2_wait area="
                  << std::fixed << std::setprecision(0) << area;
        cv::putText(annotated, gateLabel.str(),
                    cv::Point(std::max(0, box.x1),
                              std::max(16, box.y1 - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, gateColor, 1,
                    cv::LINE_AA);
      }
      logger_.debug("STAGE2 wait frame=" + std::to_string(frameIndex) +
                    " class=" + region.handoff.majorClass +
                    " bbox_area_px=" +
                    std::to_string(region.handoff.bbox.width() *
                                   region.handoff.bbox.height()) +
                    " reference_area_px=" + fixed(stage2ReferenceAreaPx_, 2));
    }
  }

  const bool hasInspectionCandidate = std::any_of(
      stage2Frame.regions.begin(), stage2Frame.regions.end(),
      [](const DetectedRegion& region) {
        return region.handoff.majorClass == "f" ||
               region.handoff.majorClass == "k";
      });
  if (hasInspectionCandidate) {
    inspectionClearFrames_ = 0;
    if (!inspectionEpisodeActive_) {
      inspectionEpisodeActive_ = true;
      inspectionReported_ = false;
      inspectionReleasePending_ = false;
      inspectionStableFrames_ = 0;
      inspectionHoldFrames_ = 0;
      inspectionStableLabels_.clear();
      logger_.info("INSPECTION episode_started frame=" +
                   std::to_string(frameIndex));
    }
    if (!inspectionHoldActive_ && !inspectionReported_ &&
        !inspectionReleasePending_) {
      bool held = true;
      if (motionHoldRequester_) {
        held = motionHoldRequester_(true, "recognizing clue");
      } else {
        if (!motionHoldUnavailableWarned_) {
          logger_.warning(
              "MOTION_HOLD service is unavailable; continuing in debug mode "
              "without actually stopping the vehicle");
          motionHoldUnavailableWarned_ = true;
        }
      }
      if (held) {
        inspectionHoldActive_ = true;
        inspectionHoldFrames_ = 0;
        logger_.info("INSPECTION hold=true frame=" +
                     std::to_string(frameIndex));
      } else {
        logger_.error("INSPECTION hold=true failed; stage2 is deferred");
        stage2Frame.regions.clear();
      }
    }
  } else if (!inspectionHoldActive_) {
    ++inspectionClearFrames_;
    if (inspectionClearFrames_ >= 5) {
      inspectionEpisodeActive_ = false;
      inspectionReported_ = false;
      inspectionReleasePending_ = false;
      inspectionStableFrames_ = 0;
      inspectionHoldFrames_ = 0;
      inspectionStableLabels_.clear();
    }
  }

  if (inspectionHoldActive_ && !inspectionReported_) {
    ++inspectionHoldFrames_;
    const int timeoutFrames = std::max(
        1, static_cast<int>(std::nearbyint(config_.cameraFps * 5.0F)));
    if (inspectionHoldFrames_ > timeoutFrames) {
      logger_.warning("INSPECTION stage2 timeout; releasing motion hold");
      inspectionReported_ = true;
      inspectionStableFrames_ = 0;
      if (motionHoldRequester_) {
        if (motionHoldRequester_(false, "stage2 timeout")) {
          inspectionHoldActive_ = false;
        } else {
          inspectionReleasePending_ = true;
        }
      } else {
        inspectionHoldActive_ = false;
      }
    }
  }

  std::vector<PrepareJob> prepareJobs;
  prepareJobs.reserve(stage2Frame.regions.size());
  for (const auto& region : stage2Frame.regions) {
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
      stage2Frame.regions,
      [this](const Stage1Detection& detection) {
        return router_->route(detection);
      },
      prepareJobs);
  std::set<std::string> stableVisionLabels;
  for (std::size_t index = 0; index < stage2Frame.regions.size(); ++index) {
    const auto& region = stage2Frame.regions[index];
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
      if (resolution.decision.status == "matched" &&
          resolution.decision.predictedClass) {
        const auto& label = *resolution.decision.predictedClass;
        if (label.size() >= 2 &&
            (label.front() == 'f' || label.front() == 'k')) {
          try {
            const int number = std::stoi(label.substr(1));
            if (number >= 1 && number <= 10 &&
                label == std::string(1, label.front()) +
                             std::to_string(number)) {
              stableVisionLabels.insert(label);
            }
          } catch (const std::exception&) {
            // Ignore labels outside the f1..f10 and k1..k10 topic contract.
          }
        }
      }
      if (!annotated.empty()) {
        drawResult(annotated, region, resolution.decision);
      }
      if (!competition) {
        ++recognitionCount;
      }
    }
  }
  std::vector<std::string> labels(stableVisionLabels.begin(),
                                  stableVisionLabels.end());
  std::stable_sort(labels.begin(), labels.end(), [](const auto& left,
                                                    const auto& right) {
    if (left.front() != right.front()) {
      return left.front() == 'f';
    }
    return std::stoi(left.substr(1)) < std::stoi(right.substr(1));
  });

  // A stable target event is the authoritative result for this frame.  Do
  // not immediately overwrite it with the ordinary empty-target aggregate
  // message below; that made /insight appear to lose the result to subscribers.
  bool publishedTargetEventThisFrame = false;
  // A confirmed person is already a reportable inspection target.  The old
  // implementation required exactly one person and one knife before it
  // emitted /insight, which meant that a correctly recognized person alone
  // was visible on /vision but could never reach the navigation speech path.
  // Keep the temporal confirmation below as the false-positive guard, while
  // allowing one or more stable person/knife labels in the event.
  const bool hasStableInspectionTarget = !labels.empty();

  if (inspectionHoldActive_ && !inspectionReported_) {
    if (!hasStableInspectionTarget) {
      inspectionStableFrames_ = 0;
      inspectionStableLabels_.clear();
    } else {
      const std::set<std::string> currentLabels(labels.begin(), labels.end());
      if (currentLabels == inspectionStableLabels_) {
        ++inspectionStableFrames_;
      } else {
        inspectionStableLabels_ = currentLabels;
        inspectionStableFrames_ = 1;
      }
      const int requiredStableFrames = config_.inspectionStableFrames;
      logger_.info("INSIGHT stability labels=" + json(labels).dump() +
                   " frames=" + std::to_string(inspectionStableFrames_) +
                   "/" + std::to_string(requiredStableFrames));
      if (inspectionStableFrames_ >= requiredStableFrames) {
        json targetLabels = json::array();
        for (const auto& label : labels) {
          targetLabels.push_back(label);
        }
        ++recognitionEventId_;
        json finalEvent = {
            {"timestamp_ms",
             static_cast<std::int64_t>(std::chrono::duration_cast<
                 std::chrono::milliseconds>(std::chrono::system_clock::now()
                                                .time_since_epoch())
                                              .count())},
            {"obstacle_detected",
             safeState ? safeState->obstacle : false},
            {"obstacle_distance_cm",
             safeState && safeState->obstacle ? safeState->distanceCm : 0},
            {"target_detected", true},
            {"targets", targetLabels}};
        if (insightPublisher_) {
          insightPublisher_(finalEvent);
          publishedTargetEventThisFrame = true;
        }
        logger_.info("INSIGHT target_event=" +
                     std::to_string(recognitionEventId_) +
                     " targets=" + finalEvent["targets"].dump());
        inspectionReported_ = true;
        inspectionStableFrames_ = 0;
        bool released = true;
        if (motionHoldRequester_) {
          released = motionHoldRequester_(false, "");
        }
        if (released) {
          inspectionHoldActive_ = false;
          inspectionReleasePending_ = false;
          logger_.info("INSPECTION hold=false frame=" +
                       std::to_string(frameIndex));
        } else {
          inspectionReleasePending_ = true;
          logger_.error("INSPECTION hold=false failed; retrying");
        }
      }
    }
  }

  if (inspectionReleasePending_ && motionHoldRequester_) {
    if (motionHoldRequester_(false, "")) {
      inspectionHoldActive_ = false;
      inspectionReleasePending_ = false;
      logger_.info("INSPECTION hold=false retry succeeded");
    }
  }

  if (visionPublisher_) {
    // Keep the internal debug topic compatible with the former /insight
    // contract: only the stable fN/kN labels are published.
    visionPublisher_(labels);
  }

  if (insightPublisher_ && !publishedTargetEventThisFrame) {
    json aggregateTargets = json::array();
    insightPublisher_({
        {"timestamp_ms",
         static_cast<std::int64_t>(std::chrono::duration_cast<
             std::chrono::milliseconds>(std::chrono::system_clock::now()
                                            .time_since_epoch())
                                          .count())},
        {"obstacle_detected", safeState ? safeState->obstacle : false},
        {"obstacle_distance_cm",
         safeState && safeState->obstacle ? safeState->distanceCm : 0},
        {"target_detected", false},
        {"targets", aggregateTargets}});
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
        {{"condition", "stable f/k track and stage2 refresh is due"},
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
         {"to", "configured knife recognition"}},
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
