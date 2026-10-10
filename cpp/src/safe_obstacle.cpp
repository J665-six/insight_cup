#include "insight_cup/safe_obstacle.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <opencv2/imgproc.hpp>

namespace insight_cup {

namespace {

std::vector<cv::Point2f> readPolygon(const json& value) {
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

std::vector<cv::Point2f> readRecordPolygon(const json& record) {
  std::vector<cv::Point2f> polygon = readPolygon(record.value("polygon", json()));
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
  if (!bbox.is_array() || bbox.size() < 4) {
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

double polygonArea(const std::vector<cv::Point2f>& polygon) {
  if (polygon.size() < 3) {
    return 0.0;
  }
  return std::abs(cv::contourArea(polygon));
}

cv::Point2f polygonCenter(const std::vector<cv::Point2f>& polygon) {
  if (polygon.empty()) {
    return {};
  }
  const cv::Moments moments = cv::moments(polygon);
  if (std::abs(moments.m00) > 1e-6) {
    return {static_cast<float>(moments.m10 / moments.m00),
            static_cast<float>(moments.m01 / moments.m00)};
  }
  cv::Point2f center;
  for (const auto& point : polygon) {
    center += point;
  }
  center *= 1.0F / static_cast<float>(polygon.size());
  return center;
}

}  // namespace

json SafeObstacleState::toJson() const {
  const std::string message =
      status == "unconfigured"
          ? "安全模块未配置"
          : status == "disabled" ? "安全模块已禁用"
                                  : obstacle ? "遇到障碍物" : "前方无障碍物";
  return {
      {"obstacle", obstacle},
      {"status", status},
      {"message", message},
      {"distance_cm", distanceCm},
      {"distance_m", distanceM},
      {"stable_seconds", stableSeconds},
      {"stable_frames", stableFrames},
      {"confidence", confidence},
      {"bbox_xyxy", bbox.valid() ? bbox.toJson() : json(nullptr)},
      {"configured", configured},
      {"config_path", configPath},
  };
}

SafeObstacleMonitor::SafeObstacleMonitor(const RuntimeConfig& config,
                                         Logger& logger)
    : config_(config), logger_(logger) {
  loadConfiguration();
  lastState_.configured = configured_;
  lastState_.configPath = loadedConfigPath_.empty()
                              ? displayPath(config_.safeZoneConfig)
                              : displayPath(loadedConfigPath_);
  if (!configured_) {
    lastState_.status = config_.safeEnabled ? "unconfigured" : "disabled";
  }
}

void SafeObstacleMonitor::loadConfiguration() {
  if (!config_.safeEnabled) {
    logger_.info("SAFE disabled by configuration");
    return;
  }

  const std::filesystem::path requested = config_.safeZoneConfig;
  // The current calibration is split into two files because the same
  // distance produces different pixel sizes for horizontal and vertical
  // obstacles.  Keep the old single-file format as a compatibility path.
  const auto calibrationDirectory = requested.empty()
                                        ? std::filesystem::path()
                                        : requested.parent_path();
  const auto horizontalPath = calibrationDirectory / "obstacle_horizontal.json";
  const auto verticalPath = calibrationDirectory / "obstacle_vertical.json";
  std::error_code pixelError;
  const bool wantsPixelCalibration =
      requested.filename() == "obstacle_distance_new.json" ||
      requested.filename() == "obstacle_horizontal.json" ||
      requested.filename() == "obstacle_vertical.json" || requested.empty();
  if (wantsPixelCalibration &&
      std::filesystem::is_regular_file(horizontalPath, pixelError) &&
      std::filesystem::is_regular_file(verticalPath, pixelError)) {
    try {
      const auto roiPath = calibrationDirectory / "obstacle_distance.json";
      loadPixelCalibration(horizontalPath, verticalPath, roiPath);
      if (configured_) {
        return;
      }
    } catch (const std::exception& exception) {
      logger_.warning("SAFE pixel calibration ignored error=" +
                      std::string(exception.what()));
      obstacleRoi_.clear();
      horizontalSamples_.clear();
      verticalSamples_.clear();
      configured_ = false;
    }
  }
  std::vector<std::filesystem::path> candidates;
  std::error_code error;
  if (!requested.empty() &&
      std::filesystem::is_regular_file(requested, error)) {
    // An explicitly supplied file has priority and is the only file tried.
    // This prevents a typo or stale hand-written calibration from silently
    // selecting an unrelated recording in the same directory.
    candidates.push_back(requested);
  } else if (!requested.empty() && requested.filename() == "safe_zones.json") {
    // The default file is optional.  When it is absent, try saved distance
    // recordings from newest to oldest and use the first complete one.
    const auto directory = requested.parent_path();
    error.clear();
    if (std::filesystem::is_directory(directory, error)) {
      std::vector<std::pair<std::filesystem::file_time_type,
                            std::filesystem::path>> timestamped;
      std::error_code iteratorError;
      for (const auto& entry :
           std::filesystem::directory_iterator(directory, iteratorError)) {
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

  if (candidates.empty()) {
    logger_.warning("SAFE no zone config found; expected " +
                    displayPath(requested));
    return;
  }

  std::vector<cv::Point2f> partialRoi;
  std::filesystem::path partialPath;
  for (const auto& candidate : candidates) {
    // A failed candidate may have populated one member before discovering a
    // malformed record.  Clear all state before trying the next recording.
    obstacleRoi_.clear();
    zones_.clear();
    horizontalSamples_.clear();
    verticalSamples_.clear();
    loadedConfigPath_.clear();
    configured_ = false;
    try {
      std::ifstream stream(candidate);
      if (!stream) {
        throw std::runtime_error("cannot open file");
      }
      json payload;
      stream >> payload;
      loadPayload(payload, candidate);
      if (configured_) {
        return;
      }
      if (!obstacleRoi_.empty()) {
        partialRoi = obstacleRoi_;
        partialPath = candidate;
      }
      throw std::runtime_error("no usable safe zones");
    } catch (const std::exception& exception) {
      logger_.warning("SAFE zone config ignored path=" +
                      displayPath(candidate) + " error=" + exception.what());
    }
  }

  if (!partialRoi.empty()) {
    obstacleRoi_ = std::move(partialRoi);
    zones_.clear();
    loadedConfigPath_ = partialPath;
    configured_ = false;
    logger_.warning("SAFE retaining #1A ROI for display while distance "
                    "calibration remains incomplete");
    return;
  }
  obstacleRoi_.clear();
  zones_.clear();
  horizontalSamples_.clear();
  verticalSamples_.clear();
  loadedConfigPath_.clear();
  configured_ = false;
  logger_.warning("SAFE no complete zone config found; expected #1A ROI plus "
                  "#2A...#6A distance zones");
}

void SafeObstacleMonitor::loadPixelCalibration(
    const std::filesystem::path& horizontalPath,
    const std::filesystem::path& verticalPath,
    const std::filesystem::path& roiPath) {
  auto readPayload = [](const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
      throw std::runtime_error("cannot open " + path.string());
    }
    json payload;
    stream >> payload;
    return payload;
  };
  auto readSamples = [](const json& payload, const std::vector<int>& groupSizes) {
    std::vector<PixelSample> samples;
    const auto boxes = payload.value("boxes", json::array());
    if (!boxes.is_array()) {
      return samples;
    }
    std::size_t index = 0;
    for (std::size_t group = 0; group < groupSizes.size(); ++group) {
      const int distance = 100 - static_cast<int>(group) * 20;
      const int count = groupSizes[group];
      std::vector<PixelSample> groupSamples;
      for (int item = 0; item < count && index < boxes.size(); ++item, ++index) {
        const auto& value = boxes[index];
        const auto polygon = readRecordPolygon(value);
        double width = value.value("width_px", 0.0);
        double height = value.value("height_px", 0.0);
        if (width <= 0.0 || height <= 0.0) {
          const auto bbox = value.value("bbox", value.value("bbox_xyxy", json()));
          if (bbox.is_array() && bbox.size() >= 4) {
            width = bbox[2].get<double>() - bbox[0].get<double>();
            height = bbox[3].get<double>() - bbox[1].get<double>();
          } else if (!polygon.empty()) {
            const auto bounds = cv::boundingRect(polygon);
            width = bounds.width;
            height = bounds.height;
          }
        }
        const double area = width * height;
        if (width > 0.0 && height > 0.0 && area > 0.0) {
          groupSamples.push_back(
              {distance, std::sqrt(area), width, height, area});
        }
      }
      if (!groupSamples.empty()) {
        PixelSample average;
        average.distanceCm = distance;
        for (const auto& sample : groupSamples) {
          average.scale += sample.scale;
          average.width += sample.width;
          average.height += sample.height;
          average.area += sample.area;
        }
        const double divisor = static_cast<double>(groupSamples.size());
        average.scale /= divisor;
        average.width /= divisor;
        average.height /= divisor;
        average.area /= divisor;
        samples.push_back(average);
      }
    }
    return samples;
  };

  const json horizontal = readPayload(horizontalPath);
  const json vertical = readPayload(verticalPath);
  horizontalSamples_ = readSamples(horizontal, {3, 3, 3, 3, 3});
  verticalSamples_ = readSamples(vertical, {3, 4, 3, 3, 3});
  if (horizontalSamples_.size() < 5 || verticalSamples_.size() < 5) {
    throw std::runtime_error("pixel calibration needs five distance groups");
  }

  // Prefer the latest explicit trapezoid calibration. The older
  // obstacle_distance.json stores only a rectangular bbox for #1A and is
  // retained solely as a compatibility fallback.
  std::vector<std::filesystem::path> roiCandidates = {
      roiPath.parent_path() / "obstacle_distance_new.json", roiPath};
  for (const auto& candidate : roiCandidates) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(candidate, error)) {
      continue;
    }
    const json payload = readPayload(candidate);
    for (const auto& value : payload.value("boxes", json::array())) {
      if (value.value("id", 0) == 1) {
        obstacleRoi_ = readRecordPolygon(value);
        break;
      }
    }
    if (obstacleRoi_.size() >= 3) {
      break;
    }
  }
  if (obstacleRoi_.size() < 3) {
    throw std::runtime_error("pixel calibration has no #1A obstacle ROI");
  }

  double horizontalAspect = 0.0;
  for (const auto& sample : horizontalSamples_) {
    horizontalAspect += sample.width / sample.height;
  }
  horizontalAspect /= static_cast<double>(horizontalSamples_.size());
  double verticalAspect = 0.0;
  for (const auto& sample : verticalSamples_) {
    verticalAspect += sample.width / sample.height;
  }
  verticalAspect /= static_cast<double>(verticalSamples_.size());
  shapeAspectThreshold_ = std::clamp(0.5 * (horizontalAspect + verticalAspect),
                                     1.05, 1.50);
  configured_ = true;
  const auto displayCalibrationPath =
      horizontalPath.parent_path() / "obstacle_distance_new.json";
  std::error_code displayPathError;
  loadedConfigPath_ = std::filesystem::is_regular_file(
                          displayCalibrationPath, displayPathError)
                          ? displayCalibrationPath
                          : horizontalPath;
  logger_.info("SAFE pixel calibration horizontal=" +
               std::to_string(horizontalSamples_.size()) +
               " vertical=" + std::to_string(verticalSamples_.size()) +
               " aspect_threshold=" + std::to_string(shapeAspectThreshold_));
}

void SafeObstacleMonitor::loadPayload(const json& payload,
                                      const std::filesystem::path& path) {
  if (payload.value("schema", std::string()) == "insight.safe_zones.v1") {
    const auto roi = readPolygon(payload.value("obstacle_roi", json()));
    const auto zoneValues = payload.value("distance_zones", json::array());
    if (roi.size() < 3 || !zoneValues.is_array()) {
      throw std::runtime_error("safe_zones.v1 requires obstacle_roi and distance_zones");
    }
    std::vector<Zone> loaded;
    int fallbackId = 2;
    for (const auto& value : zoneValues) {
      const auto polygon = readPolygon(value.value("polygon", json()));
      if (polygon.size() < 3) {
        continue;
      }
      const int distanceCm = value.value("distance_cm", 0);
      if (distanceCm <= 0) {
        continue;
      }
      loaded.push_back({value.value("id", fallbackId++), distanceCm, polygon,
                        polygonCenter(polygon), polygonArea(polygon)});
    }
    if (loaded.empty()) {
      throw std::runtime_error("safe_zones.v1 contains no usable distance zones");
    }
    obstacleRoi_ = roi;
    zones_ = std::move(loaded);
    loadedConfigPath_ = path;
    configured_ = true;
    logger_.info("SAFE config path=" + displayPath(path) +
                 " roi_points=" + std::to_string(obstacleRoi_.size()) +
                 " zones=" + std::to_string(zones_.size()));
    return;
  }
  inferDistancePayload(payload, path);
}

void SafeObstacleMonitor::inferDistancePayload(
    const json& payload, const std::filesystem::path& path) {
  const auto values = payload.value("boxes", json::array());
  if (!values.is_array()) {
    throw std::runtime_error("distance config has no boxes array");
  }
  struct Candidate {
    int id = 0;
    std::vector<cv::Point2f> polygon;
    double area = 0.0;
    cv::Point2f center;
  };
  std::vector<Candidate> candidates;
  for (const auto& value : values) {
    const auto polygon = readRecordPolygon(value);
    const double area = polygonArea(polygon);
    if (area <= 0.0) {
      continue;
    }
    candidates.push_back({value.value("id", 0), polygon, area,
                          polygonCenter(polygon)});
  }
  if (candidates.size() < 2) {
    throw std::runtime_error(
        "distance config needs #1A ROI plus at least one distance zone");
  }
  // The distance tool labels the obstacle corridor #1A and the
  // person/knife corridor #7A. The obstacle ROI must be explicitly labelled;
  // an unlabeled polygon cannot safely determine which pixels to inspect.
  auto roiIterator = std::find_if(
      candidates.begin(), candidates.end(),
      [](const Candidate& candidate) { return candidate.id == 1; });
  if (roiIterator == candidates.end()) {
    throw std::runtime_error("distance config has no #1A obstacle ROI");
  }
  const int roiSourceId = roiIterator->id;
  const std::vector<cv::Point2f> obstacleRoi = roiIterator->polygon;
  candidates.erase(roiIterator);
  // #7A is the person/knife YOLO ROI and must never be interpreted as a
  // distance mark for the obstacle module.
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [](const Candidate& candidate) { return candidate.id == 7; }),
      candidates.end());
  const bool hasDistanceIds = !candidates.empty() &&
                               std::all_of(candidates.begin(), candidates.end(),
                                           [](const Candidate& candidate) {
                                             return candidate.id >= 2 &&
                                                    candidate.id <= 6;
                                           });
  if (hasDistanceIds) {
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& left, const Candidate& right) {
                return left.id < right.id;
              });
  } else {
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& left,
                                                       const Candidate& right) {
      // If IDs were changed while editing, the 20 cm region is the largest
      // nested region and lies lowest in the image. Area is primary; y breaks
      // ties.
      if (std::abs(left.area - right.area) > 1.0) {
        return left.area > right.area;
      }
      return left.center.y > right.center.y;
    });
  }

  zones_.clear();
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (index >= 5) {
      break;
    }
    // The calibrated labels are ordered from far to near in the image:
    // #2A=100 cm, #3A=80 cm, #4A=60 cm, #5A=40 cm, #6A=20 cm.
    const int distanceCm = hasDistanceIds
                               ? 100 - static_cast<int>(index) * 20
                               : 20 + static_cast<int>(index) * 20;
    zones_.push_back({candidates[index].id, distanceCm,
                      std::move(candidates[index].polygon),
                      candidates[index].center, candidates[index].area});
  }
  loadedConfigPath_ = path;
  // Inferred distance logs are considered usable only when all five
  // distance marks (#2A...#6A) are present. A partial log still leaves the
  // ROI available in metadata/UI, but /safe remains unconfigured.
  configured_ = !obstacleRoi.empty() && zones_.size() >= 5;
  if (!configured_) {
    obstacleRoi_.clear();
    loadedConfigPath_ = path;
    logger_.warning("SAFE distance config has ROI but only " +
                    std::to_string(zones_.size()) +
                    " distance zones; need #2A...#6A");
    return;
  }
  obstacleRoi_ = obstacleRoi;
  logger_.info("SAFE inferred distance config path=" + displayPath(path) +
               " roi_source_id=" + std::to_string(roiSourceId) +
               " zones=" + std::to_string(zones_.size()) +
               " mapping=#2A100,#3A80,#4A60,#5A40,#6A20cm");
}

bool SafeObstacleMonitor::pointInPolygon(
    const cv::Point2f& point, const std::vector<cv::Point2f>& polygon) const {
  return polygon.size() >= 3 && cv::pointPolygonTest(polygon, point, false) >= 0.0;
}

std::optional<int> SafeObstacleMonitor::distanceForBox(const Box& box) const {
  if (!box.valid()) {
    return std::nullopt;
  }
  const cv::Point2f center(
      0.5F * static_cast<float>(box.x1 + box.x2),
      0.5F * static_cast<float>(box.y1 + box.y2));
  if (!pointInPolygon(center, obstacleRoi_)) {
    return std::nullopt;
  }

  const double width = static_cast<double>(box.width());
  const double height = static_cast<double>(box.height());
  const double scale = std::sqrt(width * height);
  const bool horizontal = width / std::max(1.0, height) >= shapeAspectThreshold_;
  const auto& samples = horizontal ? horizontalSamples_ : verticalSamples_;
  if (samples.size() < 2 || scale <= 0.0) {
    // Compatibility for a legacy single-file calibration.
    if (zones_.empty()) {
      return std::nullopt;
    }
    const Zone* closest = nullptr;
    float closestDistance = std::numeric_limits<float>::max();
    for (const auto& zone : zones_) {
      const float distance = cv::norm(center - zone.center);
      if (distance < closestDistance) {
        closestDistance = distance;
        closest = &zone;
      }
    }
    return closest ? std::optional<int>(closest->distanceCm) : std::nullopt;
  }

  // Samples are ordered from far to near. Interpolate in pixel scale and
  // clamp outside the calibrated 20-100 cm range.
  std::vector<PixelSample> ordered = samples;
  std::sort(ordered.begin(), ordered.end(),
            [](const PixelSample& left, const PixelSample& right) {
              return left.scale < right.scale;
            });
  double distance = static_cast<double>(ordered.front().distanceCm);
  if (scale <= ordered.front().scale) {
    distance = ordered.front().distanceCm;
  } else if (scale >= ordered.back().scale) {
    distance = ordered.back().distanceCm;
  } else {
    for (std::size_t index = 1; index < ordered.size(); ++index) {
      if (scale > ordered[index].scale) {
        continue;
      }
      const auto& low = ordered[index - 1];
      const auto& high = ordered[index];
      const double fraction = (scale - low.scale) /
                              std::max(1e-6, high.scale - low.scale);
      distance = low.distanceCm +
                 fraction * (high.distanceCm - low.distanceCm);
      break;
    }
  }
  const int rounded = static_cast<int>(std::lround(distance / 5.0) * 5.0);
  return std::clamp(rounded, 20, 100);
}

void SafeObstacleMonitor::resetCandidate() {
  candidateDistanceCm_.reset();
  distanceHistoryCm_.clear();
  filteredDistanceCm_.reset();
  candidateCenter_ = {};
  candidateBox_ = {};
  candidateConfidence_ = 0.0F;
  candidateFrames_ = 0;
  candidateStartSourceMs_.reset();
  candidateStartFrame_ = -1;
}

SafeObstacleState SafeObstacleMonitor::update(
    const DetectionFrame& frame, const cv::Size& frameSize, int processedIndex,
    std::optional<double> sourceTimeMs) {
  SafeObstacleState state;
  state.configured = configured_;
  state.configPath = loadedConfigPath_.empty()
                         ? displayPath(config_.safeZoneConfig)
                         : displayPath(loadedConfigPath_);
  if (!config_.safeEnabled) {
    state.status = "disabled";
    lastState_ = state;
    return state;
  }
  if (!configured_) {
    state.status = "unconfigured";
    lastState_ = state;
    return state;
  }

  const DetectedRegion* best = nullptr;
  std::optional<int> bestDistance;
  for (const auto& region : frame.regions) {
    if (region.handoff.majorClass != "b0" ||
        region.handoff.confidence < config_.safeConfidence ||
        !region.handoff.bbox.valid()) {
      continue;
    }
    const auto& box = region.handoff.bbox;
    const cv::Point2f center(
        0.5F * static_cast<float>(box.x1 + box.x2),
        0.5F * static_cast<float>(box.y1 + box.y2));
    if (center.x < 0.0F || center.y < 0.0F || center.x >= frameSize.width ||
        center.y >= frameSize.height) {
      continue;
    }
    const auto distance = distanceForBox(region.handoff.bbox);
    if (!distance || !best || region.handoff.confidence > best->handoff.confidence) {
      if (distance) {
        best = &region;
        bestDistance = distance;
      }
    }
  }

  if (!best || !bestDistance) {
    resetCandidate();
    state.status = "clear";
    lastState_ = state;
    return state;
  }

  const auto& box = best->handoff.bbox;
  const cv::Point2f center(
      0.5F * static_cast<float>(box.x1 + box.x2),
      0.5F * static_cast<float>(box.y1 + box.y2));
  const bool sameCandidate = candidateBox_.valid() &&
                             cv::norm(center - candidateCenter_) <=
                                 std::max(20.0F, 0.25F *
                                                     static_cast<float>(std::max(
                                                         frameSize.width,
                                                         frameSize.height)));
  if (!sameCandidate) {
    resetCandidate();
    candidateCenter_ = center;
    candidateBox_ = box;
    candidateConfidence_ = best->handoff.confidence;
    candidateFrames_ = 1;
    candidateStartSourceMs_ = sourceTimeMs;
    candidateStartFrame_ = processedIndex;
  } else {
    candidateCenter_ = center;
    candidateBox_ = box;
    candidateConfidence_ = std::max(candidateConfidence_, best->handoff.confidence);
    ++candidateFrames_;
  }

  // Pixel-only range estimates move between calibration bins when the car
  // moves. Keep a short robust history and an EMA so one jittery YOLO box
  // cannot make the reported distance jump or reset the candidate.
  distanceHistoryCm_.push_back(*bestDistance);
  while (distanceHistoryCm_.size() > 5) {
    distanceHistoryCm_.pop_front();
  }
  std::vector<int> orderedDistances(distanceHistoryCm_.begin(),
                                    distanceHistoryCm_.end());
  std::sort(orderedDistances.begin(), orderedDistances.end());
  const double medianDistance =
      orderedDistances[orderedDistances.size() / 2];
  if (!filteredDistanceCm_) {
    filteredDistanceCm_ = medianDistance;
  } else {
    constexpr double kDistanceEmaAlpha = 0.35;
    *filteredDistanceCm_ +=
        kDistanceEmaAlpha * (medianDistance - *filteredDistanceCm_);
  }
  candidateDistanceCm_ = static_cast<int>(std::lround(
      std::clamp(*filteredDistanceCm_, 20.0, 100.0) / 5.0) * 5.0);

  double stableSeconds = 0.0;
  bool usedSourceTimestamp = false;
  if (sourceTimeMs && candidateStartSourceMs_) {
    const double elapsedMs = *sourceTimeMs - *candidateStartSourceMs_;
    // Some video backends report a constant or non-monotonic POS_MSEC.  Use
    // the configured frame rate in that case so a valid obstacle can still
    // become stable after the requested duration.
    if (elapsedMs > 0.0) {
      stableSeconds = elapsedMs / 1000.0;
      usedSourceTimestamp = true;
    }
  }
  if (!usedSourceTimestamp && candidateStartFrame_ >= 0) {
    const double effectiveFps =
        static_cast<double>(std::max(1, config_.cameraFps)) /
        static_cast<double>(std::max(1, config_.videoStride));
    stableSeconds = static_cast<double>(processedIndex - candidateStartFrame_) /
                    std::max(1.0, effectiveFps);
  }
  // Obstacle confirmation is immediate. safeStableSeconds remains in the
  // diagnostic payload for compatibility, but is no longer a gate.
  state.status = "obstacle";
  state.obstacle = true;
  state.distanceCm = *candidateDistanceCm_;
  state.distanceM = static_cast<double>(state.distanceCm) / 100.0;
  state.stableSeconds = stableSeconds;
  state.stableFrames = candidateFrames_;
  state.confidence = candidateConfidence_;
  state.bbox = candidateBox_;
  lastState_ = state;
  return state;
}

json SafeObstacleMonitor::metadata() const {
  json zones = json::array();
  for (const auto& zone : zones_) {
    json polygon = json::array();
    for (const auto& point : zone.polygon) {
      polygon.push_back({point.x, point.y});
    }
    zones.push_back({{"source_id", zone.sourceId},
                     {"distance_cm", zone.distanceCm},
                     {"center", {zone.center.x, zone.center.y}},
                     {"polygon", polygon}});
  }
  json roi = json::array();
  for (const auto& point : obstacleRoi_) {
    roi.push_back({point.x, point.y});
  }
  json pixelCalibration = json::object();
  auto samplesToJson = [](const std::vector<PixelSample>& samples) {
    json result = json::array();
    for (const auto& sample : samples) {
      result.push_back({{"distance_cm", sample.distanceCm},
                        {"scale_sqrt_area", sample.scale},
                        {"width_px", sample.width},
                        {"height_px", sample.height},
                        {"area_px2", sample.area}});
    }
    return result;
  };
  if (!horizontalSamples_.empty() || !verticalSamples_.empty()) {
    pixelCalibration = {{"shape_aspect_threshold", shapeAspectThreshold_},
                         {"horizontal", samplesToJson(horizontalSamples_)},
                         {"vertical", samplesToJson(verticalSamples_)}};
  }
  return {{"enabled", config_.safeEnabled},
          {"configured", configured_},
          {"config_path", loadedConfigPath_.empty()
                               ? displayPath(config_.safeZoneConfig)
                               : displayPath(loadedConfigPath_)},
          {"stable_seconds", config_.safeStableSeconds},
          {"confidence", config_.safeConfidence},
          {"obstacle_roi", roi},
          {"distance_zones", zones},
          {"pixel_calibration", pixelCalibration}};
}

}  // namespace insight_cup
