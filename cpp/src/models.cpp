#include "insight_cup/models.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace insight_cup {

namespace {

// Pre/post-processing below is an equivalent C++ port of the fixed upstream
// InsightFace and PaddleClas files listed in cpp/SOURCE_MANIFEST.json.

using Clock = std::chrono::steady_clock;

double elapsedMilliseconds(const Clock::time_point& started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

float floatIou(float leftX1, float leftY1, float leftX2, float leftY2,
               float rightX1, float rightY1, float rightX2, float rightY2) {
  const float x1 = std::max(leftX1, rightX1);
  const float y1 = std::max(leftY1, rightY1);
  const float x2 = std::min(leftX2, rightX2);
  const float y2 = std::min(leftY2, rightY2);
  const float intersection = std::max(0.0F, x2 - x1) *
                             std::max(0.0F, y2 - y1);
  if (intersection <= 0.0F) {
    return 0.0F;
  }
  const float leftArea = std::max(0.0F, leftX2 - leftX1) *
                         std::max(0.0F, leftY2 - leftY1);
  const float rightArea = std::max(0.0F, rightX2 - rightX1) *
                          std::max(0.0F, rightY2 - rightY1);
  const float unionArea = leftArea + rightArea - intersection;
  return unionArea > 0.0F ? intersection / unionArea : 0.0F;
}

std::string collapseYoloClass(const std::string& rawName) {
  if (rawName == "b0") {
    return "b0";
  }
  if (rawName.size() > 1 && (rawName[0] == 'f' || rawName[0] == 'k') &&
      std::all_of(rawName.begin() + 1, rawName.end(),
                  [](unsigned char character) { return std::isdigit(character); })) {
    return rawName.substr(0, 1);
  }
  return {};
}

std::vector<std::string> parseYoloNames(const std::optional<std::string>& metadata) {
  std::map<int, std::string> parsed;
  if (metadata) {
    const std::regex item(R"(([0-9]+)\s*:\s*['\"]([^'\"]+)['\"])");
    for (std::sregex_iterator match(metadata->begin(), metadata->end(), item), end;
         match != end; ++match) {
      parsed[std::stoi((*match)[1].str())] = (*match)[2].str();
    }
  }
  if (parsed.empty()) {
    for (int index = 1; index <= 10; ++index) {
      parsed[index - 1] = "k" + std::to_string(index);
      parsed[index + 9] = "f" + std::to_string(index);
    }
    parsed[20] = "b0";
  }
  if (parsed.rbegin()->first + 1 != static_cast<int>(parsed.size())) {
    throw std::runtime_error("YOLO metadata class IDs are not contiguous");
  }
  std::vector<std::string> names(parsed.size());
  for (const auto& [index, name] : parsed) {
    names.at(static_cast<std::size_t>(index)) = name;
  }
  return names;
}

Box clampBox(float x1, float y1, float x2, float y2, int width, int height) {
  Box box;
  box.x1 = std::clamp(static_cast<int>(std::nearbyint(x1)), 0,
                      std::max(width - 1, 0));
  box.y1 = std::clamp(static_cast<int>(std::nearbyint(y1)), 0,
                      std::max(height - 1, 0));
  box.x2 = std::clamp(static_cast<int>(std::nearbyint(x2)), 0, width);
  box.y2 = std::clamp(static_cast<int>(std::nearbyint(y2)), 0, height);
  return box;
}

std::vector<float> blobData(const cv::Mat& image, double scale,
                            const cv::Size& size, const cv::Scalar& mean,
                            bool swapRedBlue) {
  cv::Mat blob = cv::dnn::blobFromImage(image, scale, size, mean, swapRedBlue,
                                        false, CV_32F);
  if (!blob.isContinuous()) {
    blob = blob.clone();
  }
  const float* begin = blob.ptr<float>();
  return std::vector<float>(begin, begin + blob.total());
}

float l2Normalize(std::vector<float>& values) {
  float squaredNorm = 0.0F;
  for (float value : values) {
    squaredNorm += value * value;
  }
  const float norm = std::sqrt(squaredNorm);
  if (!std::isfinite(norm) || norm <= 0.0F) {
    throw std::runtime_error("Model produced an invalid embedding");
  }
  for (float& value : values) {
    value /= norm;
  }
  return norm;
}

cv::Mat similarityTransform(const std::array<cv::Point2f, 5>& source) {
  static const std::array<cv::Point2f, 5> destination = {
      cv::Point2f(38.2946F, 51.6963F), cv::Point2f(73.5318F, 51.5014F),
      cv::Point2f(56.0252F, 71.7366F), cv::Point2f(41.5493F, 92.3655F),
      cv::Point2f(70.7299F, 92.2041F)};

  cv::Point2d sourceMean(0.0, 0.0);
  cv::Point2d destinationMean(0.0, 0.0);
  for (std::size_t index = 0; index < source.size(); ++index) {
    sourceMean += cv::Point2d(source[index].x, source[index].y);
    destinationMean += cv::Point2d(destination[index].x, destination[index].y);
  }
  sourceMean *= 1.0 / source.size();
  destinationMean *= 1.0 / destination.size();

  double denominator = 0.0;
  double aNumerator = 0.0;
  double bNumerator = 0.0;
  for (std::size_t index = 0; index < source.size(); ++index) {
    const double x = source[index].x - sourceMean.x;
    const double y = source[index].y - sourceMean.y;
    const double u = destination[index].x - destinationMean.x;
    const double v = destination[index].y - destinationMean.y;
    denominator += x * x + y * y;
    aNumerator += x * u + y * v;
    bNumerator += x * v - y * u;
  }
  if (denominator <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error("Face landmarks cannot define an alignment transform");
  }
  const double a = aNumerator / denominator;
  const double b = bNumerator / denominator;
  const double tx = destinationMean.x - a * sourceMean.x + b * sourceMean.y;
  const double ty = destinationMean.y - b * sourceMean.x - a * sourceMean.y;
  return (cv::Mat_<double>(2, 3) << a, -b, tx, b, a, ty);
}

cv::Mat rotateExpanded(const cv::Mat& image, float degrees) {
  const cv::Point2f center(image.cols * 0.5F, image.rows * 0.5F);
  cv::Mat matrix = cv::getRotationMatrix2D(center, degrees, 1.0);
  const double cosine = std::abs(matrix.at<double>(0, 0));
  const double sine = std::abs(matrix.at<double>(0, 1));
  const int outputWidth = std::max(
      1, static_cast<int>(std::nearbyint(image.rows * sine + image.cols * cosine)));
  const int outputHeight = std::max(
      1, static_cast<int>(std::nearbyint(image.rows * cosine + image.cols * sine)));
  matrix.at<double>(0, 2) += outputWidth * 0.5 - center.x;
  matrix.at<double>(1, 2) += outputHeight * 0.5 - center.y;
  cv::Mat rotated;
  cv::warpAffine(image, rotated, matrix, cv::Size(outputWidth, outputHeight),
                 cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  return rotated;
}

std::vector<std::string> loadKnifeLabels(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Knife labels not found: " + path.string());
  }
  std::map<int, std::string> indexed;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream parser(line);
    int index = -1;
    std::string label;
    parser >> index >> label;
    if (index < 0 || label.empty()) {
      throw std::runtime_error("Invalid knife label line: " + line);
    }
    indexed[index] = label;
  }
  std::vector<std::string> labels(indexed.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const auto found = indexed.find(static_cast<int>(index));
    if (found == indexed.end()) {
      throw std::runtime_error("Knife labels must be contiguous from zero");
    }
    labels[index] = found->second;
  }
  for (int index = 1; index <= 10; ++index) {
    if (labels.size() < static_cast<std::size_t>(index) ||
        labels[index - 1] != "k" + std::to_string(index)) {
      throw std::runtime_error("The first ten knife labels must be k1...k10");
    }
  }
  return labels;
}

}  // namespace

float bboxIou(const Box& left, const Box& right) {
  return floatIou(static_cast<float>(left.x1), static_cast<float>(left.y1),
                  static_cast<float>(left.x2), static_cast<float>(left.y2),
                  static_cast<float>(right.x1), static_cast<float>(right.y1),
                  static_cast<float>(right.x2), static_cast<float>(right.y2));
}

cv::Mat squareContextCrop(const cv::Mat& image, const Box& bbox,
                          float contextScale, int fillValue) {
  if (image.empty() || image.type() != CV_8UC3 || !bbox.valid() ||
      contextScale < 1.0F) {
    throw std::runtime_error("Invalid square context crop input");
  }
  const float centerX = (bbox.x1 + bbox.x2) * 0.5F;
  const float centerY = (bbox.y1 + bbox.y2) * 0.5F;
  const int side = std::max(
      2, static_cast<int>(std::ceil(std::max(bbox.width(), bbox.height()) *
                                    contextScale)));
  const int left = static_cast<int>(std::floor(centerX - side * 0.5F));
  const int top = static_cast<int>(std::floor(centerY - side * 0.5F));
  const int sourceLeft = std::max(0, left);
  const int sourceTop = std::max(0, top);
  const int sourceRight = std::min(image.cols, left + side);
  const int sourceBottom = std::min(image.rows, top + side);
  cv::Mat crop(side, side, CV_8UC3, cv::Scalar::all(fillValue));
  if (sourceRight > sourceLeft && sourceBottom > sourceTop) {
    const cv::Rect source(sourceLeft, sourceTop, sourceRight - sourceLeft,
                          sourceBottom - sourceTop);
    const cv::Rect destination(sourceLeft - left, sourceTop - top, source.width,
                               source.height);
    image(source).copyTo(crop(destination));
  }
  return crop;
}

YoloDetector::YoloDetector(const RuntimeConfig& config)
    : weights_(config.weights),
      confidence_(config.confidence),
      iou_(config.iou),
      imageSize_(config.imageSize),
      cpuThreads_(config.yoloCpuThreads),
      session_(std::make_unique<OnnxSession>(config.weights, config.yoloCpuThreads,
                                             config.provider)) {
  if (config.yoloDevice != "auto" && config.yoloDevice != "cpu") {
    throw std::runtime_error("Native YOLO currently supports --device auto or cpu");
  }
  const auto& shape = session_->inputShape();
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 ||
      shape[2] != imageSize_ || shape[3] != imageSize_) {
    throw std::runtime_error("YOLO ONNX input shape does not match --imgsz");
  }
  names_ = parseYoloNames(session_->metadata("names"));
}

std::vector<YoloDetector::RawDetection> YoloDetector::predict(
    const cv::Mat& frame) const {
  const float scale = std::min(static_cast<float>(imageSize_) / frame.rows,
                               static_cast<float>(imageSize_) / frame.cols);
  const int resizedWidth = std::max(
      1, static_cast<int>(std::nearbyint(frame.cols * static_cast<double>(scale))));
  const int resizedHeight = std::max(
      1, static_cast<int>(std::nearbyint(frame.rows * static_cast<double>(scale))));
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0,
             cv::INTER_LINEAR);
  const double horizontalPadding = (imageSize_ - resizedWidth) / 2.0;
  const double verticalPadding = (imageSize_ - resizedHeight) / 2.0;
  const int left = static_cast<int>(std::nearbyint(horizontalPadding - 0.1));
  const int right = static_cast<int>(std::nearbyint(horizontalPadding + 0.1));
  const int top = static_cast<int>(std::nearbyint(verticalPadding - 0.1));
  const int bottom = static_cast<int>(std::nearbyint(verticalPadding + 0.1));
  cv::Mat prepared;
  cv::copyMakeBorder(resized, prepared, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  std::vector<float> input = blobData(prepared, 1.0 / 255.0,
                                      cv::Size(imageSize_, imageSize_),
                                      cv::Scalar(), true);
  const auto outputs = session_->run(input, {1, 3, imageSize_, imageSize_});
  if (outputs.size() != 1 || outputs[0].shape.size() != 3 ||
      outputs[0].shape[0] != 1) {
    throw std::runtime_error("Unexpected YOLO ONNX output shape");
  }
  const TensorOutput& output = outputs[0];
  const std::size_t columns = 4 + names_.size();
  bool channelFirst = false;
  std::size_t predictionCount = 0;
  if (output.shape[1] == static_cast<int64_t>(columns)) {
    channelFirst = true;
    predictionCount = static_cast<std::size_t>(output.shape[2]);
  } else if (output.shape[2] == static_cast<int64_t>(columns)) {
    predictionCount = static_cast<std::size_t>(output.shape[1]);
  } else {
    throw std::runtime_error("YOLO output class count does not match metadata");
  }
  auto valueAt = [&](std::size_t row, std::size_t column) {
    return channelFirst ? output.data[column * predictionCount + row]
                        : output.data[row * columns + column];
  };

  std::vector<RawDetection> candidates;
  for (std::size_t row = 0; row < predictionCount; ++row) {
    int bestClass = 0;
    float bestScore = valueAt(row, 4);
    for (std::size_t classIndex = 1; classIndex < names_.size(); ++classIndex) {
      const float score = valueAt(row, 4 + classIndex);
      if (score > bestScore) {
        bestScore = score;
        bestClass = static_cast<int>(classIndex);
      }
    }
    if (bestScore < confidence_) {
      continue;
    }
    const float centerX = valueAt(row, 0);
    const float centerY = valueAt(row, 1);
    const float width = valueAt(row, 2);
    const float height = valueAt(row, 3);
    candidates.push_back({centerX - width * 0.5F, centerY - height * 0.5F,
                          centerX + width * 0.5F, centerY + height * 0.5F,
                          bestScore, bestClass});
  }

  std::vector<RawDetection> kept;
  for (std::size_t classIndex = 0; classIndex < names_.size(); ++classIndex) {
    std::vector<std::size_t> indexes;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      if (candidates[index].classId == static_cast<int>(classIndex)) {
        indexes.push_back(index);
      }
    }
    std::sort(indexes.begin(), indexes.end(), [&](std::size_t leftIndex,
                                                   std::size_t rightIndex) {
      return candidates[leftIndex].confidence > candidates[rightIndex].confidence;
    });
    while (!indexes.empty()) {
      const RawDetection selected = candidates[indexes.front()];
      kept.push_back(selected);
      indexes.erase(indexes.begin());
      indexes.erase(std::remove_if(indexes.begin(), indexes.end(),
                                   [&](std::size_t index) {
                                     const auto& candidate = candidates[index];
                                     return floatIou(
                                                selected.x1, selected.y1, selected.x2,
                                                selected.y2, candidate.x1, candidate.y1,
                                                candidate.x2, candidate.y2) > iou_;
                                   }),
                    indexes.end());
    }
  }
  std::sort(kept.begin(), kept.end(), [](const auto& leftValue,
                                         const auto& rightValue) {
    return leftValue.confidence > rightValue.confidence;
  });
  if (kept.size() > 300) {
    kept.resize(300);
  }
  for (auto& detection : kept) {
    detection.x1 = (detection.x1 - left) / scale;
    detection.x2 = (detection.x2 - left) / scale;
    detection.y1 = (detection.y1 - top) / scale;
    detection.y2 = (detection.y2 - top) / scale;
  }
  return kept;
}

DetectionFrame YoloDetector::detect(const cv::Mat& frame, int frameIndex) const {
  if (frame.empty() || frame.type() != CV_8UC3) {
    throw std::runtime_error("YOLO input must be a non-empty BGR image");
  }
  const auto started = Clock::now();
  const auto values = predict(frame);
  DetectionFrame result;
  result.inferenceMs = elapsedMilliseconds(started);
  for (std::size_t detectionIndex = 0; detectionIndex < values.size();
       ++detectionIndex) {
    const auto& value = values[detectionIndex];
    if (value.classId < 0 ||
        value.classId >= static_cast<int>(names_.size())) {
      continue;
    }
    const std::string rawName = names_[value.classId];
    const std::string majorClass = collapseYoloClass(rawName);
    if (majorClass.empty()) {
      result.ignoredClasses.push_back(rawName);
      continue;
    }
    const Box bbox = clampBox(value.x1, value.y1, value.x2, value.y2,
                              frame.cols, frame.rows);
    if (!bbox.valid()) {
      continue;
    }
    std::ostringstream id;
    id << "frame" << std::setfill('0') << std::setw(6) << frameIndex << ":det"
       << std::setw(3) << detectionIndex;
    Stage1Detection detection;
    detection.detectionId = id.str();
    detection.frameIndex = frameIndex;
    detection.majorClass = majorClass;
    detection.confidence = value.confidence;
    detection.bbox = bbox;
    detection.crop = majorClass == "k"
                         ? squareContextCrop(frame, bbox, 1.15F)
                         : frame(cv::Rect(bbox.x1, bbox.y1, bbox.width(),
                                          bbox.height()))
                               .clone();
    result.regions.push_back(
        {std::move(detection), {value.classId, rawName}});
  }

  std::sort(result.regions.begin(), result.regions.end(),
            [](const auto& left, const auto& right) {
              return left.handoff.confidence > right.handoff.confidence;
            });
  std::vector<DetectedRegion> deduplicated;
  for (auto& region : result.regions) {
    const bool duplicate = std::any_of(
        deduplicated.begin(), deduplicated.end(), [&](const auto& existing) {
          return existing.handoff.majorClass == region.handoff.majorClass &&
                 bboxIou(existing.handoff.bbox, region.handoff.bbox) >= iou_;
        });
    if (!duplicate) {
      deduplicated.push_back(std::move(region));
    }
  }
  result.regions = std::move(deduplicated);
  return result;
}

json YoloDetector::metadata() const {
  json classes = json::object();
  for (std::size_t index = 0; index < names_.size(); ++index) {
    classes[std::to_string(index)] = names_[index];
  }
  return {
      {"backend", "onnxruntime_cpp"},
      {"weights", weights_.string()},
      {"raw_classes", classes},
      {"public_classes", {"b0", "f", "k"}},
      {"confidence", confidence_},
      {"iou", iou_},
      {"major_class_nms_iou", iou_},
      {"image_size", imageSize_},
      {"device", "cpu"},
      {"cpu_threads", cpuThreads_},
  };
}

FaceRecognizer::FaceRecognizer(const RuntimeConfig& config)
    : gallery_(config.faceGallery),
      modelDirectory_(config.faceModelRoot / "models" / config.faceModelName),
      galleryPath_(config.faceGallery),
      threshold_(config.faceThreshold),
      minMargin_(config.faceMinMargin),
      detectionThreshold_(config.faceDetectionThreshold),
      detectionSize_(config.faceDetectionSize),
      upsampleMinSide_(config.faceUpsampleMinSide),
      rotationRetryDegrees_(config.faceRotationRetryDegrees),
      cpuThreads_(config.faceCpuThreads) {
  detector_ = std::make_unique<OnnxSession>(modelDirectory_ / "det_10g.onnx",
                                            cpuThreads_, config.provider);
  recognizer_ = std::make_unique<OnnxSession>(modelDirectory_ / "w600k_r50.onnx",
                                              cpuThreads_, config.provider);
  if (gallery_.dimensions() != 512) {
    throw std::runtime_error("Face gallery must contain 512-dimensional embeddings");
  }
}

std::vector<FaceRecognizer::FaceDetection> FaceRecognizer::detectFaces(
    const cv::Mat& image) const {
  const float imageRatio = static_cast<float>(image.rows) / image.cols;
  const float modelRatio = 1.0F;
  int resizedWidth = detectionSize_;
  int resizedHeight = detectionSize_;
  if (imageRatio > modelRatio) {
    resizedWidth = static_cast<int>(detectionSize_ / imageRatio);
  } else {
    resizedHeight = static_cast<int>(detectionSize_ * imageRatio);
  }
  resizedWidth = std::max(1, resizedWidth);
  resizedHeight = std::max(1, resizedHeight);
  const float detectionScale = static_cast<float>(resizedHeight) / image.rows;
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight));
  cv::Mat prepared(detectionSize_, detectionSize_, CV_8UC3, cv::Scalar::all(0));
  resized.copyTo(prepared(cv::Rect(0, 0, resizedWidth, resizedHeight)));
  std::vector<float> input = blobData(
      prepared, 1.0 / 128.0, cv::Size(detectionSize_, detectionSize_),
      cv::Scalar(127.5, 127.5, 127.5), true);
  const auto outputs = detector_->run(input, {1, 3, detectionSize_, detectionSize_});
  if (outputs.size() != 9) {
    throw std::runtime_error("SCRFD model must return nine outputs");
  }

  std::vector<FaceDetection> candidates;
  const std::array<int, 3> strides = {8, 16, 32};
  for (std::size_t level = 0; level < strides.size(); ++level) {
    const int stride = strides[level];
    const auto& scores = outputs[level].data;
    const auto& boxes = outputs[level + 3].data;
    const auto& landmarks = outputs[level + 6].data;
    const int height = detectionSize_ / stride;
    const int width = detectionSize_ / stride;
    const std::size_t expected = static_cast<std::size_t>(height * width * 2);
    if (scores.size() != expected || boxes.size() != expected * 4 ||
        landmarks.size() != expected * 10) {
      throw std::runtime_error("Unexpected SCRFD feature-map shape");
    }
    for (std::size_t index = 0; index < expected; ++index) {
      const float score = scores[index];
      if (score < detectionThreshold_) {
        continue;
      }
      const std::size_t location = index / 2;
      const float anchorX = static_cast<float>(location % width) * stride;
      const float anchorY = static_cast<float>(location / width) * stride;
      FaceDetection face;
      face.score = score;
      face.bbox = {
          (anchorX - boxes[index * 4] * stride) / detectionScale,
          (anchorY - boxes[index * 4 + 1] * stride) / detectionScale,
          (anchorX + boxes[index * 4 + 2] * stride) / detectionScale,
          (anchorY + boxes[index * 4 + 3] * stride) / detectionScale,
      };
      for (std::size_t point = 0; point < face.keypoints.size(); ++point) {
        face.keypoints[point] = cv::Point2f(
            (anchorX + landmarks[index * 10 + point * 2] * stride) /
                detectionScale,
            (anchorY + landmarks[index * 10 + point * 2 + 1] * stride) /
                detectionScale);
      }
      candidates.push_back(face);
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& left,
                                                      const auto& right) {
    return left.score > right.score;
  });
  std::vector<FaceDetection> kept;
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& selected : kept) {
      const float xx1 = std::max(candidate.bbox.x1, selected.bbox.x1);
      const float yy1 = std::max(candidate.bbox.y1, selected.bbox.y1);
      const float xx2 = std::min(candidate.bbox.x2, selected.bbox.x2);
      const float yy2 = std::min(candidate.bbox.y2, selected.bbox.y2);
      const float intersection = std::max(0.0F, xx2 - xx1 + 1.0F) *
                                 std::max(0.0F, yy2 - yy1 + 1.0F);
      const float candidateArea =
          (candidate.bbox.x2 - candidate.bbox.x1 + 1.0F) *
          (candidate.bbox.y2 - candidate.bbox.y1 + 1.0F);
      const float selectedArea =
          (selected.bbox.x2 - selected.bbox.x1 + 1.0F) *
          (selected.bbox.y2 - selected.bbox.y1 + 1.0F);
      const float overlap = intersection /
                            std::max(candidateArea + selectedArea - intersection,
                                     std::numeric_limits<float>::epsilon());
      if (overlap > 0.4F) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(candidate);
    }
  }
  return kept;
}

std::vector<float> FaceRecognizer::embed(
    const cv::Mat& image,
    const std::array<cv::Point2f, 5>& keypoints) const {
  const cv::Mat transform = similarityTransform(keypoints);
  cv::Mat aligned;
  cv::warpAffine(image, aligned, transform, cv::Size(112, 112), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar::all(0));
  std::vector<float> input = blobData(aligned, 1.0 / 127.5,
                                      cv::Size(112, 112),
                                      cv::Scalar(127.5, 127.5, 127.5), true);
  const auto outputs = recognizer_->run(input, {1, 3, 112, 112});
  if (outputs.size() != 1 || outputs[0].data.size() != 512) {
    throw std::runtime_error("ArcFace model must return one 512-dimensional vector");
  }
  std::vector<float> embedding = outputs[0].data;
  l2Normalize(embedding);
  return embedding;
}

FaceRecognizer::FaceAttempt FaceRecognizer::recognizeOnce(
    const cv::Mat& crop) const {
  cv::Mat prepared = crop;
  float upsampleScale = 1.0F;
  if (upsampleMinSide_ > 0 && std::min(crop.rows, crop.cols) < upsampleMinSide_) {
    upsampleScale = std::min(
        6.0F, static_cast<float>(upsampleMinSide_) / std::min(crop.rows, crop.cols));
    cv::resize(crop, prepared, cv::Size(), upsampleScale, upsampleScale,
               cv::INTER_CUBIC);
  }
  auto faces = detectFaces(prepared);
  if (faces.empty()) {
    Stage2Decision decision;
    decision.module = "face";
    decision.status = "no_face";
    return {std::move(decision), 0, -1.0F, -1.0F};
  }
  auto ranking = [&](const FaceDetection& face) {
    const float width = std::max(0.0F, face.bbox.x2 - face.bbox.x1);
    const float height = std::max(0.0F, face.bbox.y2 - face.bbox.y1);
    const float centerX = (face.bbox.x1 + face.bbox.x2) * 0.5F;
    const float centerY = (face.bbox.y1 + face.bbox.y2) * 0.5F;
    float offset = std::abs(centerX - prepared.cols * 0.5F) /
                   std::max(prepared.cols, 1);
    offset += std::abs(centerY - prepared.rows * 0.5F) /
              std::max(prepared.rows, 1);
    return width * height * std::max(0.5F, 1.0F - 0.15F * offset);
  };
  const auto selected = std::max_element(faces.begin(), faces.end(),
                                         [&](const auto& left, const auto& right) {
                                           return ranking(left) < ranking(right);
                                         });
  std::vector<float> embedding = embed(prepared, selected->keypoints);
  const GalleryMatch match =
      gallery_.matchByIdentityMax(embedding, threshold_, minMargin_);
  const float eyeDx = selected->keypoints[1].x - selected->keypoints[0].x;
  const float eyeDy = selected->keypoints[1].y - selected->keypoints[0].y;
  const float roll = std::atan2(eyeDy, eyeDx) * 180.0F /
                     static_cast<float>(CV_PI);

  Stage2Decision decision;
  decision.module = "face";
  decision.status = match.status;
  if (!match.predictedClass.empty()) {
    decision.predictedClass = match.predictedClass;
  }
  decision.candidateClass = match.candidateClass;
  decision.score = match.score;
  decision.scoreKind = "cosine_similarity";
  if (!match.secondBestClass.empty()) {
    decision.secondBestClass = match.secondBestClass;
    decision.secondBestScore = match.secondBestScore;
    decision.margin = match.margin;
  }
  decision.classScores = match.classScores;
  decision.metadata = {
      {"face_detection_confidence", selected->score},
      {"face_roll_degrees", roll},
      {"face_bbox_xyxy_in_crop",
       FloatBox{selected->bbox.x1 / upsampleScale,
                selected->bbox.y1 / upsampleScale,
                selected->bbox.x2 / upsampleScale,
                selected->bbox.y2 / upsampleScale}
           .toJson()},
  };
  const int statusRank = match.status == "matched"   ? 3
                         : match.status == "ambiguous" ? 2
                         : match.status == "unknown"   ? 1
                                                       : 0;
  return {std::move(decision), statusRank, match.score, selected->score};
}

Stage2Decision FaceRecognizer::recognize(const cv::Mat& crop) const {
  FaceAttempt best = recognizeOnce(crop);
  std::vector<float> attempts = {0.0F};
  float selectedDegrees = 0.0F;
  std::vector<float> retryAngles;
  if (rotationRetryDegrees_ > 0.0F) {
    if (best.decision.status == "no_face") {
      retryAngles = {-rotationRetryDegrees_, rotationRetryDegrees_};
    } else {
      const float detectionScore = best.decision.metadata.value(
          "face_detection_confidence", -1.0F);
      const float roll = best.decision.metadata.value("face_roll_degrees", 0.0F);
      const bool weak = best.decision.status == "unknown" ||
                        best.decision.status == "ambiguous" ||
                        detectionScore < 0.45F || best.score < threshold_ + 0.05F;
      if (weak && std::abs(roll) >= 8.0F) {
        retryAngles.push_back(
            std::clamp(roll, -rotationRetryDegrees_, rotationRetryDegrees_));
      }
    }
  }
  for (float degrees : retryAngles) {
    attempts.push_back(degrees);
    FaceAttempt candidate = recognizeOnce(rotateExpanded(crop, degrees));
    const auto candidateRank =
        std::make_tuple(candidate.rank, candidate.score, candidate.detectionScore);
    const auto bestRank = std::make_tuple(best.rank, best.score, best.detectionScore);
    if (candidateRank > bestRank) {
      best = std::move(candidate);
      selectedDegrees = degrees;
    }
  }
  best.decision.metadata["face_rotation_attempts_degrees"] = attempts;
  best.decision.metadata["face_selected_rotation_degrees"] = selectedDegrees;
  return best.decision;
}

json FaceRecognizer::metadata() const {
  return {
      {"backend", "insightface_cpp_onnx"},
      {"upstream_commit", "7fadd420c2351d0ffa8cac403421c1a3ed733365"},
      {"model_dir", modelDirectory_.string()},
      {"gallery", galleryPath_.string()},
      {"loaded_modules", {"detection", "recognition"}},
      {"providers", {"CPUExecutionProvider"}},
      {"det_size", {detectionSize_, detectionSize_}},
      {"det_thresh", detectionThreshold_},
      {"cpu_threads", cpuThreads_},
      {"identities", gallery_.identities()},
      {"reference_embeddings", gallery_.rows()},
      {"threshold", threshold_},
      {"min_margin", minMargin_},
      {"rotation_retry_degrees", rotationRetryDegrees_},
  };
}

KnifeRecognizer::KnifeRecognizer(const RuntimeConfig& config)
    : mode_(config.knifeMode), cpuThreads_(config.knifeCpuThreads) {
  labels_.reserve(10);
  for (int index = 1; index <= 10; ++index) {
    labels_.push_back("k" + std::to_string(index));
  }
  if (mode_ == "classification") {
    modelPath_ = config.knifeModelDir / "inference.onnx";
    labels_ = loadKnifeLabels(config.knifeModelDir / "labels.txt");
    threshold_ = config.knifeThreshold;
    minMargin_ = config.knifeMinMargin;
  } else {
    modelPath_ = config.knifeRetrievalModel;
    galleryPath_ = config.knifeGallery;
    gallery_ = std::make_unique<FeatureGallery>(galleryPath_);
    if (gallery_->dimensions() != 512) {
      throw std::runtime_error("Knife gallery must contain 512-dimensional embeddings");
    }
    threshold_ = config.knifeRetrievalThreshold;
    minMargin_ = config.knifeRetrievalMinMargin;
  }
  session_ = std::make_unique<OnnxSession>(modelPath_, cpuThreads_, config.provider);
}

std::vector<float> KnifeRecognizer::prepare(const cv::Mat& crop) const {
  if (crop.empty() || crop.type() != CV_8UC3) {
    throw std::runtime_error("Knife input must be a non-empty BGR image");
  }
  cv::Mat square = crop;
  if (crop.rows != crop.cols) {
    square = squareContextCrop(crop, Box{0, 0, crop.cols, crop.rows}, 1.0F);
  }
  cv::Mat resized;
  cv::resize(square, resized, cv::Size(224, 224), 0.0, 0.0, cv::INTER_LINEAR);
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
  const std::array<float, 3> means = {0.485F, 0.456F, 0.406F};
  const std::array<float, 3> standardDeviations = {0.229F, 0.224F, 0.225F};
  std::vector<cv::Mat> channels;
  cv::split(resized, channels);
  std::vector<float> output(3 * 224 * 224);
  const std::size_t plane = 224 * 224;
  for (std::size_t channel = 0; channel < channels.size(); ++channel) {
    channels[channel] = (channels[channel] - means[channel]) /
                        standardDeviations[channel];
    if (!channels[channel].isContinuous()) {
      channels[channel] = channels[channel].clone();
    }
    std::copy(channels[channel].ptr<float>(),
              channels[channel].ptr<float>() + plane,
              output.begin() + channel * plane);
  }
  return output;
}

Stage2Decision KnifeRecognizer::recognize(const cv::Mat& crop) const {
  std::vector<float> input = prepare(crop);
  const auto outputs = session_->run(input, {1, 3, 224, 224});
  if (outputs.size() != 1) {
    throw std::runtime_error("Knife model must return one output");
  }

  GalleryMatch match;
  std::string scoreKind;
  if (mode_ == "classification") {
    if (outputs[0].data.size() != labels_.size()) {
      throw std::runtime_error("Knife classifier output does not match labels");
    }
    float sum = 0.0F;
    for (std::size_t index = 0; index < labels_.size(); ++index) {
      const float score = outputs[0].data[index];
      if (!std::isfinite(score) || score < -1e-5F) {
        throw std::runtime_error("Knife classifier produced invalid probabilities");
      }
      sum += score;
      match.classScores[labels_[index]] = score;
    }
    if (std::abs(sum - 1.0F) > 1e-3F) {
      throw std::runtime_error("Knife classifier output is not softmax-normalized");
    }
    std::vector<std::size_t> indexes(labels_.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::partial_sort(indexes.begin(), indexes.begin() + std::min<std::size_t>(2, indexes.size()),
                      indexes.end(), [&](std::size_t left, std::size_t right) {
                        return outputs[0].data[left] > outputs[0].data[right];
                      });
    match.candidateClass = labels_[indexes[0]];
    match.score = outputs[0].data[indexes[0]];
    if (indexes.size() > 1) {
      match.secondBestClass = labels_[indexes[1]];
      match.secondBestScore = outputs[0].data[indexes[1]];
      match.margin = match.score - match.secondBestScore;
    }
    if (match.candidateClass == "other" || match.score < threshold_) {
      match.status = "unknown";
    } else if (!match.secondBestClass.empty() && match.margin < minMargin_) {
      match.status = "ambiguous";
    } else {
      match.status = "matched";
      match.predictedClass = match.candidateClass;
    }
    scoreKind = "softmax_confidence";
  } else {
    std::vector<float> embedding = outputs[0].data;
    if (embedding.size() != gallery_->dimensions()) {
      throw std::runtime_error("Knife retrieval embedding dimension mismatch");
    }
    l2Normalize(embedding);
    match = gallery_->matchByIdentityMax(embedding, threshold_, minMargin_);
    scoreKind = "cosine_similarity";
  }

  Stage2Decision decision;
  decision.module = "knife";
  decision.status = match.status;
  if (!match.predictedClass.empty()) {
    decision.predictedClass = match.predictedClass;
  }
  decision.candidateClass = match.candidateClass;
  decision.score = match.score;
  decision.scoreKind = scoreKind;
  if (!match.secondBestClass.empty()) {
    decision.secondBestClass = match.secondBestClass;
    decision.secondBestScore = match.secondBestScore;
    decision.margin = match.margin;
  }
  decision.classScores = match.classScores;
  return decision;
}

json KnifeRecognizer::metadata() const {
  json result = {
      {"backend", mode_ == "classification" ? "paddleclas_cpp_onnx"
                                               : "paddleclas_ppshitu_v2_cpp_onnx"},
      {"upstream_commit", "f1233c18455b8acde4fc42ab0bea575fa06daa8e"},
      {"model_path", modelPath_.string()},
      {"mode", mode_},
      {"labels", labels_},
      {"providers", {"CPUExecutionProvider"}},
      {"input_size", {224, 224}},
      {"threshold", threshold_},
      {"min_margin", minMargin_},
      {"cpu_threads", cpuThreads_},
  };
  if (mode_ == "classification") {
    result["architecture"] = "PPLCNetV2_base";
    result["score_kind"] = "softmax_confidence";
  } else {
    result["architecture"] = "GeneralRecognitionV2_PPLCNetV2_base";
    result["gallery"] = galleryPath_.string();
    result["gallery_references"] = gallery_->rows();
    result["embedding_size"] = gallery_->dimensions();
    result["index_method"] = "exact_inner_product_cpp";
    result["distance"] = "cosine_similarity";
  }
  return result;
}

RecognitionRouter::RecognitionRouter(std::shared_ptr<FaceRecognizer> face,
                                     std::shared_ptr<KnifeRecognizer> knife,
                                     Logger& logger)
    : face_(std::move(face)), knife_(std::move(knife)), logger_(logger) {}

Stage2Decision RecognitionRouter::route(
    const Stage1Detection& detection) const {
  try {
    if (detection.majorClass == "b0") {
      Stage2Decision decision;
      decision.module = "passthrough";
      decision.status = "passthrough";
      decision.predictedClass = "b0";
      decision.candidateClass = "b0";
      decision.score = detection.confidence;
      decision.scoreKind = "yolo_confidence";
      return decision;
    }
    if (detection.majorClass == "f") {
      return face_->recognize(detection.crop);
    }
    if (detection.majorClass == "k") {
      return knife_->recognize(detection.crop);
    }
    throw std::runtime_error("Unsupported major class: " + detection.majorClass);
  } catch (const std::exception& error) {
    const std::string module = detection.majorClass == "f"     ? "face"
                               : detection.majorClass == "k"   ? "knife"
                                                                 : "passthrough";
    logger_.error("Stage-2 failure: id=" + detection.detectionId +
                  " module=" + module + " error=" + error.what());
    Stage2Decision decision;
    decision.module = module;
    decision.status = "error";
    decision.metadata = {
        {"error", "RuntimeError: " + std::string(error.what())}};
    return decision;
  }
}

}  // namespace insight_cup
