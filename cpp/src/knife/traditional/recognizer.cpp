#include "insight_cup/knife/backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../common.hpp"

namespace insight_cup {

namespace {

void appendNormalizedBlock(std::vector<float>& output,
                           std::vector<float> values, float weight) {
  float squaredNorm = 0.0F;
  for (const float value : values) {
    squaredNorm += value * value;
  }
  const float norm = std::sqrt(squaredNorm);
  if (std::isfinite(norm) && norm > 1e-8F) {
    for (float& value : values) {
      value = value * weight / norm;
    }
  } else {
    std::fill(values.begin(), values.end(), 0.0F);
  }
  output.insert(output.end(), values.begin(), values.end());
}

float squaredDistance(const std::vector<float>& left,
                      const std::vector<float>& right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("Traditional knife feature dimensions differ");
  }
  float distance = 0.0F;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const float delta = left[index] - right[index];
    distance += delta * delta;
  }
  return distance;
}

bool isImageFile(const fs::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
         extension == ".bmp" || extension == ".webp";
}

std::vector<float> traditionalFeatureVector(const cv::Mat& canonicalImage,
                                            const cv::Mat& canonicalMask,
                                            float foregroundAreaRatio,
                                            float aspectRatio) {
  cv::Mat maskFloat;
  canonicalMask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);

  cv::Mat blurredMask;
  cv::GaussianBlur(maskFloat, blurredMask, cv::Size(5, 5), 0.0);
  std::vector<float> maskValues;
  maskValues.reserve(blurredMask.total());
  for (int row = 0; row < blurredMask.rows; ++row) {
    const float* values = blurredMask.ptr<float>(row);
    maskValues.insert(maskValues.end(), values, values + blurredMask.cols);
  }

  cv::Mat grayscale;
  cv::cvtColor(canonicalImage, grayscale, cv::COLOR_BGR2GRAY);
  grayscale.convertTo(grayscale, CV_32F, 1.0 / 255.0);
  cv::Scalar mean;
  cv::Scalar standardDeviation;
  cv::meanStdDev(grayscale, mean, standardDeviation, canonicalMask);
  const float denominator = static_cast<float>(standardDeviation[0]) + 0.08F;
  std::vector<float> intensityValues;
  intensityValues.reserve(grayscale.total());
  for (int row = 0; row < grayscale.rows; ++row) {
    const float* grayValues = grayscale.ptr<float>(row);
    const float* maskValuesRow = maskFloat.ptr<float>(row);
    for (int column = 0; column < grayscale.cols; ++column) {
      intensityValues.push_back(
          ((grayValues[column] - static_cast<float>(mean[0])) / denominator) *
          maskValuesRow[column]);
    }
  }

  cv::Mat hsv;
  cv::cvtColor(canonicalImage, hsv, cv::COLOR_BGR2HSV);
  const int channels[] = {0, 1};
  const int histogramSize[] = {12, 8};
  const float hueRange[] = {0.0F, 180.0F};
  const float saturationRange[] = {0.0F, 256.0F};
  const float* ranges[] = {hueRange, saturationRange};
  cv::Mat histogram;
  cv::calcHist(&hsv, 1, channels, canonicalMask, histogram, 2, histogramSize,
               ranges, true, false);
  const float histogramTotal = static_cast<float>(cv::sum(histogram)[0]);
  std::vector<float> histogramValues;
  histogramValues.reserve(histogram.total());
  for (int hue = 0; hue < histogramSize[0]; ++hue) {
    for (int saturation = 0; saturation < histogramSize[1]; ++saturation) {
      const float probability =
          histogramTotal > 0.0F
              ? histogram.at<float>(hue, saturation) / histogramTotal
              : 0.0F;
      histogramValues.push_back(std::sqrt(std::max(0.0F, probability)));
    }
  }

  cv::Mat columnSums;
  cv::reduce(maskFloat, columnSums, 0, cv::REDUCE_SUM, CV_32F);
  double maximumColumnSum = 0.0;
  cv::minMaxLoc(columnSums, nullptr, &maximumColumnSum);
  std::vector<float> shapeValues;
  shapeValues.reserve(static_cast<std::size_t>(columnSums.cols) + 9);
  for (int column = 0; column < columnSums.cols; ++column) {
    shapeValues.push_back(
        maximumColumnSum > 0.0
            ? columnSums.at<float>(0, column) /
                  static_cast<float>(maximumColumnSum)
            : 0.0F);
  }
  double huMoments[7] = {};
  cv::HuMoments(cv::moments(canonicalMask, true), huMoments);
  for (const double value : huMoments) {
    const double transformed =
        -std::copysign(1.0, value) * std::log10(std::abs(value) + 1e-12);
    shapeValues.push_back(
        static_cast<float>(std::clamp(transformed, -10.0, 10.0) / 10.0));
  }
  shapeValues.push_back(foregroundAreaRatio * 3.0F);
  shapeValues.push_back(std::log(std::max(aspectRatio, 1e-4F)) * 0.4F);

  std::vector<float> result;
  result.reserve(maskValues.size() + intensityValues.size() +
                 histogramValues.size() + shapeValues.size());
  appendNormalizedBlock(result, std::move(maskValues), 1.8F);
  appendNormalizedBlock(result, std::move(intensityValues), 0.45F);
  appendNormalizedBlock(result, std::move(histogramValues), 0.9F);
  appendNormalizedBlock(result, std::move(shapeValues), 0.8F);
  return result;
}

class TraditionalKnifeBackend final : public KnifeRecognitionBackend {
 public:
  explicit TraditionalKnifeBackend(const RuntimeConfig& config)
      : labels_(defaultKnifeLabels()),
        templateDir_(config.knifeTraditionalTemplateDir),
        threshold_(config.knifeTraditionalThreshold),
        minMargin_(config.knifeTraditionalMinMargin),
        cpuThreads_(config.knifeCpuThreads) {
    loadTemplates(templateDir_, config.knifeTraditionalMaxTemplatesPerClass);
  }

  Stage2Decision recognize(const cv::Mat& crop) const override;
  json metadata() const override;
  const std::string& mode() const override { return mode_; }
  std::size_t classCount() const override { return labels_.size(); }
  float threshold() const override { return threshold_; }
  float minMargin() const override { return minMargin_; }

 private:
  struct Features {
    std::vector<float> values;
    std::vector<float> mirroredValues;
    float foregroundAreaRatio = 0.0F;
    float aspectRatio = 0.0F;
  };

  struct Template {
    std::size_t classIndex = 0;
    std::vector<float> features;
  };

  std::optional<Features> extractFeatures(const cv::Mat& crop) const;
  void loadTemplates(const fs::path& root, int maxTemplatesPerClass);

  const std::string mode_ = "traditional";
  std::vector<std::string> labels_;
  fs::path templateDir_;
  std::vector<Template> templates_;
  std::vector<std::size_t> templateCounts_;
  std::size_t skippedTemplates_ = 0;
  float threshold_ = 0.30F;
  float minMargin_ = 0.0F;
  int cpuThreads_ = 1;
};

std::optional<TraditionalKnifeBackend::Features>
TraditionalKnifeBackend::extractFeatures(const cv::Mat& crop) const {
  if (crop.empty() || crop.type() != CV_8UC3) {
    throw std::runtime_error("Knife input must be a non-empty BGR image");
  }

  cv::Mat grayscale;
  cv::cvtColor(crop, grayscale, cv::COLOR_BGR2GRAY);
  cv::Mat blurred;
  cv::GaussianBlur(grayscale, blurred, cv::Size(5, 5), 0.0);

  std::vector<unsigned char> border;
  border.reserve(static_cast<std::size_t>(2 * crop.rows + 2 * crop.cols));
  for (int column = 0; column < blurred.cols; ++column) {
    border.push_back(blurred.at<unsigned char>(0, column));
    border.push_back(blurred.at<unsigned char>(blurred.rows - 1, column));
  }
  for (int row = 0; row < blurred.rows; ++row) {
    border.push_back(blurred.at<unsigned char>(row, 0));
    border.push_back(blurred.at<unsigned char>(row, blurred.cols - 1));
  }
  const auto median =
      border.begin() + static_cast<std::ptrdiff_t>(border.size() / 2);
  std::nth_element(border.begin(), median, border.end());
  const int backgroundLevel = *median;

  cv::Mat otsuMask;
  cv::threshold(blurred, otsuMask, 0.0, 255.0,
                cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  cv::Mat relativeDarkMask = cv::Mat::zeros(blurred.size(), CV_8U);
  if (backgroundLevel > 7) {
    cv::compare(blurred, backgroundLevel - 7, relativeDarkMask, cv::CMP_LT);
  }
  cv::Mat candidateMask;
  cv::bitwise_or(otsuMask, relativeDarkMask, candidateMask);

  const int kernelRadius = std::max(
      1, static_cast<int>(std::nearbyint(std::min(crop.rows, crop.cols) / 80.0)));
  const int kernelSize = kernelRadius * 2 + 1;
  const cv::Mat closeKernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
  cv::morphologyEx(candidateMask, candidateMask, cv::MORPH_CLOSE, closeKernel,
                   cv::Point(-1, -1), 2);
  cv::morphologyEx(candidateMask, candidateMask, cv::MORPH_OPEN,
                   cv::Mat::ones(3, 3, CV_8U));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(candidateMask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  const double cropArea = static_cast<double>(crop.rows) * crop.cols;
  int bestIndex = -1;
  double bestScore = -1.0;
  for (std::size_t index = 0; index < contours.size(); ++index) {
    const double area = cv::contourArea(contours[index]);
    if (area < cropArea * 0.006) {
      continue;
    }
    const cv::RotatedRect rectangle = cv::minAreaRect(contours[index]);
    const float shortSide = std::min(rectangle.size.width, rectangle.size.height);
    const float longSide = std::max(rectangle.size.width, rectangle.size.height);
    if (shortSide < 1.0F) {
      continue;
    }
    const float aspect = longSide / shortSide;
    const cv::Moments moments = cv::moments(contours[index]);
    if (moments.m00 <= 0.0) {
      continue;
    }
    const double centerX = moments.m10 / moments.m00;
    const double centerY = moments.m01 / moments.m00;
    const double centerDistance = std::hypot(
        (centerX - crop.cols * 0.5) / crop.cols,
        (centerY - crop.rows * 0.5) / crop.rows);
    const double score = area * (1.0 + std::min(aspect, 6.0F) * 0.12) *
                         std::exp(-centerDistance * 1.2);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = static_cast<int>(index);
    }
  }
  if (bestIndex < 0) {
    return std::nullopt;
  }

  cv::Mat foregroundMask = cv::Mat::zeros(candidateMask.size(), CV_8U);
  cv::drawContours(foregroundMask, contours, bestIndex, cv::Scalar(255),
                   cv::FILLED);
  const int nearbyKernelSize = std::max(3, kernelSize * 2 + 1);
  cv::Mat nearby;
  cv::dilate(foregroundMask, nearby,
             cv::Mat::ones(nearbyKernelSize, nearbyKernelSize, CV_8U));
  cv::bitwise_and(candidateMask, nearby, nearby);
  cv::bitwise_or(foregroundMask, nearby, foregroundMask);

  std::vector<std::vector<cv::Point>> foregroundContours;
  cv::findContours(foregroundMask, foregroundContours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  if (foregroundContours.empty()) {
    return std::nullopt;
  }
  const auto largest = std::max_element(
      foregroundContours.begin(), foregroundContours.end(),
      [](const auto& left, const auto& right) {
        return cv::contourArea(left) < cv::contourArea(right);
      });
  const cv::RotatedRect rectangle = cv::minAreaRect(*largest);
  const float shortSide = std::min(rectangle.size.width, rectangle.size.height);
  const float longSide = std::max(rectangle.size.width, rectangle.size.height);
  if (shortSide < 1.0F || longSide < 2.0F) {
    return std::nullopt;
  }
  const float aspectRatio = longSide / shortSide;
  const float foregroundAreaRatio = static_cast<float>(
      cv::contourArea(*largest) / std::max(1.0, cropArea));

  float angle = rectangle.angle;
  if (rectangle.size.width < rectangle.size.height) {
    angle += 90.0F;
  }
  const cv::Mat rotation =
      cv::getRotationMatrix2D(rectangle.center, angle, 1.0);
  cv::Mat rotatedImage;
  cv::Mat rotatedMask;
  cv::warpAffine(crop, rotatedImage, rotation, crop.size(), cv::INTER_LINEAR,
                 cv::BORDER_REPLICATE);
  cv::warpAffine(foregroundMask, rotatedMask, rotation, crop.size(),
                 cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
  std::vector<cv::Point> foregroundPoints;
  cv::findNonZero(rotatedMask, foregroundPoints);
  if (foregroundPoints.empty()) {
    return std::nullopt;
  }
  const cv::Rect bounds = cv::boundingRect(foregroundPoints);
  cv::Mat objectImage = rotatedImage(bounds).clone();
  cv::Mat objectMask = rotatedMask(bounds).clone();
  if (objectImage.rows > objectImage.cols) {
    cv::rotate(objectImage, objectImage, cv::ROTATE_90_CLOCKWISE);
    cv::rotate(objectMask, objectMask, cv::ROTATE_90_CLOCKWISE);
  }

  constexpr int outputWidth = 64;
  constexpr int outputHeight = 24;
  const double scale = std::min(
      static_cast<double>(outputWidth - 4) / objectImage.cols,
      static_cast<double>(outputHeight - 4) / objectImage.rows);
  const int resizedWidth =
      std::max(1, static_cast<int>(std::nearbyint(objectImage.cols * scale)));
  const int resizedHeight =
      std::max(1, static_cast<int>(std::nearbyint(objectImage.rows * scale)));
  cv::resize(objectImage, objectImage, cv::Size(resizedWidth, resizedHeight),
             0.0, 0.0, scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
  cv::resize(objectMask, objectMask, cv::Size(resizedWidth, resizedHeight), 0.0,
             0.0, cv::INTER_NEAREST);
  cv::Mat canonicalImage = cv::Mat::zeros(outputHeight, outputWidth, CV_8UC3);
  cv::Mat canonicalMask = cv::Mat::zeros(outputHeight, outputWidth, CV_8U);
  const int offsetX = (outputWidth - resizedWidth) / 2;
  const int offsetY = (outputHeight - resizedHeight) / 2;
  objectImage.copyTo(canonicalImage(
      cv::Rect(offsetX, offsetY, resizedWidth, resizedHeight)));
  objectMask.copyTo(
      canonicalMask(cv::Rect(offsetX, offsetY, resizedWidth, resizedHeight)));

  Features result;
  result.values = traditionalFeatureVector(
      canonicalImage, canonicalMask, foregroundAreaRatio, aspectRatio);
  cv::Mat mirroredImage;
  cv::Mat mirroredMask;
  cv::flip(canonicalImage, mirroredImage, 1);
  cv::flip(canonicalMask, mirroredMask, 1);
  result.mirroredValues = traditionalFeatureVector(
      mirroredImage, mirroredMask, foregroundAreaRatio, aspectRatio);
  result.foregroundAreaRatio = foregroundAreaRatio;
  result.aspectRatio = aspectRatio;
  return result;
}

void TraditionalKnifeBackend::loadTemplates(const fs::path& root,
                                            int maxTemplatesPerClass) {
  templateCounts_.assign(labels_.size(), 0);
  for (std::size_t classIndex = 0; classIndex < labels_.size(); ++classIndex) {
    const fs::path classDirectory = root / labels_[classIndex];
    if (!fs::is_directory(classDirectory)) {
      throw std::runtime_error("Traditional knife template class not found: " +
                               classDirectory.string());
    }
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(classDirectory)) {
      if (entry.is_regular_file() && isImageFile(entry.path())) {
        paths.push_back(entry.path());
      }
    }
    std::sort(paths.begin(), paths.end());
    std::vector<fs::path> selected;
    if (maxTemplatesPerClass > 0 &&
        paths.size() > static_cast<std::size_t>(maxTemplatesPerClass)) {
      selected.reserve(static_cast<std::size_t>(maxTemplatesPerClass));
      for (int index = 0; index < maxTemplatesPerClass; ++index) {
        const std::size_t sourceIndex = std::min(
            paths.size() - 1,
            static_cast<std::size_t>((index + 0.5) * paths.size() /
                                     maxTemplatesPerClass));
        selected.push_back(paths[sourceIndex]);
      }
    } else {
      selected = std::move(paths);
    }

    for (const fs::path& path : selected) {
      const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
      if (image.empty()) {
        ++skippedTemplates_;
        continue;
      }
      const auto features = extractFeatures(image);
      if (!features) {
        ++skippedTemplates_;
        continue;
      }
      templates_.push_back({classIndex, features->values});
      ++templateCounts_[classIndex];
    }
    if (templateCounts_[classIndex] == 0) {
      throw std::runtime_error("No usable traditional knife templates for " +
                               labels_[classIndex]);
    }
  }
}

Stage2Decision TraditionalKnifeBackend::recognize(const cv::Mat& crop) const {
  Stage2Decision decision;
  decision.module = "knife";
  decision.scoreKind = "traditional_template_similarity";
  const auto features = extractFeatures(crop);
  if (!features) {
    decision.status = "unknown";
    decision.metadata = {{"reason", "no_usable_foreground_contour"}};
    return decision;
  }

  std::vector<std::vector<float>> distances(labels_.size());
  for (const auto& item : templates_) {
    const float direct = squaredDistance(features->values, item.features);
    const float mirrored =
        squaredDistance(features->mirroredValues, item.features);
    distances[item.classIndex].push_back(std::min(direct, mirrored));
  }

  constexpr std::size_t neighborCount = 5;
  std::vector<float> classDistances(labels_.size(),
                                    std::numeric_limits<float>::infinity());
  for (std::size_t index = 0; index < distances.size(); ++index) {
    auto& values = distances[index];
    const std::size_t count = std::min(neighborCount, values.size());
    if (count == 0) {
      continue;
    }
    if (values.size() > count) {
      std::nth_element(values.begin(), values.begin() + count, values.end());
    }
    classDistances[index] =
        std::accumulate(values.begin(), values.begin() + count, 0.0F) / count;
  }

  std::vector<std::size_t> ranked(labels_.size());
  std::iota(ranked.begin(), ranked.end(), 0);
  std::sort(ranked.begin(), ranked.end(), [&](std::size_t left,
                                               std::size_t right) {
    return classDistances[left] < classDistances[right];
  });
  const std::size_t best = ranked.front();
  const std::size_t second = ranked.size() > 1 ? ranked[1] : ranked.front();
  const float bestScore = std::exp(-0.5F * classDistances[best]);
  const float secondScore = std::exp(-0.5F * classDistances[second]);
  for (std::size_t index = 0; index < labels_.size(); ++index) {
    decision.classScores[labels_[index]] =
        std::exp(-0.5F * classDistances[index]);
  }
  decision.candidateClass = labels_[best];
  decision.score = bestScore;
  decision.secondBestClass = labels_[second];
  decision.secondBestScore = secondScore;
  decision.margin = bestScore - secondScore;
  if (bestScore < threshold_) {
    decision.status = "unknown";
  } else if (*decision.margin < minMargin_) {
    decision.status = "ambiguous";
  } else {
    decision.status = "matched";
    decision.predictedClass = labels_[best];
  }
  decision.metadata = {
      {"foreground_area_ratio", features->foregroundAreaRatio},
      {"foreground_aspect_ratio", features->aspectRatio},
      {"nearest_class_distance", classDistances[best]},
      {"templates_compared", templates_.size()},
      {"neighbors_per_class", neighborCount},
  };
  return decision;
}

json TraditionalKnifeBackend::metadata() const {
  json templateCounts = json::object();
  for (std::size_t index = 0; index < labels_.size(); ++index) {
    templateCounts[labels_[index]] = templateCounts_[index];
  }
  return {
      {"backend", "opencv_traditional_shape_cpp"},
      {"mode", mode_},
      {"labels", labels_},
      {"device", "cpu"},
      {"template_dir", templateDir_.string()},
      {"template_count", templates_.size()},
      {"template_counts", templateCounts},
      {"skipped_templates", skippedTemplates_},
      {"feature_dimension",
       templates_.empty() ? 0 : templates_.front().features.size()},
      {"threshold", threshold_},
      {"min_margin", minMargin_},
      {"score_kind", "traditional_template_similarity"},
      {"segmentation", "adaptive_dark_foreground_contour"},
      {"matching", "flip_invariant_top5_nearest_templates"},
      {"cpu_threads", cpuThreads_},
  };
}

}  // namespace

std::unique_ptr<KnifeRecognitionBackend> createTraditionalKnifeBackend(
    const RuntimeConfig& config) {
  return std::make_unique<TraditionalKnifeBackend>(config);
}

}  // namespace insight_cup
