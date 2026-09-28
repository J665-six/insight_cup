#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "insight_cup/config.hpp"
#include "insight_cup/gallery.hpp"
#include "insight_cup/logging.hpp"
#include "insight_cup/onnx.hpp"
#include "insight_cup/types.hpp"

namespace insight_cup {

class YoloDetector {
 public:
  explicit YoloDetector(const RuntimeConfig& config);

  DetectionFrame detect(const cv::Mat& frame, int frameIndex) const;
  json metadata() const;

 private:
  struct RawDetection {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float confidence = 0.0F;
    int classId = -1;
  };

  std::vector<RawDetection> predict(const cv::Mat& frame) const;

  fs::path weights_;
  float confidence_ = 0.25F;
  float iou_ = 0.45F;
  int imageSize_ = 640;
  int cpuThreads_ = 1;
  std::unique_ptr<OnnxSession> session_;
  std::vector<std::string> names_;
};

class FaceRecognizer {
 public:
  explicit FaceRecognizer(const RuntimeConfig& config);

  Stage2Decision recognize(const cv::Mat& crop) const;
  json metadata() const;
  std::size_t identityCount() const { return gallery_.identities().size(); }
  std::size_t referenceCount() const { return gallery_.rows(); }

 private:
  struct FaceDetection {
    FloatBox bbox;
    float score = 0.0F;
    std::array<cv::Point2f, 5> keypoints{};
  };

  struct FaceAttempt {
    Stage2Decision decision;
    int rank = -1;
    float score = -1.0F;
    float detectionScore = -1.0F;
  };

  std::vector<FaceDetection> detectFaces(const cv::Mat& image) const;
  std::vector<float> embed(const cv::Mat& image,
                           const std::array<cv::Point2f, 5>& keypoints) const;
  FaceAttempt recognizeOnce(const cv::Mat& crop) const;

  std::unique_ptr<OnnxSession> detector_;
  std::unique_ptr<OnnxSession> recognizer_;
  FeatureGallery gallery_;
  fs::path modelDirectory_;
  fs::path galleryPath_;
  float threshold_ = 0.40F;
  float minMargin_ = 0.03F;
  float detectionThreshold_ = 0.18F;
  int detectionSize_ = 320;
  int upsampleMinSide_ = 320;
  float rotationRetryDegrees_ = 0.0F;
  int cpuThreads_ = 1;
};

class KnifeRecognizer {
 public:
  explicit KnifeRecognizer(const RuntimeConfig& config);

  Stage2Decision recognize(const cv::Mat& crop) const;
  json metadata() const;
  const std::string& mode() const { return mode_; }
  std::size_t classCount() const { return labels_.size(); }
  float threshold() const { return threshold_; }
  float minMargin() const { return minMargin_; }

 private:
  std::vector<float> prepare(const cv::Mat& crop) const;

  std::string mode_;
  fs::path modelPath_;
  fs::path galleryPath_;
  std::unique_ptr<OnnxSession> session_;
  std::unique_ptr<FeatureGallery> gallery_;
  std::vector<std::string> labels_;
  float threshold_ = 0.50F;
  float minMargin_ = 0.15F;
  int cpuThreads_ = 1;
};

class RecognitionRouter {
 public:
  RecognitionRouter(std::shared_ptr<FaceRecognizer> face,
                    std::shared_ptr<KnifeRecognizer> knife,
                    Logger& logger);

  Stage2Decision route(const Stage1Detection& detection) const;

 private:
  std::shared_ptr<FaceRecognizer> face_;
  std::shared_ptr<KnifeRecognizer> knife_;
  Logger& logger_;
};

cv::Mat squareContextCrop(const cv::Mat& image, const Box& bbox,
                          float contextScale = 1.15F, int fillValue = 127);
float bboxIou(const Box& left, const Box& right);

}  // namespace insight_cup
