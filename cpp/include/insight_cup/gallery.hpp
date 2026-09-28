#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace insight_cup {

namespace fs = std::filesystem;

struct GalleryMatch {
  std::string status;
  std::string candidateClass;
  std::string predictedClass;
  float score = 0.0F;
  std::string secondBestClass;
  float secondBestScore = 0.0F;
  float margin = 0.0F;
  std::map<std::string, float> classScores;
};

class FeatureGallery {
 public:
  explicit FeatureGallery(const fs::path& path);

  std::size_t rows() const { return labels_.size(); }
  std::size_t dimensions() const { return dimensions_; }
  const std::vector<std::string>& labels() const { return labels_; }
  const std::vector<std::string>& identities() const { return identities_; }

  GalleryMatch matchByIdentityMax(const std::vector<float>& embedding,
                                  float threshold,
                                  float minMargin) const;

 private:
  std::size_t dimensions_ = 0;
  std::vector<std::string> labels_;
  std::vector<std::string> identities_;
  std::vector<float> embeddings_;
};

}  // namespace insight_cup
