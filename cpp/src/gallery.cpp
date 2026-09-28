#include "insight_cup/gallery.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace insight_cup {

namespace {

template <typename Value>
Value readLittleEndian(std::istream& input) {
  std::array<unsigned char, sizeof(Value)> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) {
    throw std::runtime_error("Unexpected end of gallery file");
  }
  Value value{};
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  std::memcpy(&value, bytes.data(), sizeof(Value));
#else
  std::reverse(bytes.begin(), bytes.end());
  std::memcpy(&value, bytes.data(), sizeof(Value));
#endif
  return value;
}

std::pair<std::string, int> naturalKey(const std::string& value) {
  std::size_t index = value.size();
  while (index > 0 && std::isdigit(static_cast<unsigned char>(value[index - 1]))) {
    --index;
  }
  const std::string prefix = value.substr(0, index);
  const int number = index < value.size() ? std::stoi(value.substr(index)) : -1;
  return {prefix, number};
}

}  // namespace

FeatureGallery::FeatureGallery(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Gallery not found: " + path.string());
  }
  char magic[4]{};
  input.read(magic, sizeof(magic));
  if (!input || std::string(magic, sizeof(magic)) != "ICG1") {
    throw std::runtime_error("Unsupported gallery format: " + path.string());
  }
  const std::uint32_t rows = readLittleEndian<std::uint32_t>(input);
  const std::uint32_t columns = readLittleEndian<std::uint32_t>(input);
  if (rows == 0 || columns == 0 || rows > 1000000 || columns > 100000) {
    throw std::runtime_error("Invalid gallery dimensions: " + path.string());
  }
  dimensions_ = columns;
  labels_.reserve(rows);
  embeddings_.resize(static_cast<std::size_t>(rows) * columns);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::uint16_t labelLength = readLittleEndian<std::uint16_t>(input);
    if (labelLength == 0 || labelLength > 1024) {
      throw std::runtime_error("Invalid gallery label length");
    }
    std::string label(labelLength, '\0');
    input.read(label.data(), label.size());
    if (!input) {
      throw std::runtime_error("Unexpected end of gallery label data");
    }
    labels_.push_back(std::move(label));
    float squaredNorm = 0.0F;
    float* destination = embeddings_.data() + static_cast<std::size_t>(row) * columns;
    for (std::uint32_t column = 0; column < columns; ++column) {
      destination[column] = readLittleEndian<float>(input);
      squaredNorm += destination[column] * destination[column];
    }
    const float norm = std::sqrt(squaredNorm);
    if (!std::isfinite(norm) || norm <= 0.0F) {
      throw std::runtime_error("Gallery contains an invalid embedding");
    }
    for (std::uint32_t column = 0; column < columns; ++column) {
      destination[column] /= norm;
    }
  }
  std::set<std::string> unique(labels_.begin(), labels_.end());
  identities_.assign(unique.begin(), unique.end());
  std::sort(identities_.begin(), identities_.end(),
            [](const std::string& left, const std::string& right) {
              return naturalKey(left) < naturalKey(right);
            });
}

GalleryMatch FeatureGallery::matchByIdentityMax(
    const std::vector<float>& embedding, float threshold, float minMargin) const {
  if (embedding.size() != dimensions_) {
    throw std::runtime_error("Query embedding dimension does not match gallery");
  }
  float squaredNorm = 0.0F;
  for (float value : embedding) {
    squaredNorm += value * value;
  }
  const float norm = std::sqrt(squaredNorm);
  if (!std::isfinite(norm) || norm <= 0.0F) {
    throw std::runtime_error("Query embedding is invalid");
  }
  std::vector<float> normalizedEmbedding = embedding;
  for (float& value : normalizedEmbedding) {
    value /= norm;
  }

  std::map<std::string, float> identityScores;
  for (std::size_t row = 0; row < labels_.size(); ++row) {
    const float* reference = embeddings_.data() + row * dimensions_;
    float score = 0.0F;
    for (std::size_t column = 0; column < dimensions_; ++column) {
      score += reference[column] * normalizedEmbedding[column];
    }
    auto found = identityScores.find(labels_[row]);
    if (found == identityScores.end() || score > found->second) {
      identityScores[labels_[row]] = score;
    }
  }
  std::vector<std::pair<std::string, float>> ranked(identityScores.begin(),
                                                     identityScores.end());
  std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
    return left.second > right.second;
  });
  if (ranked.empty()) {
    throw std::runtime_error("Gallery has no identities");
  }

  GalleryMatch result;
  result.candidateClass = ranked[0].first;
  result.score = ranked[0].second;
  result.classScores = std::move(identityScores);
  if (ranked.size() > 1) {
    result.secondBestClass = ranked[1].first;
    result.secondBestScore = ranked[1].second;
    result.margin = result.score - result.secondBestScore;
  }
  if (result.score < threshold) {
    result.status = "unknown";
  } else if (ranked.size() > 1 && result.margin < minMargin) {
    result.status = "ambiguous";
  } else {
    result.status = "matched";
    result.predictedClass = result.candidateClass;
  }
  return result;
}

}  // namespace insight_cup
