#include "insight_cup/types.hpp"

namespace insight_cup {

json Stage1Detection::handoffJson(
    const std::optional<std::string>& cropPath) const {
  return {
      {"id", detectionId},
      {"major_class", majorClass},
      {"confidence", confidence},
      {"bbox_xyxy", bbox.toJson()},
      {"crop_path", cropPath ? json(*cropPath) : json(nullptr)},
  };
}

json YoloDebugInfo::toJson() const {
  return {
      {"raw_yolo_class_id", rawClassId},
      {"raw_yolo_class", rawClassName},
  };
}

json Stage2Decision::toJson() const {
  json payload = {
      {"status", status},
      {"predicted_class",
       predictedClass ? json(*predictedClass) : json(nullptr)},
      {"candidate_class",
       candidateClass ? json(*candidateClass) : json(nullptr)},
      {"score", score ? json(*score) : json(nullptr)},
      {"score_kind", scoreKind ? json(*scoreKind) : json(nullptr)},
      {"second_best_class",
       secondBestClass ? json(*secondBestClass) : json(nullptr)},
      {"second_best_score",
       secondBestScore ? json(*secondBestScore) : json(nullptr)},
      {"margin", margin ? json(*margin) : json(nullptr)},
  };
  if (metadata.is_object()) {
    payload.update(metadata);
  }
  return payload;
}

}  // namespace insight_cup
