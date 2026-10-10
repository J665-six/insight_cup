#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <opencv2/core.hpp>

#include "insight_cup/safe_obstacle.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

insight_cup::DetectionFrame detectionAt(float x, float y) {
  insight_cup::DetectionFrame frame;
  insight_cup::DetectedRegion region;
  region.handoff.majorClass = "b0";
  region.handoff.confidence = 0.9F;
  region.handoff.bbox = {static_cast<int>(x - 5.0F), static_cast<int>(y - 5.0F),
                         static_cast<int>(x + 5.0F), static_cast<int>(y + 5.0F)};
  frame.regions.push_back(std::move(region));
  return frame;
}

void writeConfig(const std::string& path) {
  std::ofstream stream(path);
  stream << R"JSON({
    "schema": "insight.safe_zones.v1",
    "obstacle_roi": [[0,0],[100,0],[100,100],[0,100]],
    "distance_zones": [
      {"id":2,"distance_cm":20,"polygon":[[45,75],[55,75],[55,85],[45,85]]},
      {"id":3,"distance_cm":40,"polygon":[[45,55],[55,55],[55,65],[45,65]]}
    ]
  })JSON";
}

void testImmediateObstacleAndDistance() {
  const std::string configPath = "/tmp/insight_safe_test.json";
  writeConfig(configPath);
  insight_cup::RuntimeConfig config;
  config.safeZoneConfig = configPath;
  config.safeStableSeconds = 2.0F;
  config.safeConfidence = 0.25F;
  insight_cup::Logger logger;
  insight_cup::SafeObstacleMonitor monitor(config, logger);

  auto immediate = monitor.update(detectionAt(50.0F, 80.0F), {100, 100}, 0,
                                  0.0);
  require(immediate.obstacle && immediate.status == "obstacle",
          "obstacle must be reported immediately");
  require(immediate.distanceCm == 20 &&
              std::abs(immediate.distanceM - 0.2) < 1e-6,
          "nearest calibration center selects 20 cm");

  auto clear = monitor.update({}, {100, 100}, 31, 2066.0);
  require(!clear.obstacle && clear.status == "clear",
          "missing obstacle clears the safety state");
}

void testDistanceLogUsesTaggedObstacleRoi() {
  const std::string configPath = "/tmp/insight_safe_distance_log.json";
  std::ofstream stream(configPath);
  stream << R"JSON({
    "schema": "insight.distance_boxes.v1",
    "boxes": [
      {"id":1,"area_px2":100,"polygon":[[10,10],[20,10],[20,20],[10,20]]},
      {"id":2,"area_px2":90,"polygon":[[30,10],[40,10],[40,20],[30,20]]},
      {"id":3,"area_px2":80,"polygon":[[50,10],[60,10],[60,20],[50,20]]},
      {"id":4,"area_px2":70,"polygon":[[70,10],[80,10],[80,20],[70,20]]},
      {"id":5,"area_px2":60,"polygon":[[90,10],[100,10],[100,20],[90,20]]},
      {"id":6,"area_px2":50,"polygon":[[110,10],[120,10],[120,20],[110,20]]},
      {"id":7,"area_px2":999999,"polygon":[[0,0],[500,0],[500,500],[0,500]]}
    ]
  })JSON";
  stream.close();

  insight_cup::RuntimeConfig config;
  config.safeZoneConfig = configPath;
  insight_cup::Logger logger;
  insight_cup::SafeObstacleMonitor monitor(config, logger);
  const auto metadata = monitor.metadata();
  require(metadata.value("configured", false),
          "complete tagged distance log must configure safe monitor");
  require(metadata.at("obstacle_roi").at(0).at(0).get<float>() == 10.0F,
          "id 1 must be used as obstacle ROI even when id 7 is larger");
  require(metadata.at("distance_zones").size() == 5,
          "id 7 person ROI must not become a distance zone");
  for (const auto& zone : metadata.at("distance_zones")) {
    require(zone.at("source_id").get<int>() != 7,
            "person ROI id 7 must be excluded from safe distance zones");
  }
}

void testDistanceLogRejectsUnTaggedObstacleRoi() {
  const std::string configPath = "/tmp/insight_safe_missing_roi.json";
  std::ofstream stream(configPath);
  stream << R"JSON({
    "schema": "insight.distance_boxes.v1",
    "boxes": [
      {"id":2,"area_px2":900,"polygon":[[10,10],[40,10],[40,40],[10,40]]},
      {"id":3,"area_px2":800,"polygon":[[50,10],[75,10],[75,35],[50,35]]},
      {"id":4,"area_px2":700,"polygon":[[80,10],[100,10],[100,30],[80,30]]},
      {"id":5,"area_px2":600,"polygon":[[105,10],[120,10],[120,25],[105,25]]},
      {"id":6,"area_px2":500,"polygon":[[125,10],[140,10],[140,20],[125,20]]},
      {"id":7,"area_px2":999999,"polygon":[[0,0],[500,0],[500,500],[0,500]]}
    ]
  })JSON";
  stream.close();

  insight_cup::RuntimeConfig config;
  config.safeZoneConfig = configPath;
  insight_cup::Logger logger;
  insight_cup::SafeObstacleMonitor monitor(config, logger);
  const auto metadata = monitor.metadata();
  require(!metadata.value("configured", true),
          "distance log without id 1 must remain unconfigured");
  require(metadata.at("obstacle_roi").empty(),
          "largest polygon must not become obstacle ROI");
}

void testSingleDistanceLogMapsCentersToDistances() {
  const std::string configPath = "/tmp/insight_safe_single_distance_log.json";
  std::ofstream stream(configPath);
  stream << R"JSON({
    "schema": "insight.distance_boxes.v1",
    "boxes": [
      {"id":1,"polygon":[[0,0],[200,0],[200,200],[0,200]]},
      {"id":2,"polygon":[[90,170],[110,170],[110,190],[90,190]]},
      {"id":3,"polygon":[[90,130],[110,130],[110,150],[90,150]]},
      {"id":4,"polygon":[[90,90],[110,90],[110,110],[90,110]]},
      {"id":5,"polygon":[[90,50],[110,50],[110,70],[90,70]]},
      {"id":6,"polygon":[[90,10],[110,10],[110,30],[90,30]]}
    ]
  })JSON";
  stream.close();

  insight_cup::RuntimeConfig config;
  config.safeZoneConfig = configPath;
  config.safeStableSeconds = 0.1F;
  config.safeConfidence = 0.1F;
  insight_cup::Logger logger;
  insight_cup::SafeObstacleMonitor monitor(config, logger);
  require(monitor.metadata().value("configured", false),
          "one distance log with id 1..6 must configure safe monitor");
  auto state = monitor.update(detectionAt(100.0F, 90.0F), {200, 200}, 0,
                              0.0);
  state = monitor.update(detectionAt(100.0F, 90.0F), {200, 200}, 3, 200.0);
  require(state.obstacle && state.distanceCm == 60,
          "obstacle center must map to the nearest #4A 60 cm region");
}

}  // namespace

int main() {
  try {
    testImmediateObstacleAndDistance();
    testDistanceLogUsesTaggedObstacleRoi();
    testDistanceLogRejectsUnTaggedObstacleRoi();
    testSingleDistanceLogMapsCentersToDistances();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
