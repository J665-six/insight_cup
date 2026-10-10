#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include <librealsense2/rs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "insight_cup/stereo_range.hpp"

namespace {

struct Options {
  std::optional<std::string> serial;
  std::optional<cv::Rect> box;
  int width = 1280;
  int height = 720;
  int fps = 30;
  bool noGui = false;
};

volatile std::sig_atomic_t stopped = 0;
void handleSignal(int) { stopped = 1; }

int integerValue(const std::string &text, const std::string &name) {
  int value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
    throw std::runtime_error("Invalid " + name + ": " + text);
  }
  return value;
}

cv::Rect parseBox(const std::string &text) {
  std::istringstream stream(text);
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;
  char separator1 = 0;
  char separator2 = 0;
  char separator3 = 0;
  if (!(stream >> x1 >> separator1 >> y1 >> separator2 >> x2 >> separator3 >>
        y2) ||
      separator1 != ',' || separator2 != ',' || separator3 != ',' ||
      stream.peek() != std::char_traits<char>::eof() || x2 <= x1 || y2 <= y1) {
    throw std::runtime_error("--bbox must be x1,y1,x2,y2 with positive area");
  }
  return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

void printHelp() {
  std::cout
      << "D455 aligned RGB-D stereo range test\n\n"
      << "Usage: insight-cup-stereo-range-test [options]\n\n"
      << "  --serial SERIAL       select a RealSense device by serial number\n"
      << "  --bbox X1,Y1,X2,Y2   fixed detection box in color image pixels\n"
      << "  --width PIXELS       color width (default 1280)\n"
      << "  --height PIXELS      color height (default 720)\n"
      << "  --fps FPS            stream frame rate (default 30)\n"
      << "  --no-gui             print JSON lines; requires --bbox\n"
      << "  --help               show this help\n\n"
      << "Interactive: drag a box over an obstacle; press c to clear or q to "
         "quit.\n"
      << "Camera coordinates: x right, y down, z forward; range is 3D "
         "distance.\n";
}

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto next = [&]() -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for " + option);
      }
      return argv[++index];
    };
    if (option == "--help" || option == "-h") {
      printHelp();
      std::exit(0);
    }
    if (option == "--serial") {
      options.serial = next();
    } else if (option == "--bbox") {
      options.box = parseBox(next());
    } else if (option == "--width") {
      options.width = integerValue(next(), "width");
    } else if (option == "--height") {
      options.height = integerValue(next(), "height");
    } else if (option == "--fps") {
      options.fps = integerValue(next(), "fps");
    } else if (option == "--no-gui") {
      options.noGui = true;
    } else {
      throw std::runtime_error("Unknown option: " + option);
    }
  }
  if (options.width <= 0 || options.height <= 0 || options.fps <= 0) {
    throw std::runtime_error("Camera width, height, and fps must be positive");
  }
  if (options.noGui && !options.box) {
    throw std::runtime_error("--no-gui requires --bbox");
  }
  return options;
}

struct Selection {
  cv::Rect box;
  cv::Point anchor;
  bool dragging = false;
  bool selected = false;
};

void onMouse(int event, int x, int y, int, void *userdata) {
  auto &selection = *static_cast<Selection *>(userdata);
  if (event == cv::EVENT_LBUTTONDOWN) {
    selection.anchor = cv::Point(x, y);
    selection.box = cv::Rect(x, y, 0, 0);
    selection.dragging = true;
    selection.selected = false;
  } else if (event == cv::EVENT_MOUSEMOVE && selection.dragging) {
    const cv::Point current(x, y);
    selection.box = cv::Rect(std::min(selection.anchor.x, current.x),
                             std::min(selection.anchor.y, current.y),
                             std::abs(current.x - selection.anchor.x),
                             std::abs(current.y - selection.anchor.y));
  } else if (event == cv::EVENT_LBUTTONUP && selection.dragging) {
    const cv::Point current(x, y);
    selection.box = cv::Rect(std::min(selection.anchor.x, current.x),
                             std::min(selection.anchor.y, current.y),
                             std::abs(current.x - selection.anchor.x),
                             std::abs(current.y - selection.anchor.y));
    selection.dragging = false;
    selection.selected = selection.box.width >= 3 && selection.box.height >= 3;
  }
}

std::string measurementJson(const cv::Rect &box,
                            const insight_cup::StereoRangeMeasurement &result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << "{\"bbox\":[" << box.x << ','
      << box.y << ',' << box.x + box.width << ',' << box.y + box.height << "]"
      << ",\"valid\":" << (result.valid ? "true" : "false") << ",\"reason\":\""
      << result.reason << '\"' << ",\"camera_x_m\":";
  if (result.valid) {
    out << result.xMeters << ",\"camera_y_m\":" << result.yMeters
        << ",\"depth_z_m\":" << result.zMeters
        << ",\"range_m\":" << result.rangeMeters;
  } else {
    out << "null,\"camera_y_m\":null,\"depth_z_m\":null,\"range_m\":null";
  }
  out << ",\"valid_samples\":" << result.validSamples
      << ",\"total_samples\":" << result.totalSamples
      << ",\"valid_ratio\":" << result.validRatio << '}';
  return out.str();
}

bool reportIntervalElapsed(std::chrono::steady_clock::time_point lastOutput,
                           std::chrono::steady_clock::time_point now) {
  return lastOutput == std::chrono::steady_clock::time_point::min() ||
         now - lastOutput >= std::chrono::milliseconds(200);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    std::signal(SIGINT, handleSignal);

    rs2::context context;
    const rs2::device_list devices = context.query_devices();
    if (devices.size() == 0) {
      throw std::runtime_error("No RealSense camera found");
    }
    if (options.serial) {
      bool found = false;
      for (auto &&device : devices) {
        if (device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) == *options.serial) {
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::runtime_error("RealSense serial not found: " +
                                 *options.serial);
      }
    }

    rs2::pipeline pipeline(context);
    rs2::config cameraConfig;
    if (options.serial) {
      cameraConfig.enable_device(*options.serial);
    }
    cameraConfig.enable_stream(RS2_STREAM_COLOR, options.width, options.height,
                               RS2_FORMAT_BGR8, options.fps);
    cameraConfig.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16,
                               options.fps);
    const rs2::pipeline_profile profile = pipeline.start(cameraConfig);
    const rs2::device device = profile.get_device();
    std::cerr << "Camera: " << device.get_info(RS2_CAMERA_INFO_NAME)
              << " serial=" << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
              << '\n';

    rs2::align alignDepthToColor(RS2_STREAM_COLOR);
    constexpr char windowName[] = "D455 stereo range test";
    Selection selection;
    if (options.box) {
      selection.box = *options.box;
      selection.selected = true;
    }
    if (!options.noGui) {
      cv::namedWindow(windowName, cv::WINDOW_NORMAL);
      cv::setMouseCallback(windowName, onMouse, &selection);
    }

    auto lastOutput = std::chrono::steady_clock::time_point::min();
    while (!stopped) {
      const rs2::frameset aligned =
          alignDepthToColor.process(pipeline.wait_for_frames(5000));
      const rs2::video_frame color = aligned.get_color_frame();
      const rs2::depth_frame depth = aligned.get_depth_frame();
      if (!color || !depth) {
        continue;
      }
      cv::Mat colorFrame(color.get_height(), color.get_width(), CV_8UC3,
                         const_cast<void *>(color.get_data()),
                         color.get_stride_in_bytes());
      cv::Mat depthFrame(depth.get_height(), depth.get_width(), CV_16UC1,
                         const_cast<void *>(depth.get_data()),
                         depth.get_stride_in_bytes());
      const rs2_intrinsics intrinsics =
          depth.get_profile().as<rs2::video_stream_profile>().get_intrinsics();

      cv::Mat display = colorFrame.clone();
      if (selection.selected || selection.dragging) {
        const cv::Rect box =
            selection.box & cv::Rect(0, 0, colorFrame.cols, colorFrame.rows);
        if (box.area() > 0) {
          cv::rectangle(display, box, cv::Scalar(0, 255, 0), 2);
          const auto result = insight_cup::measureStereoRange(
              depthFrame, box, depth.get_units(), intrinsics);
          if (selection.selected) {
            const auto now = std::chrono::steady_clock::now();
            if (reportIntervalElapsed(lastOutput, now)) {
              std::cout << measurementJson(box, result) << std::endl;
              lastOutput = now;
            }
            if (result.valid) {
              std::ostringstream label;
              label << std::fixed << std::setprecision(2) << result.rangeMeters
                    << " m (Z " << result.zMeters << " m)";
              cv::putText(display, label.str(),
                          cv::Point(box.x, std::max(24, box.y - 8)),
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0),
                          2);
            } else {
              cv::putText(display, result.reason,
                          cv::Point(box.x, std::max(24, box.y - 8)),
                          cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255),
                          2);
            }
          }
        }
      }
      if (!options.noGui) {
        cv::putText(display, "Drag obstacle box | c: clear | q: quit",
                    cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2);
        cv::imshow(windowName, display);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 'q' || key == 27) {
          break;
        }
        if (key == 'c') {
          selection.selected = false;
          selection.dragging = false;
        }
      } else if (selection.selected) {
        const auto now = std::chrono::steady_clock::now();
        if (reportIntervalElapsed(lastOutput, now)) {
          const cv::Rect box =
              selection.box & cv::Rect(0, 0, colorFrame.cols, colorFrame.rows);
          const auto result = insight_cup::measureStereoRange(
              depthFrame, box, depth.get_units(), intrinsics);
          std::cout << measurementJson(box, result) << std::endl;
          lastOutput = now;
        }
      }
    }
    pipeline.stop();
    if (!options.noGui) {
      cv::destroyWindow(windowName);
    }
    return 0;
  } catch (const rs2::error &error) {
    std::cerr << "RealSense error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
