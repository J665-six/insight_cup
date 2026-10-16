cmake_minimum_required(VERSION 3.16)

project(insight_cup LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(INSIGHT_CUP_ENABLE_REALSENSE "Build Intel RealSense input support" ON)
option(INSIGHT_CUP_ENABLE_RKNN "Build Rockchip RKNN NPU support" ON)
option(INSIGHT_CUP_ENABLE_ROS2 "Publish stable recognition labels on ROS 2" ON)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn calib3d highgui)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(INSIGHT_CUP_ONNX_ARCH "aarch64")
  set(INSIGHT_CUP_DEBIAN_ARCH "aarch64-linux-gnu")
  set(INSIGHT_CUP_LOCAL_PACKAGE_ROOT
      "${CMAKE_SOURCE_DIR}/third_party/packages-arm64/root")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(INSIGHT_CUP_ONNX_ARCH "x64")
  set(INSIGHT_CUP_DEBIAN_ARCH "x86_64-linux-gnu")
  set(INSIGHT_CUP_LOCAL_PACKAGE_ROOT
      "${CMAKE_SOURCE_DIR}/third_party/cpp-httplib-package")
else()
  message(FATAL_ERROR
          "Unsupported CPU architecture for bundled dependencies: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(INSIGHT_CUP_HAS_RKNN 0)
if(INSIGHT_CUP_ENABLE_RKNN AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(INSIGHT_CUP_DEFAULT_RKNN_RUNTIME_ROOT
      "${CMAKE_SOURCE_DIR}/../../.cache/insight-cup-rknn/runtime-2.3.2")
  set(RKNN_RUNTIME_ROOT
      "${INSIGHT_CUP_DEFAULT_RKNN_RUNTIME_ROOT}"
      CACHE PATH "RKNN Runtime SDK root")
  if(EXISTS "${RKNN_RUNTIME_ROOT}/rknn_api.h" AND
     EXISTS "${RKNN_RUNTIME_ROOT}/librknnrt.so")
    add_library(rknn_runtime SHARED IMPORTED GLOBAL)
    set_target_properties(rknn_runtime PROPERTIES
      IMPORTED_LOCATION "${RKNN_RUNTIME_ROOT}/librknnrt.so"
      INTERFACE_INCLUDE_DIRECTORIES "${RKNN_RUNTIME_ROOT}")
    set(INSIGHT_CUP_HAS_RKNN 1)
  else()
    message(WARNING
            "RKNN Runtime SDK not found under ${RKNN_RUNTIME_ROOT}; NPU support disabled")
  endif()
endif()

foreach(BROTLI_COMPONENT IN ITEMS dec enc common)
  string(TOUPPER "${BROTLI_COMPONENT}" BROTLI_COMPONENT_UPPER)
  find_library(BROTLI_${BROTLI_COMPONENT_UPPER}_LIBRARY
    NAMES "brotli${BROTLI_COMPONENT}"
    HINTS
      "${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/usr/lib/${INSIGHT_CUP_DEBIAN_ARCH}")
  if(NOT BROTLI_${BROTLI_COMPONENT_UPPER}_LIBRARY)
    message(FATAL_ERROR "Brotli ${BROTLI_COMPONENT} library was not found")
  endif()
endforeach()

set(ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-${INSIGHT_CUP_ONNX_ARCH}-1.23.2"
    CACHE PATH "ONNX Runtime C/C++ SDK root")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
          "ONNX Runtime ${INSIGHT_CUP_ONNX_ARCH} SDK not found under ${ONNXRUNTIME_ROOT}")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.23.2"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

set(CPPHTTPLIB_ROOT "${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}")
if(NOT EXISTS "${CPPHTTPLIB_ROOT}/usr/include/httplib.h")
  message(FATAL_ERROR "cpp-httplib development package is missing")
endif()
set(CPPHTTPLIB_LIBRARY
    "${CPPHTTPLIB_ROOT}/usr/lib/${INSIGHT_CUP_DEBIAN_ARCH}/libcpp-httplib.so.0.10.3")
if(NOT EXISTS "${CPPHTTPLIB_LIBRARY}")
  message(FATAL_ERROR "cpp-httplib library not found: ${CPPHTTPLIB_LIBRARY}")
endif()
add_library(cpp_httplib SHARED IMPORTED GLOBAL)
set_target_properties(cpp_httplib PROPERTIES
  IMPORTED_LOCATION "${CPPHTTPLIB_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CPPHTTPLIB_ROOT}/usr/include")

find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
  set(NLOHMANN_JSON_INCLUDE_DIR
      "${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/usr/include")
  if(NOT EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR
            "nlohmann/json.hpp was not found in the system or local packages")
  endif()
  add_library(nlohmann_json_local INTERFACE)
  target_include_directories(nlohmann_json_local INTERFACE
                             "${NLOHMANN_JSON_INCLUDE_DIR}")
  add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json_local)
endif()

if(INSIGHT_CUP_ENABLE_REALSENSE)
  set(INSIGHT_CUP_LOCAL_REALSENSE_ROOT
      "${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/opt/ros/humble")
  find_library(INSIGHT_CUP_LOCAL_REALSENSE_LIBRARY
    NAMES realsense2
    PATHS "${INSIGHT_CUP_LOCAL_REALSENSE_ROOT}/lib/${INSIGHT_CUP_DEBIAN_ARCH}"
    NO_DEFAULT_PATH)
  if(INSIGHT_CUP_LOCAL_REALSENSE_LIBRARY AND
     EXISTS "${INSIGHT_CUP_LOCAL_REALSENSE_ROOT}/include/librealsense2/rs.hpp")
    add_library(realsense2::realsense2 SHARED IMPORTED GLOBAL)
    set_target_properties(realsense2::realsense2 PROPERTIES
      IMPORTED_LOCATION "${INSIGHT_CUP_LOCAL_REALSENSE_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES
        "${INSIGHT_CUP_LOCAL_REALSENSE_ROOT}/include")
    set(INSIGHT_CUP_HAS_REALSENSE 1)
  else()
    find_package(realsense2 QUIET)
    if(realsense2_FOUND)
      set(INSIGHT_CUP_HAS_REALSENSE 1)
    else()
      message(WARNING "librealsense2 was not found; RealSense input will be disabled")
      set(INSIGHT_CUP_HAS_REALSENSE 0)
    endif()
  endif()
else()
  set(INSIGHT_CUP_HAS_REALSENSE 0)
endif()

set(INSIGHT_CUP_HAS_ROS2 0)
if(INSIGHT_CUP_ENABLE_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  set(INSIGHT_CUP_HAS_ROS2 1)
endif()

set(INSIGHT_CUP_HAS_MOTION_HOLD 0)
if(INSIGHT_CUP_HAS_ROS2)
  find_package(mission_control_interfaces QUIET)
  if(mission_control_interfaces_FOUND)
    set(INSIGHT_CUP_HAS_MOTION_HOLD 1)
  else()
    message(WARNING
            "mission_control_interfaces was not found; motion hold requests are disabled")
  endif()
endif()

set(INSIGHT_CUP_SOURCES
  cpp/src/config.cpp
  cpp/src/gallery.cpp
  cpp/src/knife/classification/recognizer.cpp
  cpp/src/knife/common.cpp
  cpp/src/knife/retrieval/recognizer.cpp
  cpp/src/knife/traditional/recognizer.cpp
  cpp/src/logging.cpp
  cpp/src/main.cpp
  cpp/src/models.cpp
  cpp/src/onnx.cpp
  cpp/src/runtime.cpp
  cpp/src/safe_obstacle.cpp
  cpp/src/scheduler.cpp
  cpp/src/state.cpp
  cpp/src/types.cpp
  cpp/src/web.cpp)
if(INSIGHT_CUP_HAS_ROS2)
  list(APPEND INSIGHT_CUP_SOURCES
    cpp/src/insight_topic.cpp
    cpp/src/safe_topic.cpp
    cpp/src/motion_hold.cpp)
endif()
if(INSIGHT_CUP_HAS_RKNN)
  list(APPEND INSIGHT_CUP_SOURCES cpp/src/rknn.cpp)
endif()
add_executable(insight-cup ${INSIGHT_CUP_SOURCES})

target_include_directories(insight-cup PRIVATE cpp/include)
target_compile_definitions(insight-cup PRIVATE
  INSIGHT_CUP_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
  INSIGHT_CUP_HAS_REALSENSE=${INSIGHT_CUP_HAS_REALSENSE}
  INSIGHT_CUP_HAS_RKNN=${INSIGHT_CUP_HAS_RKNN}
  INSIGHT_CUP_HAS_ROS2=${INSIGHT_CUP_HAS_ROS2}
  INSIGHT_CUP_HAS_MOTION_HOLD=${INSIGHT_CUP_HAS_MOTION_HOLD}
  CPPHTTPLIB_OPENSSL_SUPPORT
  CPPHTTPLIB_ZLIB_SUPPORT
  CPPHTTPLIB_BROTLI_SUPPORT)
target_compile_options(insight-cup PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(insight-cup
  onnxruntime
  cpp_httplib
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS}
  Threads::Threads
  OpenSSL::SSL
  OpenSSL::Crypto
  ZLIB::ZLIB
  "${BROTLI_DEC_LIBRARY}"
  "${BROTLI_ENC_LIBRARY}"
  "${BROTLI_COMMON_LIBRARY}")

if(INSIGHT_CUP_HAS_ROS2)
  ament_target_dependencies(insight-cup rclcpp std_msgs sensor_msgs)
  if(INSIGHT_CUP_HAS_MOTION_HOLD)
    ament_target_dependencies(insight-cup mission_control_interfaces)
  endif()
endif()

if(INSIGHT_CUP_HAS_REALSENSE)
  target_link_libraries(insight-cup realsense2::realsense2)
endif()
if(INSIGHT_CUP_HAS_RKNN)
  target_link_libraries(insight-cup rknn_runtime)
endif()

if(INSIGHT_CUP_HAS_REALSENSE)
  add_library(insight_cup_stereo_range STATIC cpp/src/stereo_range.cpp)
  target_include_directories(insight_cup_stereo_range PUBLIC cpp/include)
  target_compile_options(insight_cup_stereo_range PRIVATE
    -Wall -Wextra -Wpedantic)
  target_link_libraries(insight_cup_stereo_range PUBLIC
    realsense2::realsense2 ${OpenCV_LIBS})

  target_link_libraries(insight-cup insight_cup_stereo_range)

  set(INSIGHT_CUP_YOLO_PREVIEW_SOURCES
    tools/depth/yolo_stereo_range_preview.cpp
    cpp/src/gallery.cpp
    cpp/src/knife/classification/recognizer.cpp
    cpp/src/knife/common.cpp
    cpp/src/knife/retrieval/recognizer.cpp
    cpp/src/knife/traditional/recognizer.cpp
    cpp/src/logging.cpp
    cpp/src/models.cpp
    cpp/src/onnx.cpp
    cpp/src/types.cpp)
  if(INSIGHT_CUP_HAS_RKNN)
    list(APPEND INSIGHT_CUP_YOLO_PREVIEW_SOURCES cpp/src/rknn.cpp)
  endif()
  add_executable(insight-cup-yolo-range-preview
    ${INSIGHT_CUP_YOLO_PREVIEW_SOURCES})
  target_include_directories(insight-cup-yolo-range-preview PRIVATE cpp/include)
  target_compile_definitions(insight-cup-yolo-range-preview PRIVATE
    INSIGHT_CUP_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
    INSIGHT_CUP_HAS_RKNN=${INSIGHT_CUP_HAS_RKNN})
  target_compile_options(insight-cup-yolo-range-preview PRIVATE
    -Wall -Wextra -Wpedantic)
  target_link_libraries(insight-cup-yolo-range-preview PRIVATE
    insight_cup_stereo_range
    onnxruntime
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    Threads::Threads)
  if(INSIGHT_CUP_HAS_RKNN)
    target_link_libraries(insight-cup-yolo-range-preview PRIVATE rknn_runtime)
  endif()
  set_target_properties(insight-cup-yolo-range-preview PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/opt/ros/humble/lib/${INSIGHT_CUP_DEBIAN_ARCH};${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/opt/ros/humble/lib;${RKNN_RUNTIME_ROOT}")

  add_executable(insight-cup-stereo-range-test
    tools/depth/stereo_range_test.cpp)
  target_include_directories(insight-cup-stereo-range-test PRIVATE cpp/include)
  target_compile_options(insight-cup-stereo-range-test PRIVATE
    -Wall -Wextra -Wpedantic)
  target_link_libraries(insight-cup-stereo-range-test PRIVATE
    insight_cup_stereo_range)
endif()

set_target_properties(insight-cup PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CPPHTTPLIB_ROOT}/usr/lib/${INSIGHT_CUP_DEBIAN_ARCH};${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/opt/ros/humble/lib/${INSIGHT_CUP_DEBIAN_ARCH};${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/opt/ros/humble/lib;${RKNN_RUNTIME_ROOT}"
  INSTALL_RPATH "$ORIGIN/../lib")

include(CTest)
if(BUILD_TESTING)
  add_test(NAME insight_cup_help COMMAND insight-cup --help)
  if(INSIGHT_CUP_HAS_REALSENSE)
    add_executable(insight-cup-stereo-range-unit-test
      tests/cpp/test_stereo_range.cpp)
    target_include_directories(insight-cup-stereo-range-unit-test PRIVATE
      cpp/include)
    target_compile_options(insight-cup-stereo-range-unit-test PRIVATE
      -Wall -Wextra -Wpedantic)
    target_link_libraries(insight-cup-stereo-range-unit-test PRIVATE
      insight_cup_stereo_range)
    add_test(NAME insight_cup_stereo_range
      COMMAND insight-cup-stereo-range-unit-test)
  endif()
  add_executable(insight-cup-safe-obstacle-unit-test
    tests/cpp/test_safe_obstacle.cpp
    cpp/src/config.cpp
    cpp/src/logging.cpp
    cpp/src/safe_obstacle.cpp)
  target_include_directories(insight-cup-safe-obstacle-unit-test PRIVATE
    cpp/include)
  target_compile_definitions(insight-cup-safe-obstacle-unit-test PRIVATE
    INSIGHT_CUP_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_compile_options(insight-cup-safe-obstacle-unit-test PRIVATE
    -Wall -Wextra -Wpedantic)
  target_link_libraries(insight-cup-safe-obstacle-unit-test PRIVATE
    nlohmann_json::nlohmann_json ${OpenCV_LIBS} Threads::Threads)
  add_test(NAME insight_cup_safe_obstacle
    COMMAND insight-cup-safe-obstacle-unit-test)
endif()
