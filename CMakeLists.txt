cmake_minimum_required(VERSION 3.16)

project(insight_cup LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(INSIGHT_CUP_ENABLE_REALSENSE "Build Intel RealSense input support" ON)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn calib3d)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

set(ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.23.2"
    CACHE PATH "ONNX Runtime C/C++ SDK root")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime headers not found under ${ONNXRUNTIME_ROOT}")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.23.2"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

set(CPPHTTPLIB_ROOT "${CMAKE_SOURCE_DIR}/third_party/cpp-httplib-package")
if(NOT EXISTS "${CPPHTTPLIB_ROOT}/usr/include/httplib.h")
  message(FATAL_ERROR "cpp-httplib development package is missing")
endif()
add_library(cpp_httplib SHARED IMPORTED GLOBAL)
set_target_properties(cpp_httplib PROPERTIES
  IMPORTED_LOCATION "${CPPHTTPLIB_ROOT}/usr/lib/x86_64-linux-gnu/libcpp-httplib.so.0.10.3"
  INTERFACE_INCLUDE_DIRECTORIES "${CPPHTTPLIB_ROOT}/usr/include")

if(INSIGHT_CUP_ENABLE_REALSENSE)
  find_package(realsense2 QUIET PATHS
    /opt/ros/humble/lib/x86_64-linux-gnu/cmake/realsense2
    NO_DEFAULT_PATH)
  if(realsense2_FOUND)
    set(INSIGHT_CUP_HAS_REALSENSE 1)
  else()
    message(WARNING "librealsense2 was not found; RealSense input will be disabled")
    set(INSIGHT_CUP_HAS_REALSENSE 0)
  endif()
else()
  set(INSIGHT_CUP_HAS_REALSENSE 0)
endif()

add_executable(insight-cup
  cpp/src/config.cpp
  cpp/src/gallery.cpp
  cpp/src/logging.cpp
  cpp/src/main.cpp
  cpp/src/models.cpp
  cpp/src/onnx.cpp
  cpp/src/runtime.cpp
  cpp/src/scheduler.cpp
  cpp/src/state.cpp
  cpp/src/types.cpp
  cpp/src/web.cpp)

target_include_directories(insight-cup PRIVATE cpp/include)
target_compile_definitions(insight-cup PRIVATE
  INSIGHT_CUP_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
  INSIGHT_CUP_HAS_REALSENSE=${INSIGHT_CUP_HAS_REALSENSE}
  CPPHTTPLIB_OPENSSL_SUPPORT
  CPPHTTPLIB_ZLIB_SUPPORT
  CPPHTTPLIB_BROTLI_SUPPORT)
target_compile_options(insight-cup PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(insight-cup PRIVATE
  onnxruntime
  cpp_httplib
  ${OpenCV_LIBS}
  Threads::Threads
  OpenSSL::SSL
  OpenSSL::Crypto
  ZLIB::ZLIB
  brotlidec
  brotlienc
  brotlicommon)

if(INSIGHT_CUP_HAS_REALSENSE)
  target_link_libraries(insight-cup PRIVATE realsense2::realsense2)
endif()

set_target_properties(insight-cup PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CPPHTTPLIB_ROOT}/usr/lib/x86_64-linux-gnu"
  INSTALL_RPATH "$ORIGIN/../lib")

include(CTest)
if(BUILD_TESTING)
  add_test(NAME insight_cup_help COMMAND insight-cup --help)
endif()
