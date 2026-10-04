cmake_minimum_required(VERSION 3.16)

project(insight_cup LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(INSIGHT_CUP_ENABLE_REALSENSE "Build Intel RealSense input support" ON)
option(INSIGHT_CUP_ENABLE_RKNN "Build Rockchip RKNN NPU support" ON)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn calib3d)
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

set(INSIGHT_CUP_SOURCES
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
if(INSIGHT_CUP_HAS_RKNN)
  list(APPEND INSIGHT_CUP_SOURCES cpp/src/rknn.cpp)
endif()
add_executable(insight-cup ${INSIGHT_CUP_SOURCES})

target_include_directories(insight-cup PRIVATE cpp/include)
target_compile_definitions(insight-cup PRIVATE
  INSIGHT_CUP_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
  INSIGHT_CUP_HAS_REALSENSE=${INSIGHT_CUP_HAS_REALSENSE}
  INSIGHT_CUP_HAS_RKNN=${INSIGHT_CUP_HAS_RKNN}
  CPPHTTPLIB_OPENSSL_SUPPORT
  CPPHTTPLIB_ZLIB_SUPPORT
  CPPHTTPLIB_BROTLI_SUPPORT)
target_compile_options(insight-cup PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(insight-cup PRIVATE
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

if(INSIGHT_CUP_HAS_REALSENSE)
  target_link_libraries(insight-cup PRIVATE realsense2::realsense2)
endif()
if(INSIGHT_CUP_HAS_RKNN)
  target_link_libraries(insight-cup PRIVATE rknn_runtime)
endif()

set_target_properties(insight-cup PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib;${CPPHTTPLIB_ROOT}/usr/lib/${INSIGHT_CUP_DEBIAN_ARCH};${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/opt/ros/humble/lib/${INSIGHT_CUP_DEBIAN_ARCH};${INSIGHT_CUP_LOCAL_PACKAGE_ROOT}/opt/ros/humble/lib;${RKNN_RUNTIME_ROOT}"
  INSTALL_RPATH "$ORIGIN/../lib")

include(CTest)
if(BUILD_TESTING)
  add_test(NAME insight_cup_help COMMAND insight-cup --help)
endif()
