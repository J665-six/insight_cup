# Fixed C++ Dependencies

The native runtime builds against the following locally pinned packages:

- `onnxruntime-linux-x64-1.23.2/`: official Microsoft ONNX Runtime CPU SDK.
  Its upstream `LICENSE` and `ThirdPartyNotices.txt` are kept in that directory.
- `cpp-httplib-package/`: Ubuntu Jammy packages `libcpp-httplib-dev` and
  `libcpp-httplib0`, version `0.10.3+ds-1`. The Debian copyright manifest is
  available under `usr/share/doc/libcpp-httplib-dev/copyright`.

OpenCV, librealsense2, OpenSSL, zlib, Brotli, Threads, and nlohmann-json are
resolved from the host system by CMake.
