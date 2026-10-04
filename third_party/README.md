# Fixed C++ Dependencies

The native runtime builds against the following locally pinned packages:

- `onnxruntime-linux-x64-1.23.2/`: official Microsoft ONNX Runtime CPU SDK.
  Its upstream `LICENSE` and `ThirdPartyNotices.txt` are kept in that directory.
- `onnxruntime-linux-aarch64-1.23.2/`: the matching CPU SDK used on RK3588.
- `cpp-httplib-package/`: Ubuntu Jammy packages `libcpp-httplib-dev` and
  `libcpp-httplib0`, version `0.10.3+ds-1`. The Debian copyright manifest is
  available under `usr/share/doc/libcpp-httplib-dev/copyright`.
- `packages-arm64/root/`: locally extracted Ubuntu/ROS packages for cpp-httplib,
  nlohmann-json, Brotli, and librealsense2 on the RK3588 host.

The RKNN Runtime 2.3.2 SDK is installed separately at
`../../.cache/insight-cup-rknn/runtime-2.3.2`. CMake links directly
to its `librknnrt.so` so the older OS-provided Runtime 1.4.0 is never loaded.
Set the `RKNN_RUNTIME_ROOT` CMake cache variable when using another location.

On x86_64, CMake uses the bundled cpp-httplib package and resolves the remaining
development libraries from the host. On aarch64, it prefers the extracted ARM64
package root for cpp-httplib, librealsense2, and nlohmann-json while continuing
to use the host OpenCV, OpenSSL, zlib, Threads, and available Brotli libraries.
