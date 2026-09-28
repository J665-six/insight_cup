# 运行资源优化评估（2026-09-28）

## 结论

不建议为了降低占用而把整套项目重写为 C++。YOLO/PyTorch、ONNX Runtime、
OpenCV 的主要计算内核已经由 C/C++ 实现，Python 负责模型装配、路由、日志和 UI。
原始高 CPU 的主要原因是 ONNX Runtime 线程参数未传到 InsightFace、多个会话空闲
自旋，以及离线视频不按源 FPS 节流，不是 Python 解释器。

当前默认选择 `.pt` YOLO，适合低 CPU、接近实时运行。内存受限时可以切换直接
ONNX YOLO；它能明显降低内存，但本机 CPU 和耗时更高。

## 测试条件

- CPU：Intel Core i7-14650HX，24 个逻辑 CPU
- 视频：`f2_f5_f8_d455_color_20260918_093604_648930.mp4`
- 分辨率与长度：1280x720，30 FPS，1746 帧（58.2 秒）
- 无 UI、无标注视频、无裁剪保存
- 人脸和刀具 ONNX 各 1 个计算线程，二阶段各 2 个异步 worker
- 人脸检测输入 320，未确认刷新 15 帧，稳定后刷新 30 帧

## 根因与改动

1. InsightFace vendor 路由原先没有把 `sess_options` 传入 ONNX Runtime，检测和
   特征模型都显示 `intra_op_num_threads=0`，会自动占满机器并主动自旋。现已统一
   使用受控 ONNX 会话：明确 intra-op 线程、inter-op 为 1、顺序执行、关闭空闲
   自旋。
2. 离线视频原先尽可能快地读取，会把“批处理速度”误当成“实时相机占用”。现在
   默认按源 FPS 节流；`--unthrottled-video` 仅用于极速批处理。
3. 同一稳定轨迹原先始终每 15 帧重复二阶段推理。现在首次确认仍为 15 帧，确认后
   改为 30 帧。完整视频二阶段事件从 320 降到 200，首次确认速度不变。
4. 无 UI 时不再生成预览 JPEG 或保留网页事件；OpenCV、PyTorch、MKL、OpenBLAS
   及 ONNX Runtime 的线程数均受到限制。
5. 新增直接 ONNX YOLO 后端，按权重后缀自动选择，不加载 Ultralytics/PyTorch。

## 资源结果

早期 300 帧压力测试用于定位线程池问题，当时视频未按源 FPS 节流：

| 配置 | 平均 CPU | 峰值 RSS | 300 帧耗时 |
| --- | ---: | ---: | ---: |
| 原始线程配置 | 1389% | 2.10 GiB | 12.64 秒 |
| 修复 InsightFace ONNX 线程 | 282% | 2.09 GiB | 9.36 秒 |

以下结果均为完整 1746 帧、按 30 FPS 节流的实时口径：

| 配置 | 平均 CPU | 峰值 RSS | 1746 帧耗时 |
| --- | ---: | ---: | ---: |
| `.pt`，固定 15 帧刷新 | 163% | 1.95 GiB | 64.61 秒 |
| `.pt`，稳定后 30 帧刷新（当前默认） | **115%** | 1.96 GiB | **64.38 秒** |
| 直接 ONNX YOLO，2 线程 | 286% | **0.80 GiB** | 87.94 秒 |

`.pt` 当前默认相对固定 15 帧刷新减少约 29% 的平均 CPU，完整视频耗时没有增加。
内存基本不变，因为约 1.6 GiB 来自 Ultralytics/PyTorch 运行时，而不是 5.3 MB 的
权重文件。直接 ONNX 后端避开 PyTorch，因此节省约 60% 峰值内存。

## 识别一致性

`.pt` 与直接 ONNX 对前 300 帧逐框比较：907 个框成功匹配，平均 IoU 0.9935，
三大类一致率 100%。完整视频中，当前默认稳定输出只有 `f2/f5/f8`，没有额外错误
身份；稳定标签切换从固定刷新配置的 1 次降为 0 次。当前默认的二阶段返回延迟
中位数为 8 帧，固定刷新配置为 9 帧。

## 使用建议

默认低 CPU、接近实时：

```bash
./start.sh --source realsense
```

只看日志并关闭媒体保存：

```bash
./start.sh --source realsense --no-save-video --no-save-crops
```

低内存（约 0.8 GiB，CPU 更高且当前不能保持 30 FPS）：

```bash
./start.sh \
  --weights /home/j/trainv5/runs/yolo11n_trainv5_continue_latest-2/weights/best.onnx \
  --yolo-cpu-threads 2
```

如果后续同时要求更低内存和更低 CPU，优先评估 RTX 5060 上的 TensorRT FP16，
或 Intel CPU 上的 OpenVINO/INT8，并重新做逐框和身份回归测试。把 Python 路由、
日志和 UI 改写为 C++，但仍使用相同 libtorch/ONNX 模型，不会带来同量级收益。
