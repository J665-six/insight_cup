# 识别系统架构

## 正式数据链路

```text
RealSense D455 RGB / Video / Image
          |
          v
  YOLO detector (fN/kN/b0)
          |
          | Stage1Detection
          | id + major_class + confidence + bbox_xyxy + crop
          | raw fN/kN is excluded
          v
       Router
       /  |  \
      /   |   \
    b0    f    k
     |    |    |
     |    |    +--> PaddleClas classifier OR PP-ShiTu retrieval
     |    |                   --> k1...k10 / unknown / ambiguous
     |    +-------> InsightFace ------> f1...f10 / unknown / ambiguous / no_face
     +------------> passthrough
          |
          v
terminal log + events.jsonl + annotated frame + debug UI
```

正式运行时为单个 C++17 进程。实时模式默认以 librealsense2 C++ API 打开 Intel
RealSense D455 的彩色流，并可通过
`--realsense-serial` 固定设备。该路径不依赖 `/dev/videoN` 编号。当前阶段仅使用
D455 的 RGB 图像；深度帧尚未进入 YOLO 或两个二阶段识别模块。普通 UVC 摄像头
仍可通过 `--source 0` 等 OpenCV 设备编号显式启用。

实时相机源中，YOLO 与画面发布运行在主循环，人脸和刀具识别默认各使用两个后台
worker。YOLO 原始 `fN/kN` 合并为大类后会再次执行同大类 NMS，避免原始分类别
NMS 留下同一物体的重叠框。调度器随后对整帧目标统一分配轨迹，匹配评分综合运动
预测位置、上一位置、框尺寸和中心距离；一条旧轨迹在同一帧只能分给一个框。稳定
前默认每 15 帧刷新，身份确认后每 30 帧刷新，新目标立即进入对应 worker。因此
耗时较高的 SCRFD/ArcFace 不会阻塞相机画面，静止且已确认的目标也不会无意义地
高频重复推理。调试 UI 使用 `/api/stream.mjpg` 持续推送最新帧，慢客户端自动跳到
最新画面。
视频源也使用同一套异步调度；视频结束时等待后台任务并补写未消费的结果。图片源
保持同步调度，便于逐张检查完整识别结果。普通视频默认按源 FPS 节流；显式使用
`--unthrottled-video` 才会尽可能快地批处理。

为避免 OpenCV 和多个 ONNX Runtime 会话同时开满全部 CPU，实时进程将 OpenCV
限制为 4 线程；YOLO、人脸及刀具会话分别显式设置线程数、顺序执行并关闭空闲自旋。
正式 C++ 运行时只接受 `.onnx` YOLO 权重，不加载 PyTorch。

YOLO 的原始 `fN/kN` 只写入最终事件的 `debug` 字段，用于离线核对。它不在
`Stage1Detection` 中，`RecognitionRouter.route()` 因而无法把原始小类交给人脸
或刀具模块。

刀具二阶段由 `--knife-mode` 选择：`classification` 使用项目训练的 10 类
PPLCNetV2 Softmax 模型；`retrieval` 使用 PaddleClas 官方 PP-ShiTuV2 512 维
特征模型、由本项目刀具训练图建立的图库和 Faiss `IndexFlatIP`。两种模式接收完全
相同的 YOLO `k` 裁剪，输出也统一为第一/第二候选、分数、margin 和判定状态。

人脸和刀具的原始二阶段结果随后进入按 `track_id` 隔离的时序层。默认保存最近
3 次完整类别分数，以第一候选多数投票。首次身份和后续新赢家都必须连续出现 2 次
才成为稳定标签；确认期间 UI 显示 `confirming`，拒识时只显示状态，不把第一候选
画成正式身份。平均分模式使用相同窗口对所有类别分数求算术平均。事件的
`stage2.temporal.raw` 保留当前原始推理结果。连续 3 次无法得到可靠候选时才释放
已有稳定标签，避免短暂 `no_face/unknown` 引起闪烁，同时防止旧身份无限保留。

人脸模块支持 `--face-rotation-retry-degrees` 对无脸或明显倾斜的弱结果补偿。完整
视频 A/B 表明该功能能减少 `no_face`，但会提高二阶段耗时和返回延迟，因此默认值
为 `0`，只在倾斜样本专项调试时开启。

## 代码边界

| 路径 | 职责 |
| --- | --- |
| `cpp/src/models.cpp` | YOLO、SCRFD/ArcFace、PaddleClas 分类与 PP-ShiTu 检索 |
| `cpp/src/scheduler.cpp` | 异步线程池、几何轨迹关联和时序决策 |
| `cpp/src/runtime.cpp` | 视频、图片、D455、路由、标注与会话输出 |
| `cpp/src/state.cpp` | UI 与运行线程共享的并发状态 |
| `cpp/src/web.cpp` | HTTP API、静态 UI 与 MJPEG 推流 |
| `cpp/src/onnx.cpp` | 统一的 ONNX Runtime C++ 会话设置 |
| `src/insight_cup/app/ui/` | C++ HTTP 服务使用的 HTML/CSS/JavaScript 资源 |
| `src/insight_cup/` | 迁移回归参考，不进入 `start.sh` 正式进程 |
| `tools/inference/` | 保留的单阶段和离线识别入口 |
| `tools/data/` | 数据集、参考图库和模型准备工具 |
| `tools/evaluation/` | 独立评估工具，不进入正式实时链路 |

## 阶段间契约

正式二阶段输入定义在 `cpp/include/insight_cup/types.hpp`：

```text
Stage1Detection
  detection_id: str
  frame_index: int
  major_class: b0 | f | k
  confidence: float
  bbox_xyxy: (x1, y1, x2, y2)
  crop: BGR ndarray
```

运行事件为 `recognition_event.v1`，固定分为：

- `frame`：帧号、源时间和图像尺寸
- `stage1`：YOLO 交付的框、大类与置信度
- `route`：实际接收该裁剪的模块
- `stage2`：稳定后的匹配状态、候选、分数、margin 和可选的 `temporal.raw`
- `timing_ms`：YOLO 与二阶段耗时
- `debug`：与正式交接隔离的 YOLO 原始小类

## 会话产物

每次启动创建独立目录：

```text
outputs/runtime/<session-id>/
  manifest.json       启动参数、模型元数据与数据链路
  events.jsonl        每个目标一行的完整识别事件
  summary.json        帧数、分类计数和最终状态
  annotated.mp4       标注视频（视频或相机源）
  annotated_images/   标注图片（图片源）
  crops/f/            送入人脸模块的裁剪
  crops/k/            送入刀具模块的方形上下文裁剪
```
