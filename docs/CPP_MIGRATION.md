# C++ 运行时迁移记录

## 范围

`start.sh` 启动的正式链路已迁移到 C++17：

- OpenCV 视频、图片和普通摄像头输入
- librealsense2 D455 彩色流
- YOLO ONNX 检测、原始分类别 NMS、大类折叠与大类二次 NMS
- InsightFace SCRFD 检测、五点对齐、ArcFace 特征与身份图库匹配
- PaddleClas PPLCNetV2 分类和 PP-ShiTuV2 特征检索
- 异步二阶段、轨迹关联、滑动均值/投票与切换确认
- JSONL 日志、会话清单、标注媒体、裁剪和 HTTP 调试 UI

训练、模型导出、数据准备和评估继续使用上游 Python 工具。旧 Python 实时实现保留
为迁移回归基准，入口为 `start_python.sh`，不进入 C++ 正式运行进程。

## 上游一致性

- InsightFace commit: `7fadd420c2351d0ffa8cac403421c1a3ed733365`
- PaddleClas commit: `f1233c18455b8acde4fc42ab0bea575fa06daa8e`
- ONNX Runtime: `1.23.2`
- SCRFD: stride `8/16/32`、每位置 2 anchors、阈值 `0.18`、NMS `0.4`
- ArcFace: `112x112` 五点相似变换，`(RGB - 127.5) / 127.5`
- PaddleClas: `224x224` RGB、`1/255`、ImageNet mean/std、CHW

## 回归结果

测试日期：2026-09-28。

同一视频抽取第 0、30、60 帧，以 C++ 和 Python ONNX 实现分别同步运行：

| 模块 | 事件数 | YOLO 框/类别 | 候选/状态 | 最大分数绝对差 |
| --- | ---: | --- | --- | ---: |
| Face SCRFD + ArcFace | 9 | 完全一致 | 完全一致 | 0.002689 |
| Knife classification | 12 | 完全一致 | 完全一致 | 0.0000049 |
| Knife retrieval | 12 | 完全一致 | 完全一致 | 0.0000017 |

60 帧异步视频烟雾测试成功完成并正确识别 `f2/f5/f8`。该次 C++ 进程峰值 RSS
为 `738280 KiB`（约 721 MiB），CPU 平均占用 `151%`，处理 60 帧耗时 5.21 秒。
这是短视频、无 UI、无保存输出的迁移验证数据，不替代完整视频基准。

UI 测试已验证 `/healthz`、`/api/state`、`/api/events`、`/api/frame.jpg`、
`/api/stream.mjpg` 和静态首页；JPEG 预览为有效的 `960x540` 三通道图像。MJPEG
连续读取 2 秒收到 1,071,185 bytes，包含有效 multipart 分隔符和 JPEG 数据。
