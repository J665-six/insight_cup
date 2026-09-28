# Insight Cup 视觉识别系统

本项目把 YOLO 大类检测、人脸身份识别和刀具分类组织成一条可实时运行、可追踪
的识别链路。YOLO 保留 `fN/kN` 细分类能力用于调试，但正式二阶段只接收框、
裁剪图和 `b0/f/k` 大类。

## 一键启动（C++）

正式识别链路已迁移到 C++17，运行时不再启动 Python、PyTorch、Ultralytics、
NumPy、Faiss 或 pyrealsense2。首次构建：

```bash
./build_cpp.sh
```

构建依赖为 CMake、G++、OpenCV C++、librealsense2、OpenSSL、zlib 和 Brotli。
项目已固定 ONNX Runtime `1.23.2` 与 cpp-httplib 的本地依赖，构建过程不访问网络。
`start.sh` 在可执行文件不存在时会自动构建，之后只做增量编译。

默认通过 RealSense SDK 打开 Intel RealSense D455 的彩色摄像头：

```bash
./start.sh
```

一键启动脚本默认只在终端打印识别日志，不启动 HTTP UI，也不会生成 UI 预览 JPEG。
需要调试界面时显式添加脚本参数：

```bash
./start.sh --ui
```

机器连接多台 RealSense 时，可以固定当前 D455 的序列号：

```bash
./start.sh --realsense-serial 038122250473
```

实时识别使用 D455 的 `1280x720@30 FPS` BGR 彩色流。设备由
librealsense2 C++ API 按序列号选择，不依赖可能变化的 `/dev/videoN`。D455 的深度帧
当前不参与 YOLO、人脸或刀具识别。

使用已录制视频：

```bash
./start.sh \
  --source /home/j/knife_recording/f2_f5_f8_d455_color_20260918_093951_907387.mp4
```

终端会实时打印模型加载和每个识别目标。使用 `--ui` 时，调试界面默认地址：

`http://127.0.0.1:7860`

界面模式会自动打开浏览器。只启动服务、不自动打开浏览器时使用
`./start.sh --ui --no-open-browser`。离线任务需要在视频结束后自动关闭 UI 服务时，
再添加 `--exit-on-complete`。

常用参数：

```bash
./start.sh --realsense-serial 038122250473
./start.sh --source realsense --camera-width 1280 --camera-height 720 --camera-fps 30
./start.sh --source 0 --vid-stride 2  # 显式使用普通 OpenCV 摄像头
./start.sh --source video.mp4 --max-frames 100
./start.sh --source image.jpg --port 7861
./start.sh --source video.mp4  # 默认无 UI
./start.sh --ui --source video.mp4 --no-open-browser
./start.sh --source video.mp4 --show-raw-label
```

界面提供实时标注画面、暂停/继续/停止、模块健康状态、FPS、阶段耗时、三大类
计数和最近识别事件。完整链路与事件字段见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

实时相机模式下，UI 通过 MJPEG 连续接收最新标注帧，不再以单张 JPEG 低频轮询。
YOLO 保持逐帧检测；较慢的人脸、刀具二阶段在独立后台线程运行，并依据大类和框
位置关联短期结果。新目标会立即提交识别；未确认结果每 15 帧刷新，身份确认后改为
每 30 帧刷新，减少静止画面中的重复计算，同时不延长首次确认时间：

```bash
./start.sh --stage2-refresh-frames 15 --stage2-stable-refresh-frames 30
./start.sh --stage2-cache-iou 0.55 --stage2-workers 2
./start.sh --face-cpu-threads 1 --knife-cpu-threads 1
./start.sh --preview-width 960 --jpeg-quality 82
```

人脸与刀具默认按各自的 `track_id` 使用最近 3 次结果投票。首次身份和后续新类别
都需要连续 2 次成为赢家才会显示，短暂的 `unknown/no_face` 不会立刻清空稳定结果：

```bash
./start.sh --stage2-smoothing vote --stage2-smoothing-window 3 \
  --stage2-switch-confirmations 2
./start.sh --stage2-smoothing mean  # 对完整类别分数求滑动平均
./start.sh --stage2-smoothing none  # 关闭时序处理，用于基线对照
```

UI 只把确认后的结果画成身份标签；未确认、拒识状态不会显示成类似 `f7?` 的身份。
终端仍打印第一候选和第二候选供调试，`events.jsonl` 的 `stage2.temporal.raw` 保留
同次推理的原始结果，用于评估平滑前后的差异。

UI 默认传输 `960x540` 预览，保存的视频仍保持 D455 原始 `1280x720`。视频和
视频源也使用与实时相机相同的异步二阶段调度，保证调试画面不会被多个目标阻塞；
视频结束时会等待并写入尚未消费的二阶段结果。图片源仍采用逐帧同步识别。终端默认
只在新结果到达时打印 `RESULT`，内容为第一候选标签/分数和第二候选标签/分数；
完整路由、状态与耗时仍保存在 `events.jsonl`。`--verbose` 可恢复逐帧性能日志。

离线视频默认按源视频 FPS 播放，因此资源压力接近实时相机。只有需要尽快批处理时
才添加 `--unthrottled-video`。只看终端结果、不需要生成标注视频和裁剪时可进一步
减少编码及磁盘写入：

```bash
./start.sh --source /path/to/video.mp4 --no-save-video --no-save-crops
```

正式 C++ 运行时固定使用导出的 YOLO ONNX，不再支持 `.pt` 权重。这样不会加载
Python 和 PyTorch；同一段 60 帧回归视频中，C++ 峰值 RSS 为约 `721 MiB`，原
Python `.pt` 链路的完整视频测试峰值约 `1.96 GiB`：

```bash
./start.sh \
  --weights models/yolo/best.onnx \
  --yolo-cpu-threads 2
```

完整测试数据和选择建议见
[`training/temporal/evaluation/resource_optimization_20260928.md`](training/temporal/evaluation/resource_optimization_20260928.md)。

## 项目结构

```text
insight_cup/
  CMakeLists.txt           C++17 构建定义
  build_cpp.sh             一键构建
  start.sh                 C++ 正式运行入口
  start_python.sh          迁移回归用旧 Python 入口
  cpp/
    include/insight_cup/   阶段契约与模块接口
    src/                   模型、调度、输入、日志和 UI 服务实现
  src/insight_cup/
    app/ui/                C++ 服务复用的静态调试界面资源
    ...                    迁移回归用 Python 参考实现
  tools/
    inference/             YOLO/阶段一/阶段二独立命令
    data/                  模型、图库与数据集准备
    evaluation/            离线评估
  models/                  正式部署模型
  data/                    人脸库、刀具数据集与原始样本
  training/                训练配置、权重与评估报告
  tests/                   单元与接口测试
  outputs/                 每次运行和历史结果
  third_party/             固定的 C++ 运行依赖与许可证
  requirements/            Python 训练、数据工具和回归依赖
```

## 当前模型

- YOLO 默认：`models/yolo/best.onnx`
- 人脸检测：`models/face/models/face_only_r50/det_10g.onnx`
- 人脸特征：`models/face/models/face_only_r50/w600k_r50.onnx`
- 人脸库：`data/face_gallery/cpp/trainv5_gallery_r50.icg`
- 刀具分类：`models/knife/pplcnetv2_base_knife10/inference.onnx`
- 刀具特征：`models/knife/ppshitu_v2_retrieval/inference.onnx`
- 刀具特征库：`data/knife_gallery/cpp/trainv5_ppshitu_v2.icg`

默认阈值：

- 人脸：相似度 `0.40`，第一/第二候选 margin `0.03`
- 刀具分类：置信度 `0.50`，第一/第二候选 margin `0.15`
- 刀具检索：余弦相似度 `0.50`，不同标签候选 margin `0.02`

刀具默认仍使用当前分类器。切换到 PaddleClas PP-ShiTuV2 特征检索：

```bash
./start.sh --knife-mode retrieval
```

两种模式应使用同一视频、不同 `--session-id` 分别运行后比较。检索图库由 2296 张
训练裁剪建立，离线 Top-1 为 val `54.22%`、test `76.77%`；当前分类器对应为
val `70.14%`、test `80.67%`，因此检索模式目前是对比实验，不是默认替代。

## 会话输出

默认写入 `outputs/runtime/<session-id>/`：

- `manifest.json`：本次参数、模型信息和明确的数据传递路径
- `events.jsonl`：每个目标的 stage1、route、stage2 和耗时
- `summary.json`：最终统计与退出状态
- `annotated.mp4` 或 `annotated_images/`：标注媒体
- `crops/f/`、`crops/k/`：实际送入二阶段的裁剪

终端事件示例：

```text
RESULT frame=42 id=frame000042:det001 label=k4 probability=0.810 second=k2 second_probability=0.180
```

默认会在 YOLO 的 `fN/kN` 合并为大类后再次执行 NMS，并按整帧目标位置和运动
方向关联轨迹。人脸或刀具身份需连续两次确认后才显示，短时异常不会立即切换标签。

倾斜人脸专项测试可开启按眼睛角度校正；该选项会增加二阶段耗时，因此默认关闭：

```bash
./start.sh --face-rotation-retry-degrees 25
```

## Python 工具边界

训练、数据准备、图库重建和离线评估继续使用 InsightFace、PaddleClas 与
Ultralytics 的上游 Python 生态；这些工具不会被正式 `start.sh` 加载。迁移对照时
可通过 `./start_python.sh ...` 运行旧实现。

### 独立工具

只运行 YOLO：

```bash
python3 tools/inference/yolo_detect.py --source /path/to/video.mp4
```

生成文件式阶段一交接：

```bash
python3 tools/inference/stage1_yolo.py \
  --source /path/to/video.mp4 \
  --out outputs/stage1_test
```

分别运行人脸与刀具阶段二：

```bash
python3 tools/inference/stage2_face.py \
  --handoff outputs/stage1_test/stage1_handoff.json

python3 tools/inference/stage2_knife.py \
  --handoff outputs/stage1_test/stage1_handoff.json

python3 tools/inference/stage2_knife.py \
  --mode retrieval \
  --handoff outputs/stage1_test/stage1_handoff.json
```

评估刀具模型：

```bash
python3 tools/evaluation/evaluate_knife.py --split test --provider cpu
```

当前独立测试集 Top-1 为 `80.67%`、Top-2 为 `89.41%`；应用默认拒识规则后，
覆盖率 `73.98%`，已接受结果准确率 `93.47%`。报告位于
`training/knife/evaluation/`。

## 数据与训练工具

建立人脸参考库：

```bash
python3 tools/data/export_face_references.py \
  --data /home/j/trainv5/data.yaml \
  --split train \
  --output data/face_gallery/references_trainv5 \
  --per-class 20

python3 tools/data/prepare_face_gallery.py \
  --references data/face_gallery/references_trainv5 \
  --output data/face_gallery/trainv5_gallery_r50.npz \
  --det-thresh 0.25
```

导出刀具分类数据：

```bash
python3 tools/data/export_knife_dataset.py \
  --data /home/j/trainv5/data.yaml \
  --output data/knife_dataset/trainv5_grouped
```

刀具训练配置为 `training/knife/PPLCNetV2_base_knife10.yaml`，使用 PaddleClas
`release/2.6` 的完整外部仓库训练。正式相机进程只依赖导出的 ONNX，不加载
PaddlePaddle。

重建 PP-ShiTuV2 刀具特征库：

```bash
python3 tools/data/build_knife_retrieval_gallery.py
```

## 上游代码

C++ 的 SCRFD 解码、NMS、ArcFace 五点对齐和 PaddleClas 预处理按下面两个固定上游
版本等价移植；模型计算仍由官方导出的 ONNX 图执行，没有重新实现神经网络：

- `src/insight_cup/vendor/insightface_core/`：InsightFace commit
  `7fadd420c2351d0ffa8cac403421c1a3ed733365`，仅保留 SCRFD、ArcFace、五点对齐
  和必要模型加载代码。
- `src/insight_cup/vendor/paddleclas_core/`：PaddleClas commit
  `f1233c18455b8acde4fc42ab0bea575fa06daa8e`，保留官方分类预处理、Top-k、
  PP-ShiTu 特征归一化和 Faiss 内积检索。

各目录内保留 `UPSTREAM.md`、`SOURCE_MANIFEST.json` 和许可证。人脸模块不包含
3D 人脸、活体、换脸、属性或 GUI；刀具模块不包含 PaddleClas 的主体检测、
服务端、移动端和训练引擎。

InsightFace 代码为 MIT 许可；官方预训练模型的提供方限定其为非商业研究用途。
人脸参考图片和特征库属于生物特征数据，应保存在受控环境。

## 测试

```bash
./build_cpp.sh
ctest --test-dir build --output-on-failure
./start.sh --source /path/to/video.mp4 --max-frames 120 \
  --unthrottled-video --no-save-video --no-save-crops

# 旧 Python 回归测试
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

C++ 与 Python 同帧回归结果记录在
[`docs/CPP_MIGRATION.md`](docs/CPP_MIGRATION.md)。
