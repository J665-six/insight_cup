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

构建依赖为 CMake、G++、OpenCV C++、librealsense2、OpenSSL、zlib、Brotli 和
RKNN Runtime。RK3588 上默认加载 RKNN Runtime `2.3.2`；ONNX Runtime `1.23.2`
仅作为 CPU 回退。`start.sh` 在可执行文件不存在时会自动构建，之后只做增量编译。

`--provider auto` 是默认值：RKNN 模型齐全时自动使用 NPU。需要强制检查 NPU 或
进行 CPU 对照时分别使用：

```bash
./start.sh --provider rknn
./start.sh --provider cpu
```

启动日志中 YOLO、人脸、刀具模块均应显示 `device=npu`。
RKNN 模型通过文件路径直接交给 Runtime（`rknn_init` 的 path mode），应用层不再先
把四个模型完整读入临时缓冲；这只改变加载方式，不改变模型输入、输出或后处理。

默认通过 RealSense SDK 打开 Intel RealSense D455 的彩色摄像头。正式入口提供两个
明确的运行线路：

```bash
./start.sh --profile competition
./start.sh --profile ui-recording
```

`competition` 是默认线路，只使用 RKNN/NPU，保留终端 `RESULT`、`events.jsonl`、
manifest 和 summary；它强制关闭 HTTP UI、浏览器、预览 JPEG、标注媒体、crop、
shadow 和逐帧 verbose，并使用正常进程优先级。与这些约束冲突的参数会直接报错，
不会因参数顺序而悄悄覆盖。

`ui-recording` 同时开启调试 UI、实时预览和原分辨率标注录像，默认不保存 crop、
不写 shadow，继续使用桌面友好的较低调度优先级。远程浏览器查看时可以添加：

```bash
./start.sh --profile ui-recording --no-open-browser
```

没有指定 profile 时，普通 `./start.sh` 等同于 competition。旧的 `--ui`、
`--save-video` 等手工组合仍按 `custom` 兼容线路处理，但现场运行应显式选择上面的两个
profile。UI 线路需要额外调试证据时仍可显式开启：

```bash
./start.sh --profile ui-recording --save-crops --shadow-log
```

profile 约束同时在 `start.sh` 和 C++ 配置中生效，并写入 session manifest。直接运行
`build/insight-cup` 时应同样传入 `--profile`。competition 下图片和图片目录也不会
生成 `annotated_images/`。

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
  --source ../recordings/recording.mp4
```

终端会实时打印模型加载和每个识别目标。使用 `ui-recording` 线路时，调试界面默认地址：

`http://127.0.0.1:7860`

界面模式会自动打开浏览器。只启动服务、不自动打开浏览器时使用
`./start.sh --profile ui-recording --no-open-browser`。离线任务需要在视频结束后自动关闭
UI 服务时，再添加 `--exit-on-complete`。

常用参数：

```bash
./start.sh --realsense-serial 038122250473
./start.sh --source realsense --camera-width 1280 --camera-height 720 --camera-fps 30
./start.sh --source 0 --vid-stride 2  # 显式使用普通 OpenCV 摄像头
./start.sh --source video.mp4 --max-frames 100
./start.sh --profile ui-recording --source image.jpg --port 7861
./start.sh --source video.mp4  # 默认无 UI
./start.sh --profile ui-recording --source video.mp4 --no-open-browser
./start.sh --profile ui-recording --source video.mp4 --show-raw-label
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
./start.sh --stage2-cache-iou 0.55 --stage2-workers 1
./start.sh --face-cpu-threads 1 --knife-cpu-threads 1
./start.sh --knife-min-area-ratio 0.0025  # 仅在标定后按需启用
./start.sh --preview-width 960 --jpeg-quality 82
```

RK3588 通过 NoMachine 使用桌面时，`ui-recording` 线路限制 OpenCV/阶段二线程并以
较低调度优先级运行，为桌面留出资源；`competition` 线路使用正常进程优先级。两条
线路都将离线视频按源 FPS 播放。首次部署或升级系统后，必须先执行一次完整的桌面
稳定性配置；它会备份 NoMachine/Xorg/GDM 配置，关闭失效的厂商启动单元，限制
systemd 日志，并让 GDM 等待网络校时：

```bash
sudo tools/systemd/configure-rk3588-desktop-safe.sh
sudo systemctl restart gdm.service
```

重启 GDM/NoMachine 会断开当前会话，重新连接后配置才生效。若只做无桌面基准测试，
可以显式提高 `BUILD_JOBS`、`INSIGHT_CUP_OPENCV_THREADS` 或 `--stage2-workers`。
在图形会话中，`start.sh` 和 `start_python.sh` 会在启动前检查这些设置；检查失败时
会拒绝启动，避免再次触发 Xorg 崩溃。仅在确认没有远程桌面风险的诊断场景才使用
`INSIGHT_CUP_SKIP_NOMACHINE_CHECK=1` 绕过检查。

人脸与刀具默认按各自的 `track_id` 使用最近 3 次结果投票。首次身份和后续新类别
都需要连续 2 次成为赢家才会显示，短暂的 `unknown/no_face` 不会立刻清空稳定结果：

```bash
./start.sh --stage2-smoothing vote --stage2-smoothing-window 3 \
  --stage2-switch-confirmations 2
./start.sh --stage2-smoothing mean  # 对完整类别分数求滑动平均
./start.sh --stage2-smoothing none  # 关闭时序处理，用于基线对照
```

UI 和保存录像只绘制正式接受的结果，即 `matched` 或 `passthrough` 且存在最终类别；
未确认、拒识状态不会显示成目标，但仍保留在诊断统计和完整事件日志中。
刀具面积门默认关闭，避免漏掉远处或细长的真实目标。完成具体机位标定后，可以用
`--knife-min-area-ratio` 同时筛选正式推理结果和 UI，不能只在前端隐藏。
终端仍打印第一候选和第二候选供调试，`events.jsonl` 的 `stage2.temporal.raw` 保留
同次推理的原始结果，用于评估平滑前后的差异。

UI 默认传输 `960x540` 预览，保存的视频仍保持 D455 原始 `1280x720`。视频和
视频源也使用与实时相机相同的异步二阶段调度，保证调试画面不会被多个目标阻塞；
视频结束时会等待并写入尚未消费的二阶段结果。图片源仍采用逐帧同步识别。终端默认
只在新结果到达时打印 `RESULT`，内容为第一候选标签/分数和第二候选标签/分数；
完整路由、状态与耗时仍保存在 `events.jsonl`。`--verbose` 可恢复逐帧性能日志。

离线视频默认按源视频 FPS 播放，因此资源压力接近实时相机。只有需要尽快批处理时
才添加 `--unthrottled-video`。比赛部署应显式选择 `competition`，由 profile 固定关闭
标注媒体、裁剪和调试输出：

```bash
./start.sh --profile competition --source ../recordings/video.mp4
```

正式 C++ 运行时不支持 `.pt` 权重，也不会加载 Python 或 PyTorch。RK3588 默认
使用四个 FP16 RKNN 模型；需要诊断精度差异时可以强制回退到 ONNX CPU：

```bash
./start.sh \
  --weights models/yolo/best.onnx \
  --provider cpu --yolo-cpu-threads 2
```

完整测试数据和选择建议见
[`training/temporal/evaluation/resource_optimization_20260928.md`](training/temporal/evaluation/resource_optimization_20260928.md)。

## 项目结构

仓库同时包含“正式运行时”和“开发/回归工具”两套代码。部署和现场运行以
`start.sh` 启动的 C++17 程序为准；`src/insight_cup/` 下的 Python 实现保留用于
迁移对照、数据准备、训练和离线评估，不会被正式 C++ 进程加载。排查线上行为时，
应先沿 `cpp/` 主链路定位，不能仅根据同名 Python 模块推断当前行为。

### 目录总览

```text
insight_cup/
|-- CMakeLists.txt                 C++17 构建、平台依赖和可选后端
|-- build_cpp.sh                   C++ 配置与增量构建入口
|-- start.sh                       正式 C++ 运行入口
|-- start_python.sh                仅用于迁移回归的 Python 入口
|-- pyproject.toml                 Python 包、依赖和命令行入口定义
|-- cpp/
|   |-- include/insight_cup/       C++ 模块接口与数据契约
|   |-- src/                       C++ 模块实现和 main.cpp
|   |-- SOURCE_MANIFEST.json       C++ 等价移植的来源清单
|   `-- UPSTREAM.md                C++ 上游版本与移植范围
|-- src/insight_cup/
|   |-- app/                       Python 参考运行时
|   |   `-- ui/                    C++/Python 共用的静态调试界面
|   |-- common/                    crop 与 ONNX 公共辅助代码
|   |-- face/                      Python 人脸识别和图库实现
|   |-- knife/                     Python 刀具分类/检索实现
|   |-- vendor/                    精简的 InsightFace/PaddleClas 上游代码
|   `-- paths.py                   仓库标准路径定义
|-- tools/
|   |-- inference/                 独立阶段推理和 YOLO 演示
|   |-- data/                      参考图、数据集和图库构建
|   |-- evaluation/                人脸、刀具和 shadow/golden 评估
|   |-- benchmark/                 UI/no-UI 资源基准与报告
|   |-- rknn/                      ONNX -> RKNN 转换及原生 smoke test
|   |-- network/                   RK3588 网络共享部署工具
|   `-- systemd/                   RK3588 USB/桌面稳定性系统配置
|-- models/
|   |-- yolo/                      YOLO ONNX/PT 模型
|   |-- face/                      SCRFD 与 ArcFace ONNX 模型
|   |-- knife/                     刀具分类与特征检索模型
|   `-- rknn/rk3588_fp16/          RK3588 FP16 部署模型
|-- data/
|   |-- raw/                       原始抽帧样本及采集元数据
|   |-- face_gallery/              人脸参考图、NPZ 和 C++ ICG1 图库
|   |-- knife_dataset/             分组切分后的刀具分类数据集
|   `-- knife_gallery/             刀具检索 NPZ 和 C++ ICG1 图库
|-- training/
|   |-- knife/                     PaddleClas 配置、checkpoint 和指标
|   `-- temporal/evaluation/       时序策略与资源优化报告
|-- tests/                         Python 单元/回归测试和小型视频 fixture
|-- docs/                          架构现状与 C++ 迁移记录
|-- third_party/                   固定版本 C++ SDK、包和许可证
|-- requirements/                  Python 分场景依赖清单
|-- outputs/                       运行会话、评估和 benchmark 生成物
|-- build/                         CMake 生成目录和 insight-cup 可执行文件
`-- .venv/                         本地 Python 虚拟环境
```

`build/`、`.venv/`、`outputs/`、`__pycache__/` 和 `.pytest_cache/` 都是生成目录，
不是源码。`outputs/` 默认被 Git 忽略，可按会话清理；清理前应确认其中的日志、录像和
评估报告是否仍需复盘。根目录或模型目录中临时生成的转换诊断文件也不属于正式资产，
正式路径以配置默认值、模型 manifest 和下文“当前模型”为准。

### 入口、构建与依赖

| 路径 | 职责 | 重要边界 |
| --- | --- | --- |
| `start.sh` | 正式启动入口；检查桌面安全配置、按需增量构建，并选择比赛或 UI 录像线路 | 默认 `competition` 使用正常优先级；`ui-recording` 使用较低优先级并开启 UI 与标注录像；旧手工开关进入 `custom` 兼容线路 |
| `build_cpp.sh` | 创建/更新 `build/` 并执行 CMake 构建 | 默认 `Release`、2 个并行任务；可用环境变量覆盖构建类型和并行数 |
| `CMakeLists.txt` | 生成唯一的 C++17 可执行文件 `insight-cup` | 选择 x86_64/aarch64 本地依赖；RealSense 可选；RKNN 只在 ARM64 且 SDK 可用时编译 |
| `start_python.sh` | 启动旧 Python 全链路用于迁移结果对照 | 自动选择 `.venv`、限制线程；不是生产入口 |
| `pyproject.toml` | 定义 `insight-cup` Python 包、依赖和 console script | Python 入口为 `insight_cup.app.cli:main`，也可通过 `python -m insight_cup` 调用 |
| `requirements/runtime.txt` | Python 参考运行时依赖 | 包含 YOLO、ONNX Runtime、OpenCV 等，不影响已编译的 C++ 进程 |
| `requirements/face.txt`、`requirements/knife.txt` | 人脸/刀具模块的分场景依赖清单 | 按相应 Python 工作流选用；完整 Python 包依赖仍以 `pyproject.toml` 为准 |
| `third_party/` | 固定 ONNX Runtime 1.23.2、cpp-httplib、ARM64 离线包及许可证 | 属于构建输入；不要直接修改 SDK 头文件或共享库，升级时同步更新 CMake 和许可证说明 |

### C++ 正式运行时

接口位于 `cpp/include/insight_cup/`，对应实现位于 `cpp/src/`。除 `main.cpp` 外，
主要模块均按同名 `.hpp/.cpp` 配对：

| 模块 | 职责 |
| --- | --- |
| `main.cpp` | 组装配置、日志、共享状态、运行线程和 Web 服务；处理信号与进程退出 |
| `config.hpp/.cpp` | 保存全部默认参数，解析 CLI，展开路径，执行参数校验，并把有效配置序列化到 manifest |
| `types.hpp/.cpp` | 定义框、YOLO 调试信息、阶段一交接、阶段二原始决策、时序结果和最终结果等跨模块契约 |
| `inference.hpp` | 定义统一的 `InferenceSession` 接口，让模型层不直接依赖具体 CPU/NPU 后端 |
| `onnx.hpp/.cpp` | ONNX Runtime CPU session、张量输入输出和模型 metadata 读取 |
| `rknn.hpp/.cpp` | RKNN Runtime NPU session、NPU core 绑定和张量转换；仅在支持的平台参与构建 |
| `gallery.hpp/.cpp` | 读取 C++ `ICG1` 二进制特征图库，按身份聚合余弦分数并返回前两名和 margin |
| `models.hpp/.cpp` | YOLO 前后处理、`fN/kN -> f/k` 合并、SCRFD、ArcFace、刀具分类/检索、crop 和路由实现 |
| `scheduler.hpp/.cpp` | 目标轨迹关联、异步 worker、结果缓存、刷新周期以及 vote/mean/none 时序平滑 |
| `runtime.hpp/.cpp` | 加载模型，打开 RealSense/相机/视频/图片源，逐帧编排，画框，保存媒体并写会话清单 |
| `state.hpp/.cpp` | 维护线程安全的运行状态、计数器、最近事件、最新 JPEG，以及 pause/resume/stop 控制 |
| `logging.hpp/.cpp` | 控制台日志、JSON 文件写入和 `events.jsonl` 事件流 |
| `web.hpp/.cpp` | 提供静态 UI、状态/事件/单帧/MJPEG API、健康检查和控制接口 |
| `SOURCE_MANIFEST.json`、`UPSTREAM.md` | 记录从 InsightFace/PaddleClas 等上游等价移植的文件、版本和范围 |

正式调用链如下：

```text
start.sh
  -> main.cpp
     -> RuntimeConfig + RuntimeState + EventJournal + [--ui] DebugServer
     -> RecognitionRuntime
        -> 输入源：RealSense / OpenCV 相机或视频 / 图片或图片目录
        -> YoloDetector：raw f1...f10 / k1...k10 / b0 -> major f / k / b0
        -> AsyncStage2Scheduler：轨迹、缓存、刷新、异步 worker、时序确认
           -> b0 -> RecognitionRouter -> passthrough
           -> f  -> FaceRecognizer -> SCRFD -> ArcFace -> face ICG1 gallery
           -> k  -> KnifeRecognizer -> PPLCNetV2 classification
                                  或 -> PP-ShiTuV2 embedding -> knife ICG1 gallery
        -> RuntimeState + EventJournal + 标注媒体/crop
           -> [--ui] DebugServer -> 静态 UI + JSON/MJPEG/control API
```

这里有两个容易混淆的输出边界：`b0` 不进入二阶段，直接得到 `passthrough`；`f/k`
只有在阶段二结果最终为 `matched` 时才会作为正式识别画框。候选类别、`unknown`、
`no_face`、`confirming` 或 `processing` 仍可能出现在终端和诊断字段中，但不会被 UI
当作已接受目标画框。事件结构和画框逻辑因此分别落在 `types`、`scheduler`、
`runtime`、`logging` 四个模块中，而不是只由前端决定。

### Python 参考实现

`src/insight_cup/` 保留 C++ 迁移前的等价链路和上游 Python 适配层。它主要用于单元
测试、离线工具和 C++/Python 行为对照：

| 路径 | 职责 |
| --- | --- |
| `src/insight_cup/__main__.py` | 把 `python -m insight_cup` 转交给 Python CLI |
| `src/insight_cup/paths.py` | 统一解析项目根目录及 `models/data/outputs` 标准路径 |
| `app/cli.py`、`app/config.py` | Python 命令行、默认值、参数校验和运行时组装 |
| `app/contracts.py` | Python 版阶段一/阶段二 dataclass 契约 |
| `app/detector.py`、`app/router.py` | YOLO 大类检测、重复框抑制和 `b0/f/k` 路由 |
| `app/stage2.py` | Python 版轨迹关联、异步调度、缓存与时序平滑 |
| `app/runtime.py`、`app/sources.py` | Python 帧循环、RealSense 输入、OpenCV 输入、画框与会话输出 |
| `app/state.py`、`app/web.py`、`app/logging_utils.py` | 共享状态、HTTP/MJPEG 服务和事件日志 |
| `app/ui/` | `index.html`、`app.css`、`app.js`；C++ Web 服务也直接复用这些静态资源 |
| `common/crops.py`、`common/onnx.py` | 刀具正方形上下文 crop 和 ONNX CPU session 公共策略 |
| `face/` | InsightFace 引擎、参考图库、身份匹配、服务封装和文件式 handoff 处理 |
| `knife/` | PaddleClas 分类器、PP-ShiTuV2 特征检索、服务封装和文件式 handoff 处理 |
| `vendor/insightface_core/` | 精简的 SCRFD、ArcFace、五点对齐和必要加载代码，附来源/许可证 |
| `vendor/paddleclas_core/` | 精简的分类预处理、Top-k、特征归一化和内积检索，附来源/许可证 |

Python 的默认模型/图库格式与正式 C++ 路径并不完全相同，例如 Python 工具使用
`.npz` 图库，而 C++ 使用 `.icg`；修改时必须明确是在做迁移回归，还是在改变正式
部署行为。

### 工具目录

| 目录 | 主要文件 | 输入与输出 |
| --- | --- | --- |
| `tools/inference/` | `yolo_detect.py`、`stage1_yolo.py`、`stage2_face.py`、`stage2_knife.py`、`yolo_web_demo.py` | 独立运行 YOLO，生成/消费文件式 handoff，或启动小型图片演示；输出写入指定测试目录 |
| `tools/data/` | `export_face_references.py`、`prepare_face_gallery.py`、`export_knife_dataset.py`、`build_knife_retrieval_gallery.py`、`install_face_models.py` | 从标注数据导出人脸参考图和分组刀具集，构建 Python NPZ 图库，安装所需人脸模型 |
| `tools/evaluation/` | `evaluate_face.py`、`evaluate_knife.py`、`evaluate_shadow.py` | 评估人脸图库、刀具分类器，以及同源 `shadow.jsonl` 对 golden 标注的端到端结果 |
| `tools/benchmark/` | `run_offline_ui_benchmark.py`、`analyze_offline_ui_benchmark.py` | 成组运行离线视频的 UI/no-UI 对照，采集进程/系统资源并生成 JSON、CSV、Markdown 报告 |
| `tools/rknn/` | `convert_models.py`、`rknn_smoke.cpp`、`README.md` | 将四个 ONNX 模型转为 RK3588 FP16 RKNN，并用原生 Runtime 做最小加载/推理检查 |
| `tools/network/` | 网络共享脚本、desktop 文件和 systemd service | 配置开发主机与 RK3588 之间的网络共享；属于部署辅助，不参与识别进程 |
| `tools/systemd/` | USB host、NoMachine、Xorg/GDM 稳定性脚本和 unit | 修改目标机系统服务及桌面配置；可能重启远程桌面，只应在 RK3588 上审核后以 root 执行 |

### 模型、数据与训练资产

| 路径 | 内容 | 消费方 |
| --- | --- | --- |
| `models/yolo/` | `best.onnx` 为 C++ CPU 基线；`best.pt` 供 Python Ultralytics 路径和相关工具使用 | 一级 YOLO 检测与工具 |
| `models/face/models/face_only_r50/` | `det_10g.onnx`、`w600k_r50.onnx` 和模型 manifest | SCRFD 人脸检测与 ArcFace 512 维特征 |
| `models/knife/pplcnetv2_base_knife10/` | 10 类分类 ONNX、Paddle 导出物、labels、模型说明和导出验证 | 默认刀具分类二阶段 |
| `models/knife/ppshitu_v2_retrieval/` | 512 维特征模型、Paddle 导出物和 metadata | 实验性刀具检索二阶段 |
| `models/rknn/rk3588_fp16/` | 由上述 ONNX 转换的 YOLO、SCRFD、ArcFace、刀具分类 RKNN | RK3588 `--provider auto/rknn` 正式 NPU 路径 |
| `data/raw/` | 原始/抽帧图片及每张图片的采集元数据 | 数据筛选和后续数据集构建；默认运行链路不会主动加载 |
| `data/face_gallery/references_trainv5/` | 按 `f1...f10` 组织的人脸参考图片和 manifest | 人脸图库重建 |
| `data/face_gallery/*.npz` | Python 使用的标签、特征和参考信息 | Python 回归、工具与评估 |
| `data/face_gallery/cpp/*.icg` | `ICG1` 格式的部署图库及 manifest | C++ `FeatureGallery` |
| `data/knife_dataset/trainv5_grouped/` | 按录制组划分的 train/val/test 图片、列表、labels 和 manifest | PaddleClas 训练、检索图库构建与离线评估 |
| `data/knife_gallery/*.npz` | Python PP-ShiTuV2 特征图库和 JSON metadata | Python 检索回归与评估 |
| `data/knife_gallery/cpp/*.icg` | `ICG1` 格式的部署图库及 manifest | C++ 刀具 retrieval 模式 |
| `training/knife/` | PaddleClas YAML、训练 checkpoint、val/test 指标 | 训练复现和模型选择；运行时不直接加载 checkpoint |
| `training/temporal/evaluation/` | vote/mean/none A/B、全视频结果和资源优化结论 | 时序参数与部署资源决策依据 |

资产关系可以概括为：

```text
原始/标注数据
  -> tools/data 导出参考图或 train/val/test 数据集
     -> Python 模型提取特征 -> .npz 图库 -> Python 回归/评估
                              -> .icg 部署快照 -> C++ 正式运行时

训练 checkpoint / 上游模型
  -> 导出 ONNX -> C++ CPU 基线
               -> tools/rknn/convert_models.py -> RK3588 FP16 RKNN
```

模型二进制和图库不是手写源码。替换参考图、数据集或模型时，应同时更新对应的
manifest/metadata、标签、阈值、部署格式并重跑评估，不能只覆盖单个 `.onnx`、
`.rknn`、`.npz` 或 `.icg` 文件。人脸参考图和特征属于生物特征数据；共享仓库、
日志或会话产物前需要确认授权和脱敏要求。

当前仓库保存了训练配置和部署成品，但没有纳入完整的 PaddleClas 训练引擎、YOLO
训练配置，也没有提供 `.pt -> .onnx`、Paddle inference -> ONNX 或 `.npz -> .icg`
转换命令。刀具 YAML 需要放到固定的外部 PaddleClas `release/2.6` 环境执行；
`tools/rknn/convert_models.py` 只负责已有 ONNX 到 RKNN 的部署转换。因而升级模型或
C++ 图库时，还必须记录所用外部导出步骤和版本，不能把已有二进制成品当成完整的
可复现训练流水线。

### 测试、文档与运行产物

| 路径 | 职责 |
| --- | --- |
| `tests/test_runtime_core.py` | Python 核心契约、路由、调度、时序和运行时行为测试 |
| `tests/test_face_id.py` | 人脸图库、身份识别和相关拒识行为测试 |
| `tests/test_knife_id.py` | 刀具分类/检索、crop 和阈值行为测试 |
| `tests/test_shadow_evaluator.py` | shadow/golden 输入校验、对齐和指标计算测试 |
| `tests/fixtures/test_video.mp4` | 小型可重复视频 fixture |
| `docs/ARCHITECTURE.md` | 当前正式数据链路、模块边界、事件 schema 和会话产物的权威设计说明 |
| `docs/CPP_MIGRATION.md` | Python 到 C++ 的迁移范围、上游版本和历史一致性验证记录 |
| `outputs/runtime/<session-id>/` | 单次运行的 manifest、events、summary、可选 shadow、录像、标注图片和 crop |
| `outputs/benchmarks/<name>/` | benchmark 参数、应用日志、资源采样、分析报告和关联运行会话 |
| `build/` | CMakeCache、对象文件、CTest 信息和最终 `build/insight-cup`；可重新生成 |
| `.venv/` | Python 解释器和依赖的本地安装；可根据 `pyproject.toml/requirements` 重建 |

C++ 当前由 CTest 提供 `insight-cup --help` 构建烟雾测试；算法和契约的细粒度回归
仍主要位于 Python `tests/`。完整的手动验证命令见本文末尾“测试”章节。

### 常见修改入口

| 目标 | 首先查看 | 通常还需同步 |
| --- | --- | --- |
| 改正式 CLI 参数或默认值 | `cpp/include/insight_cup/config.hpp`、`cpp/src/config.cpp` | `start.sh` 的生产覆盖值、manifest 和 README |
| 改输入源、逐帧处理、画框或媒体保存 | `cpp/src/runtime.cpp` | `types`、`state`、`logging`、`docs/ARCHITECTURE.md` |
| 改 YOLO/人脸/刀具前后处理 | `cpp/src/models.cpp` | Python 对照实现、阈值配置、评估和模型 metadata |
| 改跟踪、缓存、刷新或时序确认 | `cpp/src/scheduler.cpp` | `config` 默认值、`training/temporal/evaluation/` 和运行时测试 |
| 改事件字段或阶段间契约 | `cpp/include/insight_cup/types.hpp`、`cpp/src/types.cpp` | `runtime`、`logging`、shadow evaluator、UI 和架构文档 |
| 改 HTTP API 或调试界面 | `cpp/src/web.cpp`、`src/insight_cup/app/ui/` | `RuntimeState` 和 Python `app/web.py` 对照实现 |
| 换模型或 RKNN 部署文件 | `models/`、`tools/rknn/convert_models.py` | 配置路径、labels、manifest、CPU/NPU 同源评估 |
| 增删人脸参考或刀具训练数据 | `data/`、`tools/data/` | NPZ/ICG 图库、manifest、训练/评估报告 |
| 改 Python 迁移参考行为 | `src/insight_cup/` | 相应 `tests/`；确认是否也需要同步 C++ 正式链路 |
| 升级 C++ 第三方依赖 | `CMakeLists.txt`、`third_party/` | `third_party/README.md`、许可证和目标架构构建验证 |

## 当前模型

- YOLO：`models/rknn/rk3588_fp16/yolo_best.rknn`
- 人脸检测：`models/rknn/rk3588_fp16/scrfd_det_10g.rknn`
- 人脸特征：`models/rknn/rk3588_fp16/arcface_w600k_r50.rknn`
- 人脸库：`data/face_gallery/cpp/trainv5_gallery_r50.icg`
- 刀具分类：`models/rknn/rk3588_fp16/knife_pplcnetv2.rknn`
- 刀具特征：`models/knife/ppshitu_v2_retrieval/inference.onnx`
- 刀具特征库：`data/knife_gallery/cpp/trainv5_ppshitu_v2.icg`

默认阈值：

- 人脸：相似度 `0.40`，第一/第二候选 margin `0.03`
- 刀具分类：置信度 `0.50`，第一/第二候选 margin `0.15`
- 刀具检索：余弦相似度 `0.50`，不同标签候选 margin `0.02`

刀具默认使用 NPU 分类器。PaddleClas PP-ShiTuV2 特征检索尚未转换为 RKNN；使用
`auto` 时只有该检索模型回退到 CPU，也可显式让整条链路使用 CPU：

```bash
./start.sh --provider cpu --knife-mode retrieval
```

两种模式应使用同一视频、不同 `--session-id` 分别运行后比较。检索图库由 2296 张
训练裁剪建立，离线 Top-1 为 val `54.22%`、test `76.77%`；当前分类器对应为
val `70.14%`、test `80.67%`，因此检索模式目前是对比实验，不是默认替代。

## 会话输出

默认写入 `outputs/runtime/<session-id>/`：

- `manifest.json`：本次参数、模型信息和明确的数据传递路径
- `events.jsonl`：每个目标的 stage1、route、stage2 和耗时
- `shadow.jsonl`：显式开启同源评估日志时，记录原始 YOLO/二阶段/时序结果
- `summary.json`：最终统计与退出状态
- `annotated.mp4` 或 `annotated_images/`：可选标注媒体
- `crops/f/`、`crops/k/`：可选的、实际送入二阶段的裁剪

终端事件示例：

```text
RESULT frame=42 id=frame000042:det001 label=k4 probability=0.810 second=k2 second_probability=0.180
```

默认会在 YOLO 的 `fN/kN` 合并为大类后再次执行 NMS，并按整帧目标位置和运动
方向关联轨迹。人脸或刀具身份需连续两次确认后才显示，短时异常不会立即切换标签。

### 同源 shadow 与 golden 评估

算法 A/B 时显式开启 shadow 日志：

```bash
./start.sh --profile custom --source ../recordings/video.mp4 \
  --shadow-log --session-id baseline-shadow
```

`shadow.jsonl` 只写实际完成二阶段推理的 `f/k` 记录，不重复写每帧缓存结果或 `b0`。
每条记录把真正送入二阶段的 source ID、帧号、框、YOLO 21 类分数、原始二阶段结果
和时序后结果放在一起。正常的 `events.jsonl` 契约不变。shadow 默认关闭；开启后会
增加少量 CPU、内存和日志写入，只应用于评估会话。固定文件还会在启动时用 64 KiB
缓冲顺序读取一遍以计算 SHA-256，不会把整段媒体载入内存；shadow 关闭时不做该读取。

评估还会核对输入来源。固定文件使用内容 SHA-256；实时采集使用一次采集会话内唯一的
`capture_id`。两边的 `kind`、内容身份和 `video_stride` 必须完全一致。固定视频才能
跨算法 A/B 复用；实时源需要先录制成文件再做可重复 A/B。`D455`、设备序列号或路径
都不是内容身份，不能代替 SHA-256 或采集会话 ID。`capture` 只适合评估同一次采集：

```json
{"kind": "capture", "capture_id": "capture-20261003-001", "video_stride": 1}
```

人工标注文件使用 source ID 对齐。`expected_class: null` 表示尚未标注，会排除在所有
真值指标之外；字符串 `"unknown"` 表示人工明确判定为陌生人、其他刀具或背景，会进入
unknown 拒识指标。其余合法标签固定为 `b0`、`f1...f10`、`k1...k10`：

```json
{
  "schema": "recognition_golden.v1",
  "input": {
    "kind": "file",
    "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "video_stride": 1
  },
  "items": [
    {"id": "frame000000:det000", "expected_class": "f3", "bbox_xyxy": [1, 2, 30, 40]},
    {"id": "frame000000:det006", "expected_class": "unknown"},
    {"id": "frame000000:det009", "expected_class": null}
  ]
}
```

这里的 `input` 必须原样复制对应 shadow 记录的顶层 `input`；示例 SHA-256 仅展示格式，
不能用于实际评估。

生成评估报告：

```bash
python3 tools/evaluation/evaluate_shadow.py \
  --shadow outputs/runtime/baseline-shadow/shadow.jsonl \
  --golden outputs/golden.json \
  --output outputs/runtime/baseline-shadow/evaluation.json
```

报告包含 ID 对齐率、缺失/无标签记录、最终覆盖率与已赋类准确率、YOLO/原始二阶段/
时序结果准确率、unknown 拒识、阶段一致性、框 IoU 和细类/大类混淆。预测类别与
source 大类不一致、真值与 source 大类冲突、`error` 或状态/最终类别矛盾的记录会单列
为 invalid，避免被总体准确率掩盖。最终覆盖率与运行时一致，只接受状态为 `matched`
或 `passthrough` 且存在最终类别的结果；`error` 不算 unknown 的正确拒识。该报告只
评估已经进入 shadow 的保留框；YOLO 完全漏掉的目标仍需使用带人工框的独立检测集衡量。
输入指纹只证明媒体内容相同。若 A/B 改动 YOLO 或 NMS，`detNN` 可能换序；golden 应
保留 `bbox_xyxy` 并检查 IoU，本工具当前不会自动把不同 ID 的检测框重新匹配。

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
python3 tools/inference/yolo_detect.py --source ../recordings/video.mp4
```

生成文件式阶段一交接：

```bash
python3 tools/inference/stage1_yolo.py \
  --source ../recordings/video.mp4 \
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
  --data ../trainv5/data.yaml \
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
  --data ../trainv5/data.yaml \
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
./start.sh --profile competition --source ../recordings/video.mp4 \
  --max-frames 120 --unthrottled-video

# 旧 Python 回归测试
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

C++ 与 Python 同帧回归结果记录在
[`docs/CPP_MIGRATION.md`](docs/CPP_MIGRATION.md)。
