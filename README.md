# Insight Cup

## 构建编译

在项目目录执行：

```bash
cd /home/j/insight_cup
source /opt/ros/humble/setup.bash
source /home/j/Public-Safety-Event-/install/setup.bash

./build_cpp.sh
```

正式识别使用 Intel RealSense D455 的 `1280x720@15 FPS` 彩色画面。

## 正式 UI 启动

本机测试需要三个终端。终端 A 启动假停车/继续服务，终端 B 启动正式 UI 识别程序，终端 C 查看服务和话题。

### 终端 A：启动停车/继续假服务

```bash
cd /home/j/Public-Safety-Event-
source /opt/ros/humble/setup.bash
source /home/j/Public-Safety-Event-/install/setup.bash

python3 /home/j/insight_cup/tools/ros/fake_motion_hold_server.py --json-log
```

看到下面的输出，说明服务已启动：

```text
FAKE motion hold server ready
```

### 终端 B：启动正式 UI 识别程序

```bash
cd /home/j/insight_cup
source /opt/ros/humble/setup.bash
source /home/j/Public-Safety-Event-/install/setup.bash

./build_cpp.sh

./start.sh \
  --profile ui-recording \
  --source realsense \
  --provider auto \
  --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

不自动打开浏览器时，在启动命令末尾添加：

```text
--no-open-browser
```

### 终端 C：查看服务和话题

先加载 ROS 环境：

```bash
source /opt/ros/humble/setup.bash
source /home/j/Public-Safety-Event-/install/setup.bash
```

查看停车/继续服务是否存在：

```bash
ros2 service list -t | grep set_motion_hold
```

检查三个话题是否存在：

```bash
ros2 topic list | grep -E '^/(safe|vision|insight)$'
```

查看完整话题消息：

```bash
ros2 topic echo /safe
ros2 topic echo /vision
ros2 topic echo /insight
```

查看 JSON 话题时，使用下面的命令显示最清晰：

```bash
ros2 topic echo /safe --field data | jq -C -R --unbuffered 'fromjson? // .'
```

```bash
ros2 topic echo /insight --field data | jq -C -R --unbuffered 'fromjson? // .'
```

`/vision` 是人物和刀具识别的内部调试话题，内容例如 `f5,k3`。

`/safe` 是障碍物调试话题，发布完整安全状态 JSON。

`/insight` 是最终综合话题，内容包括：

```json
{
  "timestamp_ms": 1234567890,
  "obstacle_detected": true,
  "obstacle_distance_cm": 40,
  "target_detected": true,
  "targets": ["f5", "k3"]
}
```
