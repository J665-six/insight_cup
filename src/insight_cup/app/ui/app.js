"use strict";

const elements = {
  source: document.getElementById("sourceText"),
  runState: document.getElementById("runState"),
  runStateText: document.getElementById("runStateText"),
  resume: document.getElementById("resumeButton"),
  pause: document.getElementById("pauseButton"),
  stop: document.getElementById("stopButton"),
  processed: document.getElementById("processedFrames"),
  fps: document.getElementById("fpsValue"),
  latency: document.getElementById("latencyValue"),
  b0: document.getElementById("b0Count"),
  f: document.getElementById("fCount"),
  k: document.getElementById("kCount"),
  frame: document.getElementById("liveFrame"),
  placeholder: document.getElementById("videoPlaceholder"),
  placeholderText: document.getElementById("placeholderText"),
  frameLabel: document.getElementById("frameLabel"),
  frameDetections: document.getElementById("frameDetectionCount"),
  yoloTiming: document.getElementById("yoloTiming"),
  stage2Timing: document.getElementById("stage2Timing"),
  session: document.getElementById("sessionId"),
  lastUpdate: document.getElementById("lastUpdate"),
  matched: document.getElementById("matchedCount"),
  unknown: document.getElementById("unknownCount"),
  ambiguous: document.getElementById("ambiguousCount"),
  noFace: document.getElementById("noFaceCount"),
  eventCount: document.getElementById("eventCount"),
  eventRows: document.getElementById("eventRows"),
  error: document.getElementById("errorMessage"),
};

const statusLabels = {
  starting: "启动中",
  loading: "加载模型",
  running: "运行中",
  paused: "已暂停",
  stopping: "停止中",
  stopped: "已停止",
  completed: "已完成",
  error: "运行错误",
};

const moduleLabels = {
  waiting: "等待",
  loading: "加载中",
  ready: "就绪",
  error: "错误",
};

const routeLabels = {
  passthrough: "直接输出",
  face: "人脸识别",
  knife: "刀具识别",
};

let hasFrame = false;
let streamRetryTimer = null;
let runtimeStatus = "starting";

function numberValue(value, digits = 1) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "-";
}

function setText(element, value) {
  element.textContent = value === null || value === undefined ? "-" : String(value);
}

function updateModule(name, payload) {
  const row = document.getElementById(`module${name}`);
  const status = payload && payload.status ? payload.status : "waiting";
  row.dataset.status = status;
  setText(row.querySelector(".module-status"), moduleLabels[status] || status);
  setText(row.querySelector(".module-detail"), payload && payload.detail ? payload.detail : "-");
}

function updateState(state) {
  const status = state.status || "starting";
  const sourceInfo = state.source_info || {};
  const sourceLabel = sourceInfo.type === "realsense"
    ? `${sourceInfo.model} · ${sourceInfo.serial} · ${sourceInfo.width}x${sourceInfo.height}@${sourceInfo.fps}`
    : (state.source || "-");
  runtimeStatus = status;
  elements.runState.dataset.state = status;
  setText(elements.runStateText, statusLabels[status] || status);
  setText(elements.source, sourceLabel);
  elements.source.title = sourceLabel;
  setText(elements.processed, state.processed_frames || 0);
  setText(elements.fps, numberValue(state.fps, 2));
  setText(elements.latency, numberValue(state.last_timing_ms?.total, 1));
  setText(elements.b0, state.major_counts?.b0 || 0);
  setText(elements.f, state.major_counts?.f || 0);
  setText(elements.k, state.major_counts?.k || 0);
  setText(elements.frameLabel, state.frame_index === null ? "帧 -" : `帧 ${state.frame_index}`);
  setText(elements.frameDetections, state.detections_in_frame || 0);
  setText(elements.yoloTiming, `${numberValue(state.last_timing_ms?.yolo, 1)} ms`);
  setText(elements.stage2Timing, `${numberValue(state.last_timing_ms?.stage2, 1)} ms`);
  setText(elements.session, state.session_id || "-");
  elements.session.title = state.session_dir || "";
  setText(elements.matched, state.status_counts?.matched || 0);
  setText(elements.unknown, state.status_counts?.unknown || 0);
  setText(elements.ambiguous, state.status_counts?.ambiguous || 0);
  setText(elements.noFace, state.status_counts?.no_face || 0);
  setText(elements.lastUpdate, new Date().toLocaleTimeString("zh-CN", { hour12: false }));
  updateModule("Yolo", state.modules?.yolo);
  updateModule("Face", state.modules?.face);
  updateModule("Knife", state.modules?.knife);

  elements.pause.disabled = status !== "running";
  elements.resume.disabled = status !== "paused";
  elements.stop.disabled = ["completed", "stopped", "error"].includes(status);
  elements.error.textContent = state.error || "";

  hasFrame = Boolean(state.has_frame);
  if (!hasFrame) {
    elements.placeholder.style.display = "flex";
    elements.placeholderText.textContent = status === "error" ? "运行失败" : "模型初始化中";
  } else {
    elements.placeholder.style.display = "none";
  }
}

function addCell(row, value, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  cell.textContent = value;
  cell.title = value;
  row.appendChild(cell);
  return cell;
}

function eventTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleTimeString("zh-CN", { hour12: false });
}

function renderEvents(events) {
  elements.eventRows.replaceChildren();
  setText(elements.eventCount, `${events.length} 条`);
  if (!events.length) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = document.createElement("td");
    cell.colSpan = 8;
    cell.textContent = "暂无识别事件";
    row.appendChild(cell);
    elements.eventRows.appendChild(row);
    return;
  }

  for (const event of events) {
    const row = document.createElement("tr");
    addCell(row, eventTime(event.timestamp), "mono");
    addCell(row, event.id || "-", "mono");

    const classCell = document.createElement("td");
    const classTag = document.createElement("span");
    const major = event.stage1?.major_class || "-";
    classTag.className = `class-tag ${major}`;
    classTag.textContent = major;
    classCell.appendChild(classTag);
    row.appendChild(classCell);

    addCell(row, routeLabels[event.route?.module] || event.route?.module || "-");

    const status = event.stage2?.status || "-";
    const predicted = event.stage2?.predicted_class;
    const candidate = event.stage2?.candidate_class;
    const resultCell = document.createElement("td");
    const resultTag = document.createElement("span");
    resultTag.className = `result-tag ${status}`;
    resultTag.textContent = predicted || status;
    resultCell.appendChild(resultTag);
    if (!predicted && candidate && candidate !== major) {
      resultCell.appendChild(document.createTextNode(`  候选 ${candidate}`));
    }
    row.appendChild(resultCell);

    addCell(row, numberValue(event.stage2?.score, 3), "mono");
    addCell(row, numberValue(event.stage2?.margin, 3), "mono");
    addCell(row, `${numberValue(event.timing_ms?.stage2, 1)} ms`, "mono");
    elements.eventRows.appendChild(row);
  }
}

async function requestJson(url, options) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || payload.message || `HTTP ${response.status}`);
  return payload;
}

async function pollState() {
  try {
    const state = await requestJson("/api/state");
    updateState(state);
  } catch (error) {
    elements.runState.dataset.state = "error";
    elements.runStateText.textContent = "连接中断";
    elements.error.textContent = error.message;
  } finally {
    window.setTimeout(pollState, 600);
  }
}

async function pollEvents() {
  try {
    const payload = await requestJson("/api/events?limit=40");
    renderEvents(payload.events || []);
  } catch (error) {
    elements.error.textContent = error.message;
  } finally {
    window.setTimeout(pollEvents, 900);
  }
}

function connectFrameStream() {
  window.clearTimeout(streamRetryTimer);
  if (document.hidden) return;
  elements.frame.src = `/api/stream.mjpg?v=${Date.now()}`;
}

elements.frame.addEventListener("load", () => {
  elements.placeholder.style.display = "none";
});

elements.frame.addEventListener("error", () => {
  const finalState = ["completed", "stopped", "error"].includes(runtimeStatus);
  if (finalState) {
    elements.frame.src = `/api/frame.jpg?v=${Date.now()}`;
    return;
  }
  streamRetryTimer = window.setTimeout(connectFrameStream, 750);
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    window.clearTimeout(streamRetryTimer);
    elements.frame.removeAttribute("src");
  } else {
    connectFrameStream();
  }
});

async function control(action) {
  try {
    const payload = await requestJson("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    updateState(payload.state);
  } catch (error) {
    elements.error.textContent = error.message;
  }
}

elements.pause.addEventListener("click", () => control("pause"));
elements.resume.addEventListener("click", () => control("resume"));
elements.stop.addEventListener("click", () => control("stop"));

pollState();
pollEvents();
connectFrameStream();
