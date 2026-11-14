#include "web_server.h"
#include "ai_inference.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <sys/param.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"

static const char* TAG = "WEB_SERVER";

#define MAX_STREAM_BUFFER_SIZE (64 * 1024)
#define PART_BOUNDARY "123456789000000000000987654321"

static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// Vùng đệm khung hình JPEG trong Octal PSRAM phục vụ streaming
static uint8_t* s_stream_frame = NULL;
static size_t   s_stream_frame_len = 0;
static SemaphoreHandle_t s_frame_mutex = NULL;

// Dữ liệu telemetry & 22 landmarks mới nhất
typedef struct {
    float ear;
    float ear_left;
    float ear_right;
    float mar;
    float yaw;
    float pitch;
    float roll;
    bool  pose_valid;
    char  status_str[32];
    bool  is_alarm_active;
    adas_eye_diag_t eye;
    float fps;
    float dec_ms;
    float ai_ms;
    float pnp_ms;
    float total_ms;
    float ear_threshold;
    float mar_threshold;
    float mouth_open_s;
    float eye_prob;
    uint32_t total_yawns;
    uint32_t total_blinks;
    int roi_active;
    int roi_x0;
    int roi_y0;
    int roi_size;
    int src_w;             // [P3] kich thuoc anh goc (de map toa do)
    int src_h;             // [P3]
    uint32_t frame_id;     // [P1]
    float img_age_ms;      // [P1]
    float roi_age_ms;      // [P1]
    float core1_ms;        // [P1]
    // [COMPARE] metadata chan doan cung frame
    int   roi_source;
    bool  face_valid;
    float face_score;
    float face_cx, face_cy, face_size;
    float raw_yaw, raw_pitch, raw_roll;
    bool  raw_valid;
    float smooth_yaw, smooth_pitch, smooth_roll;
    bool  smooth_valid;
    float bias_yaw, bias_pitch;
    float pub_yaw, pub_pitch, pub_roll;
    bool  pub_valid;
    uint32_t cam_seq;
    int64_t cap_us;
    int64_t publish_us;
    float tensor_scale;
    int32_t tensor_zp;
    bool  tensor_valid;
    point2d_t landmarks[22];
    uint32_t update_count;
} web_telemetry_data_t;

static web_telemetry_data_t s_telemetry = {};
static SemaphoreHandle_t    s_telemetry_mutex = NULL;
static volatile bool        s_recalibrate_flag = false;

// [COMPARE] Snapshot tensor INT8 96x96 cua CUNG frame (PSRAM, bao ve boi s_telemetry_mutex)
#define WEB_DIAG_TENSOR_W 96
#define WEB_DIAG_TENSOR_H 96
#define WEB_DIAG_TENSOR_LEN (WEB_DIAG_TENSOR_W * WEB_DIAG_TENSOR_H)
static int8_t* s_diag_tensor = NULL;

static httpd_handle_t s_server = NULL;
static httpd_handle_t s_server_api = NULL;   // [v2] HTTPD RIENG cho API telemetry (cong 81)

// ==============================================================================
// GIAO DIỆN WEB ADAS HUD (HTML5 / CSS / JAVASCRIPT - 100% OFFLINE)
// ==============================================================================
static const char INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AIoT DMS - Phát Hiện Tài Xế Ngủ Gật & Mất Tập Trung</title>
  <style>
    :root {
      --bg: #090c13;
      --card-bg: #131926;
      --card-border: #1f2a3f;
      --text-main: #f0f4fc;
      --text-dim: #8b9bb4;
      --accent-cyan: #00e5ff;
      --accent-green: #00e676;
      --accent-orange: #ff9100;
      --accent-red: #ff1744;
      --accent-purple: #b388ff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: linear-gradient(90deg, #101624 0%, #151d30 100%);
      border-bottom: 1px solid var(--card-border);
      padding: 12px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-icon {
      font-size: 24px;
      background: #1c263b;
      padding: 6px 10px;
      border-radius: 8px;
      border: 1px solid #2b3954;
    }
    .brand-title {
      font-size: 17px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: #fff;
    }
    .brand-sub {
      font-size: 11px;
      color: var(--accent-cyan);
      letter-spacing: 0.3px;
    }
    .header-badges {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      background: #1a2336;
      border: 1px solid var(--card-border);
      color: var(--text-dim);
    }
    .badge-live {
      background: rgba(0, 230, 118, 0.15);
      color: var(--accent-green);
      border-color: rgba(0, 230, 118, 0.4);
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .badge-live::before {
      content: "";
      width: 7px;
      height: 7px;
      background: var(--accent-green);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent-green);
      animation: pulse-dot 1.5s infinite;
    }
    @keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    main {
      flex: 1;
      padding: 16px;
      max-width: 1300px;
      margin: 0 auto;
      width: 100%;
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 16px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Video & HUD Container */
    .stream-box {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: 10px;
      overflow: hidden;
      aspect-ratio: 4/3;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid #2a374f;
    }
    .stream-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .hud-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }

    /* Banner Trạng Thái Lớn */
    .status-banner {
      padding: 14px 16px;
      border-radius: 10px;
      text-align: center;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      background: rgba(0, 230, 118, 0.12);
      color: var(--accent-green);
      border: 1px solid rgba(0, 230, 118, 0.4);
    }
    .status-banner.calib {
      background: rgba(0, 229, 255, 0.12);
      color: var(--accent-cyan);
      border-color: rgba(0, 229, 255, 0.4);
    }
    .status-banner.warn {
      background: rgba(255, 145, 0, 0.18);
      color: var(--accent-orange);
      border-color: rgba(255, 145, 0, 0.5);
    }
    .status-banner.alarm {
      background: rgba(255, 23, 68, 0.25);
      color: var(--accent-red);
      border-color: rgba(255, 23, 68, 0.8);
      box-shadow: 0 0 25px rgba(255, 23, 68, 0.6);
      animation: alert-flash 0.5s infinite alternate;
    }
    @keyframes alert-flash {
      from { transform: scale(1); filter: brightness(1); }
      to { transform: scale(1.01); filter: brightness(1.3); }
    }

    /* Buttons Panel */
    .btn-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .btn {
      flex: 1;
      min-width: 120px;
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--card-border);
      background: #1b2436;
      color: var(--text-main);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.2s ease;
    }
    .btn:hover { background: #243049; border-color: #3b4d70; }
    .btn.active {
      background: rgba(0, 229, 255, 0.18);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
    }
    .btn-recalib {
      background: linear-gradient(135deg, #1e3a8a, #2563eb);
      border-color: #3b82f6;
      color: #fff;
    }
    .btn-recalib:hover { background: linear-gradient(135deg, #2563eb, #1d4ed8); }

    /* Cards Grid bên phải */
    .metrics-grid {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .metric-card {
      background: #161e2e;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .metric-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .metric-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-dim);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .metric-badge {
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 12px;
      background: #202b40;
    }
    .metric-val-large {
      font-size: 26px;
      font-weight: 800;
      font-family: monospace;
      color: #fff;
      display: flex;
      align-items: baseline;
      gap: 8px;
    }
    .metric-threshold {
      font-size: 12px;
      font-weight: 500;
      color: var(--text-dim);
    }

    /* Thanh đo tiến trình */
    .progress-bar-wrap {
      width: 100%;
      height: 8px;
      background: #0f1521;
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }
    .progress-bar-fill {
      height: 100%;
      width: 0%;
      background: var(--accent-cyan);
      border-radius: 6px;
      transition: width 0.15s ease, background-color 0.2s ease;
    }

    /* Head Pose Sub-grid */
    .pose-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      margin-top: 4px;
    }
    .pose-box {
      background: #0f1521;
      padding: 8px 10px;
      border-radius: 8px;
      text-align: center;
      border: 1px solid #1a253a;
    }
    .pose-label {
      font-size: 10px;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 600;
    }
    .pose-val {
      font-size: 15px;
      font-weight: 700;
      font-family: monospace;
      margin-top: 2px;
      color: var(--text-main);
    }

    /* Perf table */
    .perf-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
    }
    .perf-item {
      background: #0f1521;
      padding: 6px 8px;
      border-radius: 6px;
      text-align: center;
      border: 1px solid #1a253a;
    }
    .perf-item .lbl { font-size: 10px; color: var(--text-dim); }
    .perf-item .val { font-size: 12px; font-weight: 700; font-family: monospace; color: var(--accent-purple); }

    footer {
      text-align: center;
      padding: 12px;
      font-size: 11px;
      color: var(--text-dim);
      border-top: 1px solid var(--card-border);
      background: #0d121c;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">🚘</div>
      <div>
        <div class="brand-title">AIoT DRIVER MONITORING SYSTEM</div>
        <div class="brand-sub">100% Edge AI trên ESP32-S3 N16R8 CAM (OV5640)</div>
      </div>
    </div>
    <div class="header-badges">
      <div class="badge badge-live">LIVE STREAM</div>
      <div class="badge" id="badge-fps">FPS: --</div>
      <div class="badge" id="badge-mem">RAM: --</div>
    </div>
  </header>

  <main>
    <!-- CỘT TRÁI: VIDEO & HUD MỐC KHUÔN MẶT -->
    <div class="card">
      <div class="stream-box">
        <img id="streamImg" class="stream-img" src="/stream" alt="Video Camera OV5640">
        <canvas id="hudCanvas" class="hud-canvas"></canvas>
      </div>

      <!-- Trạng Thái Lớn -->
      <div id="statusBanner" class="status-banner">
        <span>ĐANG KẾT NỐI...</span>
      </div>

      <!-- Nút Điều Khiển Nhanh -->
      <div class="btn-row">
        <button id="btnRecalib" class="btn btn-recalib" onclick="triggerRecalib()">
          🔄 Hiệu chuẩn lại (5s)
        </button>
        <button id="btnSound" class="btn active" onclick="toggleAudio()">
          🔊 Còi Web: BẬT
        </button>
        <button id="btnOverlay" class="btn active" onclick="toggleOverlay()">
          🎯 Vẽ 22 mốc: BẬT
        </button>
        <button class="btn" onclick="window.open('http://' + location.hostname + ':81/debug', '_blank')">
          🔬 Debug same-frame
        </button>
      </div>
    </div>

    <!-- CỘT PHẢI: CHỈ SỐ SINH TRẮC HỌC & ĐỘ TRỄ AI -->
    <div class="metrics-grid">

      <!-- EAR: MẮT & CHỚP MẮT -->
      <div class="metric-card">
        <div class="metric-header">
          <div class="metric-title">👁️ TỈ LỆ MỞ MẮT (EAR - Eye Aspect Ratio)</div>
          <div id="earBadge" class="metric-badge" style="color: var(--accent-green);">BÌNH THƯỜNG</div>
        </div>
        <div class="metric-val-large">
          <span id="earVal">0.00</span>
          <span class="metric-threshold" id="earThr">Ngưỡng báo: 0.180</span>
        </div>
        <div class="progress-bar-wrap">
          <div id="earBar" class="progress-bar-fill"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-dim); margin-top:2px;">
          <span>Mắt trái: <b id="earLeft" style="color:#fff;">--</b> | Mắt phải: <b id="earRight" style="color:#fff;">--</b></span>
          <span>Chớp mắt: <b id="blinkCount" style="color:var(--accent-cyan);">0</b></span>
        </div>
      </div>

      <!-- MAR: MIỆNG & NGÁP -->
      <div class="metric-card">
        <div class="metric-header">
          <div class="metric-title">👄 TỈ LỆ MỞ MIỆNG (MAR - Mouth Aspect Ratio)</div>
          <div id="marBadge" class="metric-badge" style="color: var(--accent-green);">ĐÓNG</div>
        </div>
        <div class="metric-val-large">
          <span id="marVal">0.00</span>
          <span class="metric-threshold" id="marThr">Ngưỡng ngáp: 0.600</span>
        </div>
        <div class="progress-bar-wrap">
          <div id="marBar" class="progress-bar-fill"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-dim); margin-top:2px;">
          <span>Thời gian há: <b id="mouthTime" style="color:#fff;">0.0s</b></span>
          <span>Số lần ngáp: <b id="yawnCount" style="color:var(--accent-orange);">0</b></span>
        </div>
      </div>

      <!-- HEAD POSE: GÓC ĐẦU 3D -->
      <div class="metric-card">
        <div class="metric-header">
          <div class="metric-title">🧭 GÓC ĐẦU 3D (Head Pose POSIT & Hình Học)</div>
          <div id="poseBadge" class="metric-badge">NHÌN THẲNG</div>
        </div>
        <div class="pose-row">
          <div class="pose-box">
            <div class="pose-label">YAW (Trái / Phải)</div>
            <div class="pose-val" id="poseYaw">+0.0°</div>
          </div>
          <div class="pose-box">
            <div class="pose-label">PITCH (Cúi / Ngửa)</div>
            <div class="pose-val" id="posePitch">+0.0°</div>
          </div>
          <div class="pose-box">
            <div class="pose-label">ROLL (Nghiêng)</div>
            <div class="pose-val" id="poseRoll">+0.0°</div>
          </div>
        </div>
      </div>

      <!-- THỜI GIAN THỰC THI PIPELINE AI -->
      <div class="metric-card">
        <div class="metric-header">
          <div class="metric-title">⚡ ĐỘ TRỄ TỪNG KHÂU (ESP32-S3 Hardware Latency)</div>
          <div class="metric-badge" id="totalMs" style="color: var(--accent-purple);">-- ms / frame</div>
        </div>
        <div class="perf-row">
          <div class="perf-item">
            <div class="lbl">Decode JPEG</div>
            <div class="val" id="latDec">-- ms</div>
          </div>
          <div class="perf-item">
            <div class="lbl">TinyDriverNet</div>
            <div class="val" id="latAi">-- ms</div>
          </div>
          <div class="perf-item">
            <div class="lbl">PnP Solver</div>
            <div class="val" id="latPnp">-- ms</div>
          </div>
          <div class="perf-item">
            <div class="lbl">FPS Thực</div>
            <div class="val" id="perfFps" style="color:var(--accent-green);">--</div>
          </div>
        </div>
      </div>

    </div>
  </main>

  <footer>
    Project 13: Edge AI Driver Drowsiness & Distraction Detection &bull; ESP32-S3 WROOM-1 N16R8 &bull; OV5640 DVP CAM &bull; Standalone Web Dashboard
  </footer>

  <script>
    // Trạng thái cục bộ
    let soundEnabled = true;
    let overlayEnabled = true;
    let audioCtx = null;
    let beeperOsc = null;
    let isAlarmSounding = false;

    // Web Audio Synthesizer còi báo động
    function initAudio() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
    }
    function playAlarmBeep() {
      if (!soundEnabled) return;
      initAudio();
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
      if (!isAlarmSounding) {
        isAlarmSounding = true;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime); // 880 Hz
        gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        setTimeout(() => {
          osc.stop();
          isAlarmSounding = false;
        }, 180);
      }
    }

    function toggleAudio() {
      soundEnabled = !soundEnabled;
      const btn = document.getElementById('btnSound');
      btn.className = soundEnabled ? 'btn active' : 'btn';
      btn.innerText = soundEnabled ? '🔊 Còi Web: BẬT' : '🔇 Còi Web: TẮT';
      if (soundEnabled) initAudio();
    }

    function toggleOverlay() {
      overlayEnabled = !overlayEnabled;
      const btn = document.getElementById('btnOverlay');
      btn.className = overlayEnabled ? 'btn active' : 'btn';
      btn.innerText = overlayEnabled ? '🎯 Vẽ 22 mốc: BẬT' : '⭕ Vẽ 22 mốc: TẮT';
      if (!overlayEnabled) {
        const canvas = document.getElementById('hudCanvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    function triggerRecalib() {
      apiFetch('/api/recalibrate', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
          const banner = document.getElementById('statusBanner');
          banner.className = 'status-banner calib';
          banner.innerHTML = '<span>🔄 ĐANG BẮT ĐẦU HIỆU CHUẨN LẠI (5 GIÂY)...</span>';
        })
        .catch(e => console.error('Recalib error:', e));
    }

    // Vẽ Canvas Lớp Phủ 22 Mốc (Landmarks) & Vùng Crop (ROI)
    function drawHUD(data) {
      if (!overlayEnabled) return;
      const canvas = document.getElementById('hudCanvas');
      const img = document.getElementById('streamImg');
      if (!canvas || !img) return;

      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, w, h);

      if (!data.landmarks || data.landmarks.length < 44) return;

      // Toạ độ 22 điểm mốc được chuẩn hoá [0, 1] trên không gian CROP 96x96.
      // Ảnh hiển thị bằng object-fit: contain -> phải tính đúng vùng nội dung
      // (trừ viền letterbox) và dùng src_w/src_h thật, không hard-code 320x240.
      const roi = data.roi || { active: 0, x: 0, y: 0, s: 0 };
      const srcW = data.src_w || 320;
      const srcH = data.src_h || 240;
      const scale = Math.min(w / srcW, h / srcH);
      const dw = srcW * scale, dh = srcH * scale;
      const offx = (w - dw) * 0.5, offy = (h - dh) * 0.5;
      let ox = offx, oy = offy, szX = dw, szY = dh;

      // Nếu crop bám mặt: vẽ khung ROI trên vùng ảnh thật
      if (roi.active && roi.s > 0) {
        ox = offx + roi.x * scale;
        oy = offy + roi.y * scale;
        szX = roi.s * scale;
        szY = roi.s * scale;
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.45)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(ox, oy, szX, szY);
        ctx.setLineDash([]);
      }

      const pts = [];
      for (let i = 0; i < 22; i++) {
        const px = ox + data.landmarks[i * 2] * szX;
        const py = oy + data.landmarks[i * 2 + 1] * szY;
        pts.push({ x: px, y: py });
      }

      // Màu sắc theo trạng thái
      const isAlarm = data.alarm;
      const eyeColor = isAlarm ? '#ff1744' : '#00e676';
      const mouthColor = (data.mar > data.mar_thr) ? '#ff9100' : '#00e5ff';

      // 1. Mắt Trái (P0..P5)
      ctx.strokeStyle = eyeColor;
      ctx.fillStyle = eyeColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i <= 5; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      ctx.stroke();

      // 2. Mắt Phải (P6..P11)
      ctx.beginPath();
      ctx.moveTo(pts[6].x, pts[6].y);
      for (let i = 7; i <= 11; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      ctx.stroke();

      // 3. Miệng (P12..P17)
      ctx.strokeStyle = mouthColor;
      ctx.beginPath();
      ctx.moveTo(pts[12].x, pts[12].y);
      ctx.lineTo(pts[14].x, pts[14].y);
      ctx.lineTo(pts[13].x, pts[13].y);
      ctx.lineTo(pts[15].x, pts[15].y);
      ctx.closePath();
      ctx.stroke();

      // 4. Mũi & Cằm (P18..P21)
      ctx.strokeStyle = 'rgba(179, 136, 255, 0.8)';
      ctx.beginPath();
      ctx.moveTo(pts[18].x, pts[18].y);
      ctx.lineTo(pts[19].x, pts[19].y);
      ctx.lineTo(pts[20].x, pts[20].y);
      ctx.lineTo(pts[21].x, pts[21].y);
      ctx.stroke();

      // Vẽ các điểm mốc tròn nhỏ
      ctx.fillStyle = '#ffffff';
      for (let i = 0; i < 22; i++) {
        ctx.beginPath();
        ctx.arc(pts[i].x, pts[i].y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }

      // 5. Mũi tên chỉ hướng quay đầu (Head Pose vector từ chóp mũi P19)
      // [v3] VẼ 3 TRỤC X / Y / Z THẬT (giống laptop) từ yaw/pitch/roll.
      // Trước đây chỉ vẽ 1 đoạn 2D giả (sin(yaw), -sin(pitch)) nên nhìn như
      // "trục xếp thẳng hàng", thiếu hẳn trục Z và roll.
      // Quy ước: X (đỏ) = ngang mặt, Y (xanh lá) = dọc mặt (xuống cằm),
      //          Z (xanh dương) = hướng mũi (ra trước). Yaw quay quanh Y,
      //          Pitch quanh X, Roll quanh Z.
      if (data.yaw !== undefined && data.pitch !== undefined && data.roll !== undefined) {
        const nx = pts[19].x;
        const ny = pts[19].y;
        const len = szX * 0.45;
        const ry = (data.yaw   * Math.PI) / 180.0;
        const rp = (data.pitch * Math.PI) / 180.0;
        const rr = (data.roll  * Math.PI) / 180.0;
        const cy1 = Math.cos(ry), sy1 = Math.sin(ry);
        const cp1 = Math.cos(rp), sp1 = Math.sin(rp);
        const cr1 = Math.cos(rr), sr1 = Math.sin(rr);
        // R = Ry(yaw) * Rx(pitch) * Rz(roll)
        const R = [
          [ cy1*cr1 + sy1*sp1*sr1, -cy1*sr1 + sy1*sp1*cr1,  sy1*cp1],
          [ cp1*sr1,                cp1*cr1,               -sp1    ],
          [-sy1*cr1 + cy1*sp1*sr1,  sy1*sr1 + cy1*sp1*cr1,  cy1*cp1]
        ];
        const axes = [
          [1, 0, 0, '#ff1744'],
          [0, 1, 0, '#00e676'],
          [0, 0, 1, '#2979ff']
        ];
        for (let k = 0; k < 3; k++) {
          const a = axes[k];
          const x3 = R[0][0]*a[0] + R[0][1]*a[1] + R[0][2]*a[2];
          const y3 = R[1][0]*a[0] + R[1][1]*a[1] + R[1][2]*a[2];
          const z3 = R[2][0]*a[0] + R[2][1]*a[1] + R[2][2]*a[2];
          const f = 1.0 / (1.0 + 0.5 * z3);   // phối cảnh nhẹ cho dễ nhìn
          ctx.strokeStyle = a[3];
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(nx, ny);
          ctx.lineTo(nx + x3 * len * f, ny + y3 * len * f);
          ctx.stroke();
        }
      }
    }

    // Polling Telemetry từ ESP32 (/status) mỗi 120ms (~8 lần/giây)
    // [v2] API telemetry nam o CONG 81 (HTTP server RIENG) vi stream MJPEG o cong 80
    // chay vong lap vo han trong task httpd -> se chan /status neu dung chung server.
    // Neu khong ket noi duoc (vi du firmware cu) -> tu dong quay ve duong dan tuong doi.
    const API_BASE = 'http://' + location.hostname + ':81';
    let apiUseBase = true;
    async function apiFetch(path, opts) {
      if (apiUseBase) {
        try { return await fetch(API_BASE + path, opts); }
        catch (e) { apiUseBase = false; }
      }
      return await fetch(path, opts);
    }

    async function updateStatus() {
      try {
        const res = await apiFetch('/status');
        if (!res.ok) return;
        const d = await res.json();

        // 1. Banner Trạng Thái
        const banner = document.getElementById('statusBanner');
        let bannerText = d.status || 'BÌNH THƯỜNG';
        let bannerClass = 'status-banner';

        if (d.alarm) {
          bannerClass += ' alarm';
          bannerText = '🚨 CẢNH BÁO: ' + bannerText;
          playAlarmBeep();
        } else if (bannerText.includes('CALIBRAT')) {
          bannerClass += ' calib';
          bannerText = '🔵 ĐANG TỰ HIỆU CHUẨN...';
        } else if (bannerText.includes('WARN') || bannerText.includes('YAWN') || bannerText.includes('DISTRACT')) {
          bannerClass += ' warn';
          bannerText = '⚠️ ' + bannerText;
        } else {
          bannerText = '✅ ' + bannerText;
        }
        banner.className = bannerClass;
        banner.innerText = bannerText;

        // 2. EAR
        document.getElementById('earVal').innerText = d.ear.toFixed(2);
        document.getElementById('earThr').innerText = 'Ngưỡng báo: ' + d.ear_thr.toFixed(3);
        document.getElementById('earLeft').innerText = (d.ear_l !== undefined ? d.ear_l.toFixed(2) : '--');
        document.getElementById('earRight').innerText = (d.ear_r !== undefined ? d.ear_r.toFixed(2) : '--');
        document.getElementById('blinkCount').innerText = d.blinks || 0;

        const earPct = Math.min(Math.max((d.ear / 0.5) * 100, 0), 100);
        const earBar = document.getElementById('earBar');
        earBar.style.width = earPct + '%';
        earBar.style.backgroundColor = (d.ear < d.ear_thr) ? 'var(--accent-red)' : 'var(--accent-cyan)';

        const earBadge = document.getElementById('earBadge');
        if (d.ear < d.ear_thr) {
          earBadge.innerText = 'NHẮM MẮT';
          earBadge.style.color = 'var(--accent-red)';
        } else {
          earBadge.innerText = 'BÌNH THƯỜNG';
          earBadge.style.color = 'var(--accent-green)';
        }

        // 3. MAR
        document.getElementById('marVal').innerText = d.mar.toFixed(2);
        document.getElementById('marThr').innerText = 'Ngưỡng ngáp: ' + d.mar_thr.toFixed(3);
        document.getElementById('mouthTime').innerText = (d.mouth_s || 0).toFixed(1) + 's';
        document.getElementById('yawnCount').innerText = d.yawns || 0;

        const marPct = Math.min(Math.max((d.mar / 1.0) * 100, 0), 100);
        const marBar = document.getElementById('marBar');
        marBar.style.width = marPct + '%';
        marBar.style.backgroundColor = (d.mar > d.mar_thr) ? 'var(--accent-orange)' : 'var(--accent-green)';

        const marBadge = document.getElementById('marBadge');
        if (d.mar > d.mar_thr) {
          marBadge.innerText = 'ĐANG NGÁP';
          marBadge.style.color = 'var(--accent-orange)';
        } else {
          marBadge.innerText = 'ĐÓNG';
          marBadge.style.color = 'var(--accent-green)';
        }

        // 4. Head Pose
        document.getElementById('poseYaw').innerText = (d.yaw >= 0 ? '+' : '') + d.yaw.toFixed(1) + '°';
        document.getElementById('posePitch').innerText = (d.pitch >= 0 ? '+' : '') + d.pitch.toFixed(1) + '°';
        document.getElementById('poseRoll').innerText = (d.roll >= 0 ? '+' : '') + d.roll.toFixed(1) + '°';

        const poseBadge = document.getElementById('poseBadge');
        if (Math.abs(d.yaw) > 30.0 || Math.abs(d.pitch) > 25.0) {
          poseBadge.innerText = 'MẤT TẬP TRUNG';
          poseBadge.style.color = 'var(--accent-orange)';
        } else {
          poseBadge.innerText = 'NHÌN THẲNG';
          poseBadge.style.color = 'var(--accent-green)';
        }

        // 5. Performance
        document.getElementById('badge-fps').innerText = 'FPS: ' + d.fps.toFixed(1);
        document.getElementById('perfFps').innerText = d.fps.toFixed(1);
        document.getElementById('latDec').innerText = d.dec.toFixed(1) + ' ms';
        document.getElementById('latAi').innerText = d.ai.toFixed(1) + ' ms';
        document.getElementById('latPnp').innerText = (d.pnp !== undefined ? d.pnp.toFixed(1) : '0.1') + ' ms';
        document.getElementById('totalMs').innerText = d.total.toFixed(1) + ' ms / frame';

        if (d.psram_free) {
          document.getElementById('badge-mem').innerText = 'PSRAM: ' + Math.round(d.psram_free / 1024) + ' KB';
        }

        // 6. Vẽ HUD Canvas
        drawHUD(d);

      } catch (err) {
        // Mất kết nối tạm thời
      } finally {
        setTimeout(updateStatus, 120);
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      setTimeout(updateStatus, 300);
      // Tự động kết nối lại luồng video nếu bị ngắt
      const img = document.getElementById('streamImg');
      img.onerror = () => {
        setTimeout(() => { img.src = '/stream?t=' + Date.now(); }, 1000);
      };
    });
  </script>
</body>
</html>
)rawliteral";

// ==============================================================================
// [P3] TRANG DEBUG SAME-FRAME: ảnh JPEG và 22 landmark lấy từ CÙNG một response
// (/debug/snapshot) nên không còn cảnh "ảnh frame mới + landmark frame cũ".
// Toạ độ: landmark chuẩn hoá trong không gian CROP -> đổi ra ảnh gốc ->
// đổi ra vùng hiển thị thật (object-fit: contain, có viền letterbox).
// ==============================================================================
static const char DEBUG_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>P13 Debug - Same-frame snapshot</title>
<style>
  body { margin:0; background:#0b0f18; color:#e6edf7; font-family:system-ui,"Segoe UI",Roboto,sans-serif; }
  header { padding:10px 14px; border-bottom:1px solid #223049; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; }
  .meta { font:12px/1.5 ui-monospace,Consolas,monospace; color:#8fa6c8; }
  .wrap { position:relative; margin:12px auto; max-width:960px; background:#000; border:1px solid #223049; border-radius:8px; overflow:hidden; }
  img { display:block; width:100%; height:auto; }
  canvas { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  #err { color:#ff6b6b; font-size:12px; padding:0 14px; }
  button { background:#1b2436; color:#e6edf7; border:1px solid #2b3954; border-radius:6px; padding:6px 10px; cursor:pointer; }
</style>
</head>
<body>
<header>
  <h1>🔬 P13 Debug — ảnh &amp; landmark CÙNG frame</h1>
  <span class="meta" id="info">đang chờ frame...</span>
  <button onclick="toggle()" id="btn">Tạm dừng</button>
</header>
<div class="wrap" id="wrap">
  <img id="img" alt="snapshot"/>
  <canvas id="cv"></canvas>
</div>
<div id="err"></div>
<script>
let running = true, busy = false, pending = null;
const img = document.getElementById('img');
const cv = document.getElementById('cv');
const info = document.getElementById('info');
const err = document.getElementById('err');

function toggle() {
  running = !running;
  document.getElementById('btn').innerText = running ? 'Tạm dừng' : 'Tiếp tục';
}

img.addEventListener('load', () => { if (pending) { draw(pending); pending = null; } });

function draw(d) {
  const srcW = d.w || img.naturalWidth || 320;
  const srcH = d.h || img.naturalHeight || 240;
  const cw = img.clientWidth, ch = img.clientHeight;
  if (!cw || !ch) return;
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cw, ch);

  const scale = Math.min(cw / srcW, ch / srcH);
  const dw = srcW * scale, dh = srcH * scale;
  const offx = (cw - dw) / 2, offy = (ch - dh) / 2;
  const roi = d.roi || { active: 0 };
  let ox = offx, oy = offy, bs = dw;

  if (roi.active && roi.s > 0) {
    ox = offx + roi.x * scale;
    oy = offy + roi.y * scale;
    bs = roi.s * scale;
    ctx.strokeStyle = 'rgba(0,229,255,0.75)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(ox, oy, bs, bs);
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0,229,255,0.9)';
    ctx.font = '12px ui-monospace,Consolas,monospace';
    ctx.fillText('crop used ' + roi.x + ',' + roi.y + ',' + roi.s, ox + 4, Math.max(12, oy - 4));
  }

  // 106-point pretrained landmark overlay (2d106det) — tiny cyan dots
  const lm106 = d.lm106 || [];
  if (lm106.length >= 212) {
    ctx.fillStyle = 'rgba(0,229,255,0.85)';
    for (let i = 0; i < 106; i++) {
      const px = ox + lm106[i*2] * bs;
      const py = oy + lm106[i*2+1] * bs;
      ctx.beginPath(); ctx.arc(px, py, 1.3, 0, 6.283); ctx.fill();
    }
  }

  const lm = d.landmarks || [];
  if (lm.length < 44) return;
  const pts = [];
  for (let i = 0; i < 22; i++) pts.push({ x: ox + lm[i*2] * bs, y: oy + lm[i*2+1] * bs });

  ctx.strokeStyle = '#00e676'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i <= 5; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pts[6].x, pts[6].y);
  for (let i = 7; i <= 11; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath(); ctx.stroke();
  ctx.strokeStyle = '#ff9100';
  ctx.beginPath(); ctx.moveTo(pts[12].x, pts[12].y);
  ctx.lineTo(pts[14].x, pts[14].y); ctx.lineTo(pts[13].x, pts[13].y);
  ctx.lineTo(pts[15].x, pts[15].y); ctx.closePath(); ctx.stroke();
  ctx.strokeStyle = 'rgba(179,136,255,0.9)';
  ctx.beginPath(); ctx.moveTo(pts[18].x, pts[18].y);
  ctx.lineTo(pts[19].x, pts[19].y); ctx.lineTo(pts[20].x, pts[20].y);
  ctx.lineTo(pts[21].x, pts[21].y); ctx.stroke();
  ctx.fillStyle = '#fff';
  for (let i = 0; i < 22; i++) { ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, 2, 0, 6.283); ctx.fill(); }
}

async function loop() {
  if (running && !busy) {
    busy = true;
    try {
      const r = await fetch('/debug/snapshot?t=' + Date.now(), { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      err.innerText = '';
      info.innerText = 'frame #' + d.frame_id + ' | img_age ' + d.img_age.toFixed(0) +
        ' ms | roi_age ' + (d.roi_age >= 0 ? d.roi_age.toFixed(0) : '--') + ' ms | src ' +
        (d.w || '?') + 'x' + (d.h || '?') + ' | roi ' +
        (d.roi && d.roi.active ? (d.roi.x + ',' + d.roi.y + ',' + d.roi.s) : 'none');
      pending = d;
      img.src = 'data:image/jpeg;base64,' + d.jpeg_b64;
    } catch (e) {
      err.innerText = 'Lỗi snapshot: ' + e.message;
    }
    busy = false;
  }
  setTimeout(loop, 150);
}
loop();
</script>
</body>
</html>
)rawliteral";

// ==============================================================================
// CÁC HANDLER CỦA HTTP SERVER
// ==============================================================================

// 1. GET / : Trả về trang chủ Dashboard
static esp_err_t index_handler(httpd_req_t* req) {
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_set_hdr(req, "Content-Encoding", "identity");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

// 1b. GET /capture : Trả về 1 khung hình JPEG chụp nhanh (hỗ trợ polling / snapshot)
static esp_err_t capture_handler(httpd_req_t* req) {
    if (!s_frame_mutex || !s_stream_frame || s_stream_frame_len == 0) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    esp_err_t res = ESP_OK;
    if (xSemaphoreTake(s_frame_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        httpd_resp_set_type(req, "image/jpeg");
        httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
        res = httpd_resp_send(req, (const char*)s_stream_frame, s_stream_frame_len);
        xSemaphoreGive(s_frame_mutex);
    } else {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    return res;
}

// 2. GET /stream : Luồng video MJPEG (multipart/x-mixed-replace)
static esp_err_t stream_handler(httpd_req_t* req) {
    esp_err_t res = ESP_OK;
    char part_buf[128];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "10");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    httpd_resp_set_hdr(req, "Pragma", "no-cache");

    ESP_LOGI(TAG, "📹 Client đã kết nối vào luồng video MJPEG /stream");

    // Vùng đệm cục bộ trong PSRAM để copy frame gửi đi, tránh giữ mutex lâu
    uint8_t* local_buf = (uint8_t*)heap_caps_malloc(MAX_STREAM_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!local_buf) {
        ESP_LOGE(TAG, "Không thể cấp phát local_buf trong PSRAM cho stream!");
        return ESP_FAIL;
    }

    while (true) {
        size_t current_len = 0;
        if (xSemaphoreTake(s_frame_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            if (s_stream_frame && s_stream_frame_len > 0 && s_stream_frame_len <= MAX_STREAM_BUFFER_SIZE) {
                memcpy(local_buf, s_stream_frame, s_stream_frame_len);
                current_len = s_stream_frame_len;
            }
            xSemaphoreGive(s_frame_mutex);
        }

        if (current_len > 0) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
            if (res != ESP_OK) break;

            size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, (unsigned int)current_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
            if (res != ESP_OK) break;

            res = httpd_resp_send_chunk(req, (const char*)local_buf, current_len);
            if (res != ESP_OK) break;
        }

        // Khống chế nhịp gửi ~10-12 FPS để tránh nghẽn băng thông Wi-Fi
        vTaskDelay(pdMS_TO_TICKS(85));
    }

    free(local_buf);
    ESP_LOGW(TAG, "Client đã ngắt kết nối luồng video /stream");
    return res;
}

// 3b. GET /debug/lm106 : 106 landmark points of the pretrained model (separate from /status)
static esp_err_t lm106_handler(httpd_req_t* req) {
    char buf[4096];
    int off = snprintf(buf, sizeof(buf), "{\"lm106\":[");
    for (int i = 0; i < 106 && off < (int)sizeof(buf) - 32; i++) {
        off += snprintf(buf + off, sizeof(buf) - off, "%.3f,%.3f%s",
                        g_lm106[i][0], g_lm106[i][1], (i < 105) ? "," : "");
    }
    off += snprintf(buf + off, sizeof(buf) - off, "]}");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    return httpd_resp_send(req, buf, off);
}

// 3. GET /status : Trả về số liệu JSON thời gian thực
static esp_err_t status_handler(httpd_req_t* req) {
    char* json_str = (char*)heap_caps_malloc(3072, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!json_str) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    web_telemetry_data_t t;
    if (xSemaphoreTake(s_telemetry_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        t = s_telemetry;
        xSemaphoreGive(s_telemetry_mutex);
    } else {
        free(json_str);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    int offset = snprintf(json_str, 3072,
        "{\"ear\":%.2f,\"ear_l\":%.2f,\"ear_r\":%.2f,\"eye_prob\":%.3f,\"mar\":%.2f,"
        "\"yaw\":%.1f,\"pitch\":%.1f,\"roll\":%.1f,\"status\":\"%s\","
        "\"alarm\":%s,\"fps\":%.1f,\"dec\":%.1f,\"ai\":%.1f,\"pnp\":%.1f,\"total\":%.1f,"
        "\"ear_thr\":%.3f,\"mar_thr\":%.3f,\"mouth_s\":%.2f,\"yawns\":%u,\"blinks\":%u,"
        "\"roi\":{\"active\":%d,\"x\":%d,\"y\":%d,\"s\":%d,\"source\":%d},"
        "\"src_w\":%d,\"src_h\":%d,"
        "\"frame_id\":%u,\"cam_seq\":%u,\"img_age\":%.1f,\"roi_age\":%.1f,\"core1\":%.1f,"
        "\"eye\":{\"valid\":%s,\"closed\":%s,\"closed_ms\":%.1f,\"age_ms\":%.1f,\"reset\":\"%s\"},"
        "\"psram_free\":%u,\"sram_free\":%u,\"landmarks\":[",
        t.ear, t.ear_left, t.ear_right, t.eye_prob, t.mar,
        t.pose_valid ? t.yaw : 0.0f,
        t.pose_valid ? t.pitch : 0.0f,
        t.pose_valid ? t.roll : 0.0f,
        t.status_str,
        t.is_alarm_active ? "true" : "false",
        t.fps,
        t.dec_ms, t.ai_ms, t.pnp_ms, t.total_ms,
        t.ear_threshold, t.mar_threshold, t.mouth_open_s,
        (unsigned int)t.total_yawns,
        (unsigned int)t.total_blinks,
        t.roi_active, t.roi_x0, t.roi_y0, t.roi_size, t.roi_source,
        t.src_w, t.src_h,
        (unsigned int)t.frame_id, (unsigned int)t.cam_seq,
        (double)t.img_age_ms, (double)t.roi_age_ms,
        (double)t.core1_ms,
        t.eye.valid ? "true" : "false", t.eye.closed ? "true" : "false",
        (double)t.eye.closed_ms, (double)t.eye.age_ms,
        t.eye.reset_reason ? t.eye.reset_reason : "no_sample",
        (unsigned int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        (unsigned int)heap_caps_get_free_size(MALLOC_CAP_INTERNAL)
    );

    for (int i = 0; i < 22; i++) {
        offset += snprintf(json_str + offset, 3072 - offset,
                           "%.3f,%.3f%s",
                           t.landmarks[i].x, t.landmarks[i].y,
                           (i < 21) ? "," : "");
    }
    offset += snprintf(json_str + offset, 3072 - offset, "]}");

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    esp_err_t res = httpd_resp_send(req, json_str, offset);

    free(json_str);
    return res;
}

// 4. POST /api/recalibrate : Nhận lệnh hiệu chuẩn lại từ Web UI
static esp_err_t recalibrate_handler(httpd_req_t* req) {
    s_recalibrate_flag = true;
    ESP_LOGI(TAG, "🔄 Người dùng đã kích hoạt hiệu chuẩn lại từ Web UI!");

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    const char* resp = "{\"status\":\"ok\",\"message\":\"Recalibration triggered\"}";
    return httpd_resp_send(req, resp, strlen(resp));
}

// 5. GET /debug : Trang debug same-frame (chạy trên HTTPD cổng 81, không bị stream chặn)
static esp_err_t debug_page_handler(httpd_req_t* req) {
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    return httpd_resp_send(req, DEBUG_HTML, strlen(DEBUG_HTML));
}

// [COMPARE] Chi giu ky tu an toan cho session_id (tranh pha JSON).
static void debug_sanitize_token(const char* in, char* out, size_t out_len) {
    size_t j = 0;
    if (!out || out_len == 0) return;
    if (in) {
        for (size_t i = 0; in[i] != '\0' && j + 1 < out_len; i++) {
            unsigned char c = (unsigned char)in[i];
            if (isalnum(c) || c == '-' || c == '_' || c == '.') {
                out[j++] = (char)c;
            }
        }
    }
    out[j] = '\0';
}

static size_t b64_encode_to(char* dst, const uint8_t* src, size_t len) {
    static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0, i = 0;
    for (; i + 2 < len; i += 3) {
        uint32_t v = ((uint32_t)src[i] << 16) |
                     ((uint32_t)src[i + 1] << 8) |
                     (uint32_t)src[i + 2];
        dst[o++] = B64[(v >> 18) & 63];
        dst[o++] = B64[(v >> 12) & 63];
        dst[o++] = B64[(v >> 6) & 63];
        dst[o++] = B64[v & 63];
    }
    if (i < len) {
        uint32_t v = (uint32_t)src[i] << 16;
        if (i + 1 < len) v |= (uint32_t)src[i + 1] << 8;
        dst[o++] = B64[(v >> 18) & 63];
        dst[o++] = B64[(v >> 12) & 63];
        dst[o++] = (i + 1 < len) ? B64[(v >> 6) & 63] : '=';
        dst[o++] = '=';
    }
    return o;
}

// 6. GET /debug/snapshot : trả JSON chứa metadata + landmark + tensor INT8 + JPEG
// của CÙNG một frame. Ảnh, telemetry và tensor được copy dưới cùng thứ tự khóa
// (frame -> telemetry) như web_server_publish_result() nên luôn nhất quán.
// Query: ?session=<id>&tensor=0|1   (tensor=0 để giảm tải khi chỉ cần JPEG)
static esp_err_t debug_snapshot_handler(httpd_req_t* req) {
    if (!s_frame_mutex || !s_telemetry_mutex || !s_stream_frame || !s_diag_tensor) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    uint8_t* jpeg_copy = (uint8_t*)heap_caps_malloc(MAX_STREAM_BUFFER_SIZE,
                                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t* tensor_copy = (int8_t*)heap_caps_malloc(WEB_DIAG_TENSOR_LEN,
                                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!jpeg_copy || !tensor_copy) {
        free(jpeg_copy);
        free(tensor_copy);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    web_telemetry_data_t t = {};
    size_t jlen = 0;
    bool have_frame = false;
    bool have_tensor = false;

    if (xSemaphoreTake(s_frame_mutex, pdMS_TO_TICKS(200)) == pdTRUE) {
        if (xSemaphoreTake(s_telemetry_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            t = s_telemetry;
            if (s_telemetry.tensor_valid) {
                memcpy(tensor_copy, s_diag_tensor, WEB_DIAG_TENSOR_LEN);
                have_tensor = true;
            }
            if (s_stream_frame_len > 0 && s_stream_frame_len <= MAX_STREAM_BUFFER_SIZE) {
                memcpy(jpeg_copy, s_stream_frame, s_stream_frame_len);
                jlen = s_stream_frame_len;
                have_frame = true;
            }
            xSemaphoreGive(s_telemetry_mutex);
        }
        xSemaphoreGive(s_frame_mutex);
    }

    if (!have_frame) {
        free(jpeg_copy);
        free(tensor_copy);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // Query: session id + co lay tensor hay khong
    char session[64] = "";
    bool want_tensor = true;
    size_t qlen = httpd_req_get_url_query_len(req);
    if (qlen > 0 && qlen < 256) {
        char qbuf[256];
        if (httpd_req_get_url_query_str(req, qbuf, sizeof(qbuf)) == ESP_OK) {
            char val[96];
            if (httpd_query_key_value(qbuf, "session", val, sizeof(val)) == ESP_OK) {
                debug_sanitize_token(val, session, sizeof(session));
            }
            if (httpd_query_key_value(qbuf, "tensor", val, sizeof(val)) == ESP_OK) {
                want_tensor = (atoi(val) != 0);
            }
        }
    }

    const size_t b64_jpeg = 4 * ((jlen + 2) / 3) + 4;
    const size_t b64_tensor = 4 * ((WEB_DIAG_TENSOR_LEN + 2) / 3) + 4;
    const size_t cap = b64_jpeg + b64_tensor + 8192;
    char* out = (char*)heap_caps_malloc(cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!out) {
        free(jpeg_copy);
        free(tensor_copy);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    int off = snprintf(out, cap,
        "{\"session_id\":\"%s\",\"frame_id\":%u,\"cam_seq\":%u,"
        "\"img_age\":%.2f,\"roi_age\":%.2f,\"core1\":%.2f,"
        "\"dec\":%.2f,\"ai\":%.2f,\"pnp\":%.2f,\"total\":%.2f,\"fps\":%.2f,"
        "\"cap_us\":%lld,\"publish_us\":%lld,"
        "\"w\":%d,\"h\":%d,"
        "\"roi\":{\"active\":%d,\"x\":%d,\"y\":%d,\"s\":%d,\"source\":%d},"
        "\"face\":{\"valid\":%s,\"score\":%.4f,\"cx\":%.2f,\"cy\":%.2f,\"size\":%.2f},"
        "\"ear\":{\"left\":%.5f,\"right\":%.5f,\"avg\":%.5f},\"mar\":%.5f,"
        "\"eye\":{\"valid\":%s,\"closed\":%s,\"closed_ms\":%.1f,\"age_ms\":%.1f,\"reset\":\"%s\"},"
        "\"status\":\"%s\",\"alarm\":%s,"
        "\"ear_threshold\":%.4f,\"mar_threshold\":%.4f,"
        "\"pose\":{\"raw\":{\"yaw\":%.3f,\"pitch\":%.3f,\"roll\":%.3f,\"valid\":%s},"
        "\"smoothed\":{\"yaw\":%.3f,\"pitch\":%.3f,\"roll\":%.3f,\"valid\":%s},"
        "\"bias\":{\"yaw\":%.3f,\"pitch\":%.3f},"
        "\"published\":{\"yaw\":%.3f,\"pitch\":%.3f,\"roll\":%.3f,\"valid\":%s}},"
        "\"landmarks\":[",
        session,
        (unsigned)t.frame_id, (unsigned)t.cam_seq,
        (double)t.img_age_ms, (double)t.roi_age_ms, (double)t.core1_ms,
        (double)t.dec_ms, (double)t.ai_ms, (double)t.pnp_ms, (double)t.total_ms,
        (double)t.fps,
        (long long)t.cap_us, (long long)t.publish_us,
        t.src_w, t.src_h,
        t.roi_active, t.roi_x0, t.roi_y0, t.roi_size, t.roi_source,
        t.face_valid ? "true" : "false", (double)t.face_score,
        (double)t.face_cx, (double)t.face_cy, (double)t.face_size,
        (double)t.ear_left, (double)t.ear_right, (double)t.ear, (double)t.mar,
        t.eye.valid ? "true" : "false", t.eye.closed ? "true" : "false",
        (double)t.eye.closed_ms, (double)t.eye.age_ms,
        t.eye.reset_reason ? t.eye.reset_reason : "no_sample",
        t.status_str, t.is_alarm_active ? "true" : "false",
        (double)t.ear_threshold, (double)t.mar_threshold,
        (double)t.raw_yaw, (double)t.raw_pitch, (double)t.raw_roll,
        t.raw_valid ? "true" : "false",
        (double)t.smooth_yaw, (double)t.smooth_pitch, (double)t.smooth_roll,
        t.smooth_valid ? "true" : "false",
        (double)t.bias_yaw, (double)t.bias_pitch,
        (double)t.pub_yaw, (double)t.pub_pitch, (double)t.pub_roll,
        t.pub_valid ? "true" : "false");

    for (int i = 0; i < 22; i++) {
        off += snprintf(out + off, cap - (size_t)off, "%.6f,%.6f%s",
                        (double)t.landmarks[i].x, (double)t.landmarks[i].y,
                        (i < 21) ? "," : "");
    }

    if (!want_tensor) {
        off += snprintf(out + off, cap - (size_t)off,
                        "],\"tensor_valid\":false,\"tensor\":null,\"tensor_skipped\":true,\"jpeg_b64\":\"");
    } else if (have_tensor) {
        off += snprintf(out + off, cap - (size_t)off,
                        "],\"tensor_valid\":true,\"tensor\":{\"encoding\":\"int8\","
                        "\"w\":%d,\"h\":%d,\"scale\":%.9g,\"zero_point\":%d,\"b64\":\"",
                        WEB_DIAG_TENSOR_W, WEB_DIAG_TENSOR_H,
                        (double)t.tensor_scale, (int)t.tensor_zp);
        off += (int)b64_encode_to(out + off, (const uint8_t*)tensor_copy, WEB_DIAG_TENSOR_LEN);
        off += snprintf(out + off, cap - (size_t)off, "\"},\"jpeg_b64\":\"");
    } else {
        off += snprintf(out + off, cap - (size_t)off,
                        "],\"tensor_valid\":false,\"tensor\":null,\"jpeg_b64\":\"");
    }

    off += (int)b64_encode_to(out + off, jpeg_copy, jlen);
    off += snprintf(out + off, cap - (size_t)off, "\"}");
    out[off] = '\0';

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    esp_err_t res = httpd_resp_send(req, out, off);

    free(out);
    free(jpeg_copy);
    free(tensor_copy);
    return res;
}

// ==============================================================================
// PUBLIC API IMPLEMENTATION
// ==============================================================================

bool web_server_init(void) {
    if (s_server != NULL) {
        ESP_LOGW(TAG, "Web Server đã chạy từ trước!");
        return true;
    }

    s_frame_mutex = xSemaphoreCreateMutex();
    s_telemetry_mutex = xSemaphoreCreateMutex();

    // Cấp phát vùng đệm frame streaming trong 8MB Octal PSRAM
    s_stream_frame = (uint8_t*)heap_caps_malloc(MAX_STREAM_BUFFER_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_stream_frame) {
        ESP_LOGE(TAG, "Không thể cấp phát s_stream_frame trong PSRAM!");
        return false;
    }
    s_diag_tensor = (int8_t*)heap_caps_malloc(WEB_DIAG_TENSOR_LEN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_diag_tensor) {
        ESP_LOGW(TAG, "[COMPARE] Khong cap phat duoc tensor 96x96 -> /debug/snapshot khong co tensor");
    }

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    config.max_open_sockets = 7;
    config.stack_size = 6144;
    config.core_id = 0; // Chạy trên Core 0 (nhường Core 1 trọn vẹn cho AI)
    config.lru_purge_enable = true;

    ESP_LOGI(TAG, "Đang khởi động HTTP Web Server trên cổng %d...", config.server_port);
    esp_err_t err = httpd_start(&s_server, &config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Khởi động HTTP Web Server thất bại: 0x%x", err);
        return false;
    }

    // Đăng ký các URI routes
    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,
        .user_ctx  = NULL
    };
    httpd_register_uri_handler(s_server, &index_uri);

    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };
    httpd_register_uri_handler(s_server, &stream_uri);

    httpd_uri_t capture_uri = {
        .uri       = "/capture",
        .method    = HTTP_GET,
        .handler   = capture_handler,
        .user_ctx  = NULL
    };
    httpd_register_uri_handler(s_server, &capture_uri);

    httpd_uri_t status_uri = {
        .uri       = "/status",
        .method    = HTTP_GET,
        .handler   = status_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t recalib_uri = {
        .uri       = "/api/recalibrate",
        .method    = HTTP_POST,
        .handler   = recalibrate_handler,
        .user_ctx  = NULL
    };

    // [P3] Debug same-frame: trang + snapshot (metadata & JPEG trong 1 response)
    httpd_uri_t debug_page_uri = {
        .uri       = "/debug",
        .method    = HTTP_GET,
        .handler   = debug_page_handler,
        .user_ctx  = NULL
    };
    httpd_uri_t debug_snap_uri = {
        .uri       = "/debug/snapshot",
        .method    = HTTP_GET,
        .handler   = debug_snapshot_handler,
        .user_ctx  = NULL
    };

    // ========================================================================
    // [v2 - FIX QUAN TRONG] HTTPD THU 2 (cong 81) CHI cho API telemetry.
    // LY DO: esp_http_server chay TAT CA handler inline trong MOT task duy nhat
    // (httpd_process_session goi handler truc tiep). stream_handler MJPEG lap vo
    // han -> CHAN MOI request khac -> /status (polling 120ms) va /api/recalibrate
    // (nut bam) KHONG BAO GIO duoc phuc vu => Dashboard ket "DANG KET NOI...",
    // thong so 0.00, nut khong bam duoc (dung trieu chung quan sat duoc).
    // Tach API sang instance rieng => cap nhat duoc TRONG KHI video dang stream.
    // ========================================================================
    httpd_config_t cfg2 = HTTPD_DEFAULT_CONFIG();
    cfg2.server_port = 81;
    cfg2.ctrl_port = 32769;          // phai KHAC ctrl_port cua instance 1 (32768)
    cfg2.max_open_sockets = 4;
    cfg2.stack_size = 4096;
    cfg2.core_id = 0;
    cfg2.lru_purge_enable = true;
    if (httpd_start(&s_server_api, &cfg2) == ESP_OK) {
        httpd_register_uri_handler(s_server_api, &status_uri);
        httpd_register_uri_handler(s_server_api, &recalib_uri);
        httpd_register_uri_handler(s_server_api, &debug_page_uri);
        httpd_register_uri_handler(s_server_api, &debug_snap_uri);
        ESP_LOGI(TAG, "API telemetry: http://<IP>:81/status  (khong bi stream chan)");
        ESP_LOGI(TAG, "[P3] Debug same-frame: http://<IP>:81/debug");
    } else {
        ESP_LOGE(TAG, "Khong tao duoc HTTPD API (cong 81) -> dashboard se khong cap nhat");
        // Fallback: van dang ky tren cong 80 (chap nhan bi stream chan)
        httpd_register_uri_handler(s_server, &status_uri);
        httpd_register_uri_handler(s_server, &recalib_uri);
    }

    ESP_LOGI(TAG, "✅ HTTP Web Server đã sẵn sàng phục vụ Dashboard & Video Stream!");
    return true;
}

void web_server_update_frame(const uint8_t* jpeg_data, size_t jpeg_len) {
    if (!s_frame_mutex || !s_stream_frame || !jpeg_data || jpeg_len == 0) return;
    if (jpeg_len > MAX_STREAM_BUFFER_SIZE) return;

    if (xSemaphoreTake(s_frame_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        memcpy(s_stream_frame, jpeg_data, jpeg_len);
        s_stream_frame_len = jpeg_len;
        xSemaphoreGive(s_frame_mutex);
    }
}

// Cập nhật s_telemetry. Caller PHẢI giữ s_telemetry_mutex.
static void telemetry_store_locked(const adas_metrics_t* metrics,
                                   const point2d_t landmarks_22[22],
                                   int roi_active, int roi_x0, int roi_y0, int roi_size,
                                   int src_w, int src_h,
                                   float dec_ms, float ai_ms, float pnp_ms, float total_ms,
                                   uint32_t frame_id, float img_age_ms, float roi_age_ms,
                                   float core1_ms,
                                   const web_diag_meta_t* diag) {
    s_telemetry.ear            = metrics->ear;
    s_telemetry.ear_left       = metrics->ear_left;
    s_telemetry.ear_right      = metrics->ear_right;
    s_telemetry.mar            = metrics->mar;
    s_telemetry.yaw            = metrics->pose.yaw;
    s_telemetry.pitch          = metrics->pose.pitch;
    s_telemetry.roll           = metrics->pose.roll;
    s_telemetry.pose_valid     = metrics->pose.is_valid;
    strncpy(s_telemetry.status_str, metrics->status_str, sizeof(s_telemetry.status_str) - 1);
    s_telemetry.status_str[sizeof(s_telemetry.status_str) - 1] = '\0';
    s_telemetry.is_alarm_active = metrics->is_alarm_active;
    s_telemetry.eye             = metrics->eye;
    s_telemetry.fps            = metrics->esp32_fps;
    s_telemetry.dec_ms         = dec_ms;
    s_telemetry.ai_ms          = ai_ms;
    s_telemetry.pnp_ms         = pnp_ms;
    s_telemetry.total_ms       = total_ms;
    s_telemetry.ear_threshold  = metrics->ear_threshold;
    s_telemetry.mar_threshold  = metrics->mar_threshold;
    s_telemetry.mouth_open_s   = metrics->mouth_open_s;
    s_telemetry.eye_prob       = metrics->eye_prob;
    s_telemetry.total_yawns    = metrics->total_yawns;
    s_telemetry.total_blinks   = metrics->total_blinks;
    s_telemetry.roi_active     = roi_active;
    s_telemetry.roi_x0         = roi_x0;
    s_telemetry.roi_y0         = roi_y0;
    s_telemetry.roi_size       = roi_size;
    if (src_w > 0 && src_h > 0) {
        s_telemetry.src_w = src_w;
        s_telemetry.src_h = src_h;
    }
    s_telemetry.frame_id       = frame_id;
    s_telemetry.img_age_ms     = img_age_ms;
    s_telemetry.roi_age_ms     = roi_age_ms;
    s_telemetry.core1_ms       = core1_ms;

    // [COMPARE] Metadata + tensor cua CUNG frame. Neu khong co diag/tensor
    // thi danh dau invalid de khong bao gio tra tensor cu voi JPEG moi.
    s_telemetry.tensor_valid = false;
    if (diag) {
        s_telemetry.roi_source = diag->roi_source;
        s_telemetry.face_valid = diag->face_valid;
        s_telemetry.face_score = diag->face_score;
        s_telemetry.face_cx    = diag->face_cx;
        s_telemetry.face_cy    = diag->face_cy;
        s_telemetry.face_size  = diag->face_size;
        s_telemetry.raw_yaw    = diag->pose_raw.yaw;
        s_telemetry.raw_pitch  = diag->pose_raw.pitch;
        s_telemetry.raw_roll   = diag->pose_raw.roll;
        s_telemetry.raw_valid  = diag->pose_raw.is_valid;
        s_telemetry.smooth_yaw   = diag->pose_smoothed.yaw;
        s_telemetry.smooth_pitch = diag->pose_smoothed.pitch;
        s_telemetry.smooth_roll  = diag->pose_smoothed.roll;
        s_telemetry.smooth_valid = diag->pose_smoothed.is_valid;
        s_telemetry.bias_yaw   = diag->bias_yaw;
        s_telemetry.bias_pitch = diag->bias_pitch;
        s_telemetry.pub_yaw    = diag->pose_published.yaw;
        s_telemetry.pub_pitch  = diag->pose_published.pitch;
        s_telemetry.pub_roll   = diag->pose_published.roll;
        s_telemetry.pub_valid  = diag->pose_published.is_valid;
        s_telemetry.cam_seq    = diag->cam_seq;
        s_telemetry.cap_us     = diag->cap_us;
        s_telemetry.publish_us = diag->publish_us;
        s_telemetry.tensor_scale = diag->input_scale;
        s_telemetry.tensor_zp    = diag->input_zero_point;
        if (diag->tensor && s_diag_tensor &&
            diag->tensor_w == WEB_DIAG_TENSOR_W && diag->tensor_h == WEB_DIAG_TENSOR_H) {
            memcpy(s_diag_tensor, diag->tensor, WEB_DIAG_TENSOR_LEN);
            s_telemetry.tensor_valid = true;
        }
    }

    if (landmarks_22) {
        memcpy(s_telemetry.landmarks, landmarks_22, sizeof(point2d_t) * 22);
    }
    s_telemetry.update_count++;
}

void web_server_update_telemetry(const adas_metrics_t* metrics,
                                const point2d_t landmarks_22[22],
                                int roi_active, int roi_x0, int roi_y0, int roi_size,
                                float dec_ms, float ai_ms, float pnp_ms, float total_ms,
                                uint32_t frame_id, float img_age_ms, float roi_age_ms,
                                float core1_ms) {
    if (!s_telemetry_mutex || !metrics) return;

    if (xSemaphoreTake(s_telemetry_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        telemetry_store_locked(metrics, landmarks_22, roi_active, roi_x0, roi_y0, roi_size,
                               0, 0, dec_ms, ai_ms, pnp_ms, total_ms,
                               frame_id, img_age_ms, roi_age_ms, core1_ms, NULL);
        xSemaphoreGive(s_telemetry_mutex);
    }
}

bool web_server_publish_result(const uint8_t* jpeg_data, size_t jpeg_len,
                               const adas_metrics_t* metrics,
                               const point2d_t landmarks_22[22],
                               int roi_active, int roi_x0, int roi_y0, int roi_size,
                               int src_w, int src_h,
                               float dec_ms, float ai_ms, float pnp_ms, float total_ms,
                               uint32_t frame_id, float img_age_ms, float roi_age_ms,
                               float core1_ms,
                               const web_diag_meta_t* diag) {
    if (!s_frame_mutex || !s_telemetry_mutex || !metrics) return false;

    bool frame_ok = false;
    // Thứ tự khóa: frame -> telemetry (giống debug_snapshot_handler) để
    // /debug/snapshot luôn đọc được cặp (ảnh, landmark) của cùng một frame.
    if (xSemaphoreTake(s_frame_mutex, pdMS_TO_TICKS(20)) == pdTRUE) {
        if (jpeg_data && jpeg_len > 0 && s_stream_frame && jpeg_len <= MAX_STREAM_BUFFER_SIZE) {
            memcpy(s_stream_frame, jpeg_data, jpeg_len);
            s_stream_frame_len = jpeg_len;
            frame_ok = true;
        } else {
            // [COMPARE] Khong co JPEG cua frame nay -> KHONG giu anh cu, neu khong
            // /debug/snapshot se ghep JPEG cu voi telemetry/tensor moi.
            s_stream_frame_len = 0;
        }
        if (xSemaphoreTake(s_telemetry_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            telemetry_store_locked(metrics, landmarks_22, roi_active, roi_x0, roi_y0, roi_size,
                                   src_w, src_h, dec_ms, ai_ms, pnp_ms, total_ms,
                                   frame_id, img_age_ms, roi_age_ms, core1_ms, diag);
            xSemaphoreGive(s_telemetry_mutex);
        }
        xSemaphoreGive(s_frame_mutex);
    }
    return frame_ok;
}

bool web_server_is_recalibrate_requested(void) {
    if (s_recalibrate_flag) {
        s_recalibrate_flag = false;
        return true;
    }
    return false;
}










