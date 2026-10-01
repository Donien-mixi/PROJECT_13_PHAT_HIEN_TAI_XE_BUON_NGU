#include "web_server.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
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
    float fps;
    float dec_ms;
    float ai_ms;
    float pnp_ms;
    float total_ms;
    float ear_threshold;
    float mar_threshold;
    float mouth_open_s;
    uint32_t total_yawns;
    uint32_t total_blinks;
    int roi_active;
    int roi_x0;
    int roi_y0;
    int roi_size;
    point2d_t landmarks[22];
    uint32_t update_count;
} web_telemetry_data_t;

static web_telemetry_data_t s_telemetry = {};
static SemaphoreHandle_t    s_telemetry_mutex = NULL;
static volatile bool        s_recalibrate_flag = false;

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

      // Toạ độ 22 điểm mốc được chuẩn hoá [0, 1] trên không gian CROP 96x96
      // Nếu có thông tin ROI (vùng crop trên khung QVGA 320x240)
      const roi = data.roi || { active: 0, x: 0, y: 0, s: 0 };
      let ox = 0, oy = 0, szX = w, szY = h;

      // Nếu crop bám mặt: vẽ khung ROI trên toàn màn hình
      if (roi.active && roi.s > 0) {
        const rx = (roi.x / 320.0) * w;
        const ry = (roi.y / 240.0) * h;
        const rsX = (roi.s / 320.0) * w;
        const rsY = (roi.s / 240.0) * h;
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.45)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(rx, ry, rsX, rsY);
        ctx.setLineDash([]);
        ox = rx; oy = ry; szX = rsX; szY = rsY;
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
        "{\"ear\":%.2f,\"ear_l\":%.2f,\"ear_r\":%.2f,\"mar\":%.2f,"
        "\"yaw\":%.1f,\"pitch\":%.1f,\"roll\":%.1f,\"status\":\"%s\","
        "\"alarm\":%s,\"fps\":%.1f,\"dec\":%.1f,\"ai\":%.1f,\"pnp\":%.1f,\"total\":%.1f,"
        "\"ear_thr\":%.3f,\"mar_thr\":%.3f,\"mouth_s\":%.2f,\"yawns\":%u,\"blinks\":%u,"
        "\"roi\":{\"active\":%d,\"x\":%d,\"y\":%d,\"s\":%d},"
        "\"psram_free\":%u,\"sram_free\":%u,\"landmarks\":[",
        t.ear, t.ear_left, t.ear_right, t.mar,
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
        t.roi_active, t.roi_x0, t.roi_y0, t.roi_size,
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

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    config.max_open_sockets = 7;
    config.stack_size = 8192;
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
    cfg2.stack_size = 6144;
    cfg2.core_id = 0;
    cfg2.lru_purge_enable = true;
    if (httpd_start(&s_server_api, &cfg2) == ESP_OK) {
        httpd_register_uri_handler(s_server_api, &status_uri);
        httpd_register_uri_handler(s_server_api, &recalib_uri);
        ESP_LOGI(TAG, "API telemetry: http://<IP>:81/status  (khong bi stream chan)");
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

void web_server_update_telemetry(const adas_metrics_t* metrics,
                                const point2d_t landmarks_22[22],
                                int roi_active, int roi_x0, int roi_y0, int roi_size,
                                float dec_ms, float ai_ms, float pnp_ms, float total_ms) {
    if (!s_telemetry_mutex || !metrics) return;

    if (xSemaphoreTake(s_telemetry_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
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
        s_telemetry.fps            = metrics->esp32_fps;
        s_telemetry.dec_ms         = dec_ms;
        s_telemetry.ai_ms          = ai_ms;
        s_telemetry.pnp_ms         = pnp_ms;
        s_telemetry.total_ms       = total_ms;
        s_telemetry.ear_threshold  = metrics->ear_threshold;
        s_telemetry.mar_threshold  = metrics->mar_threshold;
        s_telemetry.mouth_open_s   = metrics->mouth_open_s;
        s_telemetry.total_yawns    = metrics->total_yawns;
        s_telemetry.total_blinks   = metrics->total_blinks;
        s_telemetry.roi_active     = roi_active;
        s_telemetry.roi_x0         = roi_x0;
        s_telemetry.roi_y0         = roi_y0;
        s_telemetry.roi_size       = roi_size;

        if (landmarks_22) {
            memcpy(s_telemetry.landmarks, landmarks_22, sizeof(point2d_t) * 22);
        }
        s_telemetry.update_count++;

        xSemaphoreGive(s_telemetry_mutex);
    }
}

bool web_server_is_recalibrate_requested(void) {
    if (s_recalibrate_flag) {
        s_recalibrate_flag = false;
        return true;
    }
    return false;
}
