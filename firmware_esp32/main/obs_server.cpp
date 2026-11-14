// [14/11] Minimal observation server for the Project 13 board.
//
// Why a new module instead of the old web_server.cpp: that file was written for the TinyDriver
// pipeline (telemetry + MJPEG) and pulls in a lot of dead state. This one only needs the frame the
// AI just processed, so it stays ~300 lines and cannot perturb the inference path.
//
// Credentials come from wifi_secret.h (gitignored). Nothing here logs the password.

#include "obs_server.h"

#include <string.h>
#include <stdio.h>

#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#if __has_include("wifi_secret.h")
#include "wifi_secret.h"
#else
#warning "wifi_secret.h missing: create firmware_esp32/main/wifi_secret.h with TD_WIFI_SSID/PASSWORD"
#define TD_WIFI_SSID     "changeme"
#define TD_WIFI_PASSWORD ""
#endif

static const char* TAG = "OBS";

#define OBS_W 320
#define OBS_H 240
#define BMP_HEADER_BYTES 54
#define BMP_PALETTE_BYTES (256 * 4)
#define BMP_ROW_BYTES ((OBS_W + 3) & ~3)
#define BMP_IMAGE_BYTES (BMP_ROW_BYTES * OBS_H)
#define BMP_TOTAL (BMP_HEADER_BYTES + BMP_PALETTE_BYTES + BMP_IMAGE_BYTES)

static uint8_t* s_bmp = nullptr;      // PSRAM: the served image (header + palette + pixels)
static uint8_t* s_send = nullptr;     // PSRAM: copy handed to the HTTP task while sending
static SemaphoreHandle_t s_lock = nullptr;
static httpd_handle_t s_httpd = nullptr;
static volatile bool s_have_frame = false;

// metrics published by the AI task
static struct {
    float ear, ear_thr, ear_base, mar, mar_thr, yaw, pitch, yaw_raw, pitch_raw;
    float yaw_bias, pitch_bias, det_score, det_ms, fps;
    int eye_closed, eye_valid, state, calibrated;
    uint32_t frame;
    char reset_reason[24];
    char status[48];
} s_m;

static void bmp_init_header(uint8_t* p) {
    const uint32_t off = BMP_HEADER_BYTES + BMP_PALETTE_BYTES;
    const uint32_t size = BMP_TOTAL;
    memset(p, 0, BMP_HEADER_BYTES);
    p[0] = 'B'; p[1] = 'M';
    memcpy(p + 2, &size, 4);
    memcpy(p + 10, &off, 4);
    const uint32_t hdr = 40;
    memcpy(p + 14, &hdr, 4);
    const int32_t w = OBS_W, h = OBS_H;
    memcpy(p + 18, &w, 4);
    memcpy(p + 22, &h, 4);
    const uint16_t planes = 1, bpp = 8;
    memcpy(p + 26, &planes, 2);
    memcpy(p + 28, &bpp, 2);
    const uint32_t comp = 0, img = BMP_IMAGE_BYTES;
    memcpy(p + 30, &comp, 4);
    memcpy(p + 34, &img, 4);
    const int32_t ppm = 2835;
    memcpy(p + 38, &ppm, 4);
    memcpy(p + 42, &ppm, 4);
    const uint32_t colors = 256;
    memcpy(p + 46, &colors, 4);
    // greyscale palette
    for (int i = 0; i < 256; i++) {
        uint8_t* e = p + BMP_HEADER_BYTES + i * 4;
        e[0] = e[1] = e[2] = (uint8_t)i;
        e[3] = 0;
    }
}

// BMP rows are bottom-up; draw with (x, y_from_top).
static inline void put(uint8_t* pix, int x, int y) {
    if (x < 0 || x >= OBS_W || y < 0 || y >= OBS_H) return;
    pix[(OBS_H - 1 - y) * BMP_ROW_BYTES + x] = 255;
}

static void draw_overlay(uint8_t* pix, const image_crop_roi_t* roi, const float pts[98][2]) {
    if (roi && roi->size > 1) {
        const int x0 = roi->x0, y0 = roi->y0;
        const int x1 = roi->x0 + roi->size - 1, y1 = roi->y0 + roi->size - 1;
        for (int x = x0; x <= x1; x++) { put(pix, x, y0); put(pix, x, y1); }
        for (int y = y0; y <= y1; y++) { put(pix, x0, y); put(pix, x1, y); }
    }
    if (pts) {
        for (int k = 0; k < 98; k++) {
            const int px = (int)(pts[k][0] + 0.5f), py = (int)(pts[k][1] + 0.5f);
            put(pix, px, py); put(pix, px + 1, py); put(pix, px, py + 1);
        }
    }
}

/* ------------------------------------------------------------------ HTTP */

static const char* STATE_NAMES[] = {
    "CALIBRATING", "NORMAL", "SLOW_BLINK_WARNING", "MICROSLEEP_ALARM",
    "YAWN_WARNING", "FATIGUE_ALARM", "DISTRACTION_ALARM"
};

static esp_err_t h_status(httpd_req_t* req) {
    char buf[800];
    xSemaphoreTake(s_lock, portMAX_DELAY);
    const int st = (s_m.state >= 0 && s_m.state <= 6) ? s_m.state : 0;
    int n = snprintf(buf, sizeof(buf),
        "{\"frame\":%u,\"fps\":%.2f,\"calibrated\":%d,"
        "\"ear\":%.3f,\"ear_threshold\":%.3f,\"ear_baseline\":%.3f,"
        "\"mar\":%.3f,\"mar_threshold\":%.3f,"
        "\"yaw\":%.1f,\"yaw_raw\":%.1f,\"yaw_bias\":%.1f,"
        "\"pitch\":%.1f,\"pitch_raw\":%.1f,\"pitch_bias\":%.1f,"
        "\"eye_valid\":%d,\"eye_closed\":%d,\"eye_reset\":\"%s\","
        "\"detector_score\":%.3f,\"detector_ms\":%.0f,"
        "\"state\":\"%s\",\"status\":\"%s\",\"has_frame\":%d}",
        (unsigned)s_m.frame, (double)s_m.fps, s_m.calibrated,
        (double)s_m.ear, (double)s_m.ear_thr, (double)s_m.ear_base,
        (double)s_m.mar, (double)s_m.mar_thr,
        (double)s_m.yaw, (double)s_m.yaw_raw, (double)s_m.yaw_bias,
        (double)s_m.pitch, (double)s_m.pitch_raw, (double)s_m.pitch_bias,
        s_m.eye_valid, s_m.eye_closed, s_m.reset_reason,
        (double)s_m.det_score, (double)s_m.det_ms,
        STATE_NAMES[st], s_m.status, s_have_frame ? 1 : 0);
    xSemaphoreGive(s_lock);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, buf, n);
}

// [14/11] Button "Hieu chuan lai": re-runs the per-driver baseline (pose bias + EAR/MAR window).
static esp_err_t h_recalib(httpd_req_t* req) {
    adas_controller_request_recalibration();
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true,\"calibrating\":true}");
}

static esp_err_t h_snapshot(httpd_req_t* req) {
    if (!s_have_frame) {
        httpd_resp_set_status(req, "503 Service Unavailable");
        return httpd_resp_sendstr(req, "no frame yet");
    }
    xSemaphoreTake(s_lock, portMAX_DELAY);
    memcpy(s_send, s_bmp, BMP_TOTAL);
    xSemaphoreGive(s_lock);
    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, (const char*)s_send, BMP_TOTAL);
}

static esp_err_t h_root(httpd_req_t* req) {
    // [14/11] Alarm-first page: a big colour banner + tab title + optional beep so a yawn or a
    // closed-eyes episode is impossible to miss while watching the board from a phone.
    static const char page[] =
        "<!doctype html><html><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Project 13</title><style>"
        "body{background:#0d1117;color:#e6edf3;font-family:system-ui;margin:0;padding:14px}"
        "#banner{border-radius:12px;padding:18px 14px;text-align:center;font-size:30px;font-weight:800;"
        "letter-spacing:.5px;transition:background .25s}"
        "#sub{text-align:center;font-size:14px;opacity:.85;margin:6px 0 14px}"
        "img{image-rendering:pixelated;width:100%;max-width:640px;display:block;margin:0 auto;"
        "border:1px solid #30363d;border-radius:8px}"
        ".row{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:12px;font-size:15px}"
        ".kv{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px 12px}"
        ".bar{height:14px;background:#21262d;border-radius:7px;margin-top:6px;position:relative}"
        ".fill{height:100%;background:#2ea043;transition:width .3s;border-radius:7px}"
        ".tick{position:absolute;top:-3px;width:2px;height:20px;background:#e6edf3;opacity:.85}"
        ".sm{font-size:11px;color:#8b949e}"
        "#log{margin-top:14px;font-size:13px;color:#8b949e;white-space:pre-line}"
        "button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:8px;"
        "padding:8px 14px;font-size:14px}"
        "@keyframes bl{50%{opacity:.35}} .blink{animation:bl .7s infinite}"
        "</style></head><body>"
        "<div id=banner style='background:#21262d'>ĐANG KHỞI ĐỘNG…</div>"
        "<div id=sub>chờ dữ liệu từ ESP32</div>"
        "<img id=im src='/snapshot.bmp'>"
        "<div class=row>"
        "<div class=kv>EAR <b id=ear>–</b> / ngưỡng <b id=thr>–</b> (baseline <b id=base>–</b>)"
        "<div class=bar><div id=earbar class=fill style='width:0%'></div><div id=eartick class=tick style='left:0%'></div></div></div>"
        "<div class=kv>MAR <b id=mar>–</b> / ngưỡng <b id=marthr>–</b>"
        "<div class=bar><div id=marbar class=fill style='width:0%'></div><div id=martick class=tick style='left:0%'></div></div></div>"
        "<div class=kv>detector <b id=det>–</b> · <b id=dms>–</b> ms</div>"
        "<div class=kv>frame <b id=fr>–</b> · <b id=fps>–</b> FPS</div>"
        "<div class=kv>yaw <b id=yaw>–</b>° <span class=sm>(raw <span id=yawraw>–</span>°, bias <span id=yawb>–</span>°)</span></div>"
        "<div class=kv>pitch <b id=pitch>–</b>° <span class=sm>(raw <span id=pitchraw>–</span>°, bias <span id=pitchb>–</span>°)</span></div>"
        "</div>"
        "<div class=row><button id=snd>🔔 Bật âm thanh cảnh báo</button>"
        "<button id=cal>↻ Hiệu chuẩn lại (ngồi thẳng, nhìn camera)</button></div>"
        "<div id=log>chưa có cảnh báo</div>"
        "<script>"
        "let beepOn=false,ac=null,lastKey='',hist=[];"
        "snd.onclick=()=>{beepOn=!beepOn;ac=ac||new (window.AudioContext||window.webkitAudioContext)();"
        "snd.textContent=beepOn?'🔕 Tắt âm thanh':'🔔 Bật âm thanh cảnh báo';};"
        "cal.onclick=async()=>{cal.textContent='⏳ đang hiệu chuẩn…';"
        "await fetch('/recalib',{method:'POST'});setTimeout(()=>cal.textContent="
        "'↻ Hiệu chuẩn lại (ngồi thẳng, nhìn camera)',1500);};"
        "function beep(n){if(!beepOn||!ac)return;for(let i=0;i<n;i++){const o=ac.createOscillator(),g=ac.createGain();"
        "o.frequency.value=880;o.connect(g);g.connect(ac.destination);const t=ac.currentTime+i*0.35;"
        "g.gain.setValueAtTime(0.25,t);g.gain.exponentialRampToValueAtTime(0.001,t+0.3);o.start(t);o.stop(t+0.3);}}"
        "const C={CALIBRATING:['#30363d','ĐANG HIỆU CHUẨN','hãy ngồi vào khung, 55–60 cm'],"
        "NORMAL:['#1f6f3f','BÌNH THƯỜNG','mắt mở, đầu thẳng'],"
        "SLOW_BLINK_WARNING:['#8a6d00','MỞ/NHẮM CHẬM','mắt bắt đầu chậm'],"
        "MICROSLEEP_ALARM:['#b91c1c','⚠ BUỒN NGỦ — MẮT NHẮM','giữ mắt mở!'],"
        "FATIGUE_ALARM:['#b91c1c','⚠ MỆT MỎI','nghỉ ngơi!'],"
        "YAWN_WARNING:['#a16207','⚠ NGÁP','đã phát hiện ngáp'],"
        "DISTRACTION_ALARM:['#6d28d9','⚠ MẤT TẬP TRUNG','nhìn về phía trước']};"
        "async function tick(){try{"
        "const r=await fetch('/status',{cache:'no-store'});const j=await r.json();"
        "const[c,t,s]=C[j.state]||C.NORMAL;const b=document.getElementById('banner');"
        "b.style.background=c;b.textContent=t;document.getElementById('sub').textContent=s+' · '+j.status;"
        "const alarm=j.state.includes('ALARM')||j.state==='YAWN_WARNING';"
        "b.className=alarm?'blink':'';document.title=alarm?('⚠ '+t):'Project 13';"
        "if(alarm){beep(j.state.includes('ALARM')?2:1);}"
        "if(alarm&&lastKey!==j.state+hist.length){hist.unshift(new Date().toLocaleTimeString()+'  '+t);"
        "hist=hist.slice(0,6);document.getElementById('log').textContent=hist.join('\\n');lastKey=j.state+hist.length;}"
        "else if(!alarm){document.getElementById('log').textContent=hist.length?hist.join('\\n'):'chưa có cảnh báo';}"
        "ear.textContent=j.ear.toFixed(3);thr.textContent=j.ear_threshold.toFixed(3);"
        "base.textContent=j.ear_baseline.toFixed(3);"
        "mar.textContent=j.mar.toFixed(3);marthr.textContent=j.mar_threshold.toFixed(3);"
        "det.textContent=j.detector_score.toFixed(3);dms.textContent=j.detector_ms.toFixed(0);"
        "fr.textContent=j.frame;fps.textContent=j.fps.toFixed(2);"
        "yaw.textContent=j.yaw.toFixed(1);yawraw.textContent=j.yaw_raw.toFixed(1);yawb.textContent=j.yaw_bias.toFixed(1);"
        "pitch.textContent=j.pitch.toFixed(1);pitchraw.textContent=j.pitch_raw.toFixed(1);pitchb.textContent=j.pitch_bias.toFixed(1);"
        // EAR bar: full scale 0.40 (a normal open eye is 0.17-0.25, so ~50% is EXPECTED);
        // the white tick marks the calibrated threshold.
        "earbar.style.width=Math.min(100,100*j.ear/0.40)+'%';"
        "earbar.style.background=j.ear>=j.ear_threshold?'#2ea043':'#d29922';"
        "eartick.style.left=Math.min(100,100*j.ear_threshold/0.40)+'%';"
        // MAR bar: full scale 1.20; tick at the calibrated yawn threshold.
        "marbar.style.width=Math.min(100,100*j.mar/1.20)+'%';"
        "marbar.style.background=j.mar>j.mar_threshold?'#d29922':'#2ea043';"
        "martick.style.left=Math.min(100,100*j.mar_threshold/1.20)+'%';"
        "im.src='/snapshot.bmp?t='+Date.now();"
        "}catch(e){banner.textContent='mất kết nối tới ESP32';banner.style.background='#30363d';}}"
        "tick();setInterval(tick,1500);"
        "</script></body></html>";
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, page, sizeof(page) - 1);
}

static bool start_http(void) {
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.max_uri_handlers = 6;
    cfg.lru_purge_enable = true;
    cfg.stack_size = 8192;
    cfg.task_priority = 3;
    if (httpd_start(&s_httpd, &cfg) != ESP_OK) {
        ESP_LOGE(TAG, "httpd_start failed (sram free=%u KB)",
                 (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));
        return false;
    }
    httpd_uri_t u_root = {.uri = "/", .method = HTTP_GET, .handler = h_root};
    httpd_uri_t u_stat = {.uri = "/status", .method = HTTP_GET, .handler = h_status};
    httpd_uri_t u_snap = {.uri = "/snapshot.bmp", .method = HTTP_GET, .handler = h_snapshot};
    httpd_uri_t u_cal = {.uri = "/recalib", .method = HTTP_POST, .handler = h_recalib};
    httpd_register_uri_handler(s_httpd, &u_root);
    httpd_register_uri_handler(s_httpd, &u_stat);
    httpd_register_uri_handler(s_httpd, &u_snap);
    httpd_register_uri_handler(s_httpd, &u_cal);
    return true;
}

static void wifi_event(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi mat ket noi, thu lai...");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* ev = (ip_event_got_ip_t*)data;
        ESP_LOGI(TAG, "================ MO http://%d.%d.%d.%d/ de xem board ================",
                 IP2STR(&ev->ip_info.ip));
    }
}

bool obs_server_start(void) {
    s_bmp = (uint8_t*)heap_caps_malloc(BMP_TOTAL, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    s_send = (uint8_t*)heap_caps_malloc(BMP_TOTAL, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_bmp || !s_send) {
        ESP_LOGE(TAG, "khong cap duoc buffer BMP trong PSRAM");
        return false;
    }
    bmp_init_header(s_bmp);
    s_lock = xSemaphoreCreateMutex();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event, nullptr, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event, nullptr, nullptr));

    wifi_config_t wc = {};
    strncpy((char*)wc.sta.ssid, TD_WIFI_SSID, sizeof(wc.sta.ssid) - 1);
    strncpy((char*)wc.sta.password, TD_WIFI_PASSWORD, sizeof(wc.sta.password) - 1);
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    bool open = (TD_WIFI_PASSWORD[0] == '\0');
    if (open) wc.sta.threshold.authmode = WIFI_AUTH_OPEN;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "Wi-Fi STA ssid=%s (pass %s) | sram free=%u KB", TD_WIFI_SSID,
             open ? "OPEN" : "set", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >> 10));

    if (!start_http()) return false;
    ESP_LOGI(TAG, "HTTP server san sang: / , /status , /snapshot.bmp");
    return true;
}

/* -------------------------------------------------------------- publish */

void obs_server_publish(const uint8_t* gray, int w, int h,
                        const image_crop_roi_t* roi,
                        const float pts98[98][2],
                        const adas_metrics_t* metrics,
                        float det_score, float det_ms,
                        uint32_t frame_idx) {
    if (!s_bmp || !s_lock || !gray || w <= 0 || h <= 0) return;
    uint8_t* pix = s_bmp + BMP_HEADER_BYTES + BMP_PALETTE_BYTES;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    // copy, bottom-up
    for (int y = 0; y < OBS_H && y < h; y++) {
        const uint8_t* src = gray + (size_t)(h - 1 - y) * (size_t)w;
        memcpy(pix + (size_t)y * BMP_ROW_BYTES, src, (size_t)(w < OBS_W ? w : OBS_W));
    }
    draw_overlay(pix, roi, pts98);
    if (metrics) {
        s_m.ear = metrics->ear;
        s_m.ear_thr = metrics->ear_threshold;
        s_m.mar = metrics->mar;
        s_m.mar_thr = metrics->mar_threshold;
        s_m.yaw = metrics->pose.yaw;
        s_m.pitch = metrics->pose.pitch;
        s_m.eye_valid = metrics->eye.valid ? 1 : 0;
        s_m.eye_closed = metrics->eye.closed ? 1 : 0;
        s_m.state = (int)metrics->state;
        s_m.fps = metrics->esp32_fps;
        // raw pose = corrected pose (metrics already had the bias subtracted) + the bias itself
        s_m.yaw_bias = adas_controller_get_yaw_bias();
        s_m.pitch_bias = adas_controller_get_pitch_bias();
        s_m.yaw_raw = metrics->pose.yaw + s_m.yaw_bias;
        s_m.pitch_raw = metrics->pose.pitch + s_m.pitch_bias;
        s_m.ear_base = adas_controller_get_ear_baseline();
        s_m.calibrated = adas_controller_is_calibrated() ? 1 : 0;
        snprintf(s_m.status, sizeof(s_m.status), "%s", metrics->status_str);
        snprintf(s_m.reset_reason, sizeof(s_m.reset_reason), "%s",
                 metrics->eye.reset_reason ? metrics->eye.reset_reason : "-");
    }
    s_m.det_score = det_score;
    s_m.det_ms = det_ms;
    s_m.frame = frame_idx;
    s_have_frame = true;
    xSemaphoreGive(s_lock);
}
