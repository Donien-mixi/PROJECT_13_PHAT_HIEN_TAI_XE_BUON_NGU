// Host replay test for the ADAS FSM (firmware_esp32/main/adas_controller.cpp).
//
// Runs the SAME firmware source on the laptop with a virtual clock, at the two frame
// rates the product uses:
//   * 4,3 FPS  (v3 / Lite: 232 ms/frame, no capture->FSM latency)
//   * 0,12 FPS (PIPNet-98: 8266 ms/frame, 7,7 s capture->FSM latency)
//
// Camera/PnP systematic offset is emulated as SYS_YAW = +50° (what the board actually
// reports when looking straight). Scenarios per rate:
//   S1 open x30      -> calibration must finish, bias≈+50°, state NORMAL
//   S2 closed x8     -> MICROSLEEP at the 1,5 s-equivalent sample (4,3 FPS: 7th, 0,12 FPS: 2nd)
//   S3 turned x12    -> eye sample invalid ("head_turned"), DISTRACTION after ~3 s
//   S4 forward x12   -> distraction clears, NORMAL (no residual eye alarm)
//   S5 yawn x3       -> YAWN_WARNING
//
// Build & run (no board needed):
//   g++ -std=c++17 -O2 -I firmware_esp32/main tools/adas_replay_test.cpp `
//       firmware_esp32/main/adas_controller.cpp -o output/adas_replay_20261113/adas_replay_test.exe
//   output/adas_replay_20261113/adas_replay_test.exe
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "adas_controller.h"

// Defined in the host branch of adas_controller.cpp (C++ linkage).
void adas_host_set_now_us(int64_t t_us);

static const float SYS_YAW = 50.0f;   // systematic PnP offset (bias candidate)
static const float TURN_DEG = 45.0f;  // real head rotation in S3

static void make_landmarks(bool closed, float mar_target, point2d_t lm[22]) {
    const float ear = closed ? 0.100f : 0.200f;
    const float h = 0.15f;
    const float e = ear * h / 2.0f;
    const float cx_l = 0.38f, cx_r = 0.62f, cy = 0.42f;
    lm[0] = {cx_l - h / 2, cy};
    lm[1] = {cx_l - h / 4, cy - e};
    lm[2] = {cx_l + h / 4, cy - e};
    lm[3] = {cx_l + h / 2, cy};
    lm[4] = {cx_l + h / 4, cy + e};
    lm[5] = {cx_l - h / 4, cy + e};
    lm[6] = {cx_r - h / 2, cy};
    lm[7] = {cx_r - h / 4, cy - e};
    lm[8] = {cx_r + h / 4, cy - e};
    lm[9] = {cx_r + h / 2, cy};
    lm[10] = {cx_r + h / 4, cy + e};
    lm[11] = {cx_r - h / 4, cy + e};
    const float w = 0.20f;
    const float h_out = 0.40f * mar_target * w * 2.0f;
    const float h_in = 0.20f * mar_target * w * 2.0f;
    lm[12] = {0.40f, 0.70f};
    lm[13] = {0.60f, 0.70f};
    lm[14] = {0.45f, 0.70f - h_out / 2};
    lm[15] = {0.55f, 0.70f + h_out / 2};
    lm[16] = {0.47f, 0.70f - h_in / 2};
    lm[17] = {0.53f, 0.70f + h_in / 2};
    lm[18] = {0.50f, 0.55f};
    lm[19] = {0.50f, 0.60f};
    lm[20] = {0.50f, 0.80f};
    lm[21] = {0.50f, 0.85f};
}

struct Mode {
    const char* name;
    int64_t frame_us;
    int64_t latency_us;
    int k_micro_expected;   // 1,5 s in samples
};

struct Seg {
    std::string scenario;
    int alarms = 0;
    int first_alarm = -1;
    adas_state_t first_alarm_state = ADAS_STATE_NORMAL;
    adas_state_t last_state = ADAS_STATE_CALIBRATING;
    std::string last_status;
    std::string eye_reason;
};

static Seg run_segment(const Mode& m, int64_t& t_us, const char* name, int frames,
                       bool closed, float mar, float yaw_deg) {
    Seg s;
    s.scenario = name;
    for (int i = 0; i < frames; i++) {
        point2d_t lm[22];
        make_landmarks(closed, mar, lm);
        head_pose_t pose = {};
        pose.yaw = yaw_deg;
        pose.pitch = 0.0f;
        pose.is_valid = true;
        const int64_t capture = t_us;
        adas_host_set_now_us(capture + m.latency_us);
        adas_metrics_t mtr;
        memset(&mtr, 0, sizeof(mtr));
        adas_controller_update(lm, &pose, 1e6f / (float)m.frame_us, &mtr);
        s.last_state = mtr.state;
        s.last_status = mtr.status_str;
        s.eye_reason = mtr.eye.reset_reason ? mtr.eye.reset_reason : "-";
        if (mtr.is_alarm_active) {
            s.alarms++;
            if (s.first_alarm < 0) {
                s.first_alarm = i;
                s.first_alarm_state = mtr.state;
            }
        }
        t_us += m.frame_us;
    }
    return s;
}

static const char* state_name(adas_state_t s) {
    switch (s) {
        case ADAS_STATE_CALIBRATING: return "CALIBRATING";
        case ADAS_STATE_NORMAL: return "NORMAL";
        case ADAS_STATE_SLOW_BLINK_WARNING: return "SLOW_BLINK";
        case ADAS_STATE_MICROSLEEP_ALARM: return "MICROSLEEP";
        case ADAS_STATE_YAWN_WARNING: return "YAWN";
        case ADAS_STATE_FATIGUE_ALARM: return "FATIGUE";
        case ADAS_STATE_DISTRACTION_ALARM: return "DISTRACTION";
    }
    return "?";
}

static int frames_for(const Mode& m, double seconds) {
    int n = (int)(seconds * 1e6 / (double)m.frame_us) + 1;
    return n < 3 ? 3 : n;   // at least 3 samples: one period is the minimum duration
}

int main() {
    const Mode modes[] = {
        {"4.3FPS", 232000, 0, 7},
        {"0.12FPS", 8266000, 7700000, 2},
    };
    int failures = 0;

    printf("mode,scenario,frames,alarms,first_alarm,state,eye_reason\n");
    for (const Mode& m : modes) {
        int64_t t = 1000000000LL;
        adas_host_set_now_us(t);
        adas_controller_init();

        Seg s1 = run_segment(m, t, "S1 open", 30, false, 0.25f, SYS_YAW);
        float bias = adas_controller_get_yaw_bias();
        float thr = 0;
        {
            // ear threshold is reported in metrics; re-read via a no-op call is not possible,
            // so use the calibration log value indirectly: store from last S1 frame.
        }
        Seg s2 = run_segment(m, t, "S2 closed", 8, true, 0.25f, SYS_YAW);
        Seg s3 = run_segment(m, t, "S3 turned", frames_for(m, 5.0), false, 0.25f, SYS_YAW + TURN_DEG);
        Seg s4 = run_segment(m, t, "S4 forward", frames_for(m, 5.0), false, 0.25f, SYS_YAW);
        Seg s5 = run_segment(m, t, "S5 yawn", frames_for(m, 2.0), false, 0.85f, SYS_YAW);
        (void)thr;

        Seg* segs[] = {&s1, &s2, &s3, &s4, &s5};
        for (Seg* s : segs)
            printf("%s,%s,%s,%d,%d,%s,%s\n", m.name, s->scenario.c_str(),
                   "", s->alarms, s->first_alarm, state_name(s->last_state),
                   s->eye_reason.c_str());

        // --- assertions ---
        auto check = [&](const char* what, bool ok) {
            printf("  [%s] %s %s\n", ok ? "PASS" : "FAIL", m.name, what);
            if (!ok) failures++;
        };
        check("calibration bias ~ +50 deg", fabsf(bias - SYS_YAW) < 2.0f);
        check("S1 normal, no alarm", s1.alarms == 0 && s1.last_state == ADAS_STATE_NORMAL);
        check("S2 microsleep at k-th sample",
              s2.first_alarm == m.k_micro_expected - 1 &&
              s2.first_alarm_state == ADAS_STATE_MICROSLEEP_ALARM);
        check("S3 eye sample invalid on head turn", s3.eye_reason == "head_turned");
        check("S3 distraction alarm", s3.last_state == ADAS_STATE_DISTRACTION_ALARM);
        check("S4 recovers to NORMAL", s4.last_state == ADAS_STATE_NORMAL && s4.alarms == 0);
        check("S5 yawn warning", s5.last_state == ADAS_STATE_YAWN_WARNING ||
                                 s5.last_state == ADAS_STATE_FATIGUE_ALARM);
        printf("  bias=%.1f deg (expected %.1f), k_micro=%d, S2 first alarm frame=%d\n\n",
               bias, SYS_YAW, m.k_micro_expected, s2.first_alarm);
    }
    printf("RESULT: %s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
