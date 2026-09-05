#include "pnp_solver.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#else
#define ESP_LOGI(tag, fmt, ...) printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) fprintf(stderr, "[ERROR][%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

static const char* TAG = "PNP_SOLVER";

#define RAD_TO_DEG(r) ((r) * 57.29577951308232f)
#define DEG_TO_RAD(d) ((d) * 0.017453292519943f)

#define NUM_PNP_POINTS 6

// 6 Anthropometric 3D Keypoint Coordinates (in mm, Nose Tip as Origin)
// Standard Camera Coordinate Frame: X right (+X), Y down (+Y), Z forward (-Z)
// Points: [0] Nose Tip, [1] Chin (+Y down), [2] Left Eye Outer (-X, -Y up), [3] Right Eye Outer (+X, -Y up), [4] Mouth Left (-X, +Y down), [5] Mouth Right (+X, +Y down)
static const float MODEL_3D[NUM_PNP_POINTS][3] = {
    {   0.0f,    0.0f,    0.0f}, // P19: Nose Tip
    {   0.0f,   65.0f,  -35.0f}, // P21: Chin (+Y points down matching image plane v)
    { -43.0f,  -32.0f,  -30.0f}, // P0:  Left Eye Outer Corner (-Y points up matching image plane v)
    {  43.0f,  -32.0f,  -30.0f}, // P9:  Right Eye Outer Corner (-Y points up matching image plane v)
    { -30.0f,   30.0f,  -20.0f}, // P12: Mouth Left Corner (+Y points down)
    {  30.0f,   30.0f,  -20.0f}  // P13: Mouth Right Corner (+Y points down)
};

// Keypoint indices in the 22-landmark output
static const int LANDMARK_MAP[NUM_PNP_POINTS] = {
    19, // Nose Tip
    21, // Chin
    0,  // Left Eye Outer (P0)
    9,  // Right Eye Outer (P9 - symmetric with P0 across nose axis)
    12, // Mouth Left
    13  // Mouth Right
};

// Precomputed Pseudoinverse Matrix A_pinv = (A^T * A)^(-1) * A^T (Shape 3 x 6)
static float s_A_pinv[3][NUM_PNP_POINTS];
static bool s_is_initialized = false;

// Compute 3x3 Matrix Inverse
static bool invert_3x3(const float m[3][3], float inv[3][3]) {
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if (fabsf(det) < 1e-7f) {
        return false;
    }

    float inv_det = 1.0f / det;
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;

    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

    return true;
}

void pnp_solver_init(void) {
    // Compute A^T * A (Shape 3 x 3)
    float AtA[3][3] = {0};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            for (int k = 0; k < NUM_PNP_POINTS; k++) {
                AtA[r][c] += MODEL_3D[k][r] * MODEL_3D[k][c];
            }
        }
    }

    // Invert (A^T * A)
    float AtA_inv[3][3];
    if (!invert_3x3(AtA, AtA_inv)) {
        ESP_LOGE(TAG, "Lỗi tính nghịch đảo ma trận mô hình 3D!");
        return;
    }

    // Compute A_pinv = AtA_inv * A^T (Shape 3 x 6)
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < NUM_PNP_POINTS; c++) {
            s_A_pinv[r][c] = 0.0f;
            for (int k = 0; k < 3; k++) {
                s_A_pinv[r][c] += AtA_inv[r][k] * MODEL_3D[c][k];
            }
        }
    }

    s_is_initialized = true;
    ESP_LOGI(TAG, "Đã khởi tạo bộ giải thuần C++ POSIT PnP thành công!");
}

bool pnp_solve_head_pose(const point2d_t landmarks_22[22], head_pose_t* out_pose) {
    if (!out_pose) return false;
    if (!s_is_initialized) {
        pnp_solver_init();
    }

    // Virtual Camera Intrinsics (focal length f = 96.0 mm, principal point cx = cy = 48.0)
    const float f = 96.0f;
    const float cx = 48.0f;
    const float cy = 48.0f;

    // Centered 2D Coordinates on Image Plane (in pixels)
    float u[NUM_PNP_POINTS];
    float v[NUM_PNP_POINTS];

    for (int i = 0; i < NUM_PNP_POINTS; i++) {
        int lm_idx = LANDMARK_MAP[i];
        // landmarks are [0.0, 1.0], scale to [0, 96]
        u[i] = (landmarks_22[lm_idx].x * 96.0f) - cx;
        v[i] = (landmarks_22[lm_idx].y * 96.0f) - cy;
    }

    // Iterative POSIT Algorithm (Dementhon & Davis 1995)
    float eps[NUM_PNP_POINTS] = {0};
    float r1[3] = {0}, r2[3] = {0}, r3[3] = {0};
    float Tz = 300.0f; // Initial distance estimate (300 mm)

    const int MAX_ITER = 4;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Orthographic projection correction: x_prime = u * (1 + eps)
        float x_prime[NUM_PNP_POINTS];
        float y_prime[NUM_PNP_POINTS];
        for (int i = 0; i < NUM_PNP_POINTS; i++) {
            x_prime[i] = u[i] * (1.0f + eps[i]);
            y_prime[i] = v[i] * (1.0f + eps[i]);
        }

        // Vector I = A_pinv * x_prime, Vector J = A_pinv * y_prime
        float I[3] = {0};
        float J[3] = {0};
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < NUM_PNP_POINTS; c++) {
                I[r] += s_A_pinv[r][c] * x_prime[c];
                J[r] += s_A_pinv[r][c] * y_prime[c];
            }
        }

        // Vector norms
        float s1 = sqrtf(I[0] * I[0] + I[1] * I[1] + I[2] * I[2]);
        float s2 = sqrtf(J[0] * J[0] + J[1] * J[1] + J[2] * J[2]);
        if (s1 < 1e-5f || s2 < 1e-5f) {
            out_pose->is_valid = false;
            return false;
        }

        float s = (s1 + s2) * 0.5f;
        Tz = f / s;

        // Normalized row vectors of Rotation Matrix
        r1[0] = I[0] / s1; r1[1] = I[1] / s1; r1[2] = I[2] / s1;
        r2[0] = J[0] / s2; r2[1] = J[1] / s2; r2[2] = J[2] / s2;

        // Cross product: r3 = r1 x r2
        r3[0] = r1[1] * r2[2] - r1[2] * r2[1];
        r3[1] = r1[2] * r2[0] - r1[0] * r2[2];
        r3[2] = r1[0] * r2[1] - r1[1] * r2[0];

        // Update perspective distortion factor epsilon_i
        for (int i = 0; i < NUM_PNP_POINTS; i++) {
            eps[i] = (r3[0] * MODEL_3D[i][0] + r3[1] * MODEL_3D[i][1] + r3[2] * MODEL_3D[i][2]) / Tz;
        }
    }

    // Gram-Schmidt Orthogonalization to guarantee pure rotation SO(3)
    // r1_norm = normalize(r1)
    float norm_r1 = sqrtf(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
    r1[0] /= norm_r1; r1[1] /= norm_r1; r1[2] /= norm_r1;

    // r2 = normalize(r2 - (r1 . r2) * r1)
    float dot12 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
    r2[0] -= dot12 * r1[0];
    r2[1] -= dot12 * r1[1];
    r2[2] -= dot12 * r1[2];
    float norm_r2 = sqrtf(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]);
    r2[0] /= norm_r2; r2[1] /= norm_r2; r2[2] /= norm_r2;

    // r3 = r1 x r2
    r3[0] = r1[1] * r2[2] - r1[2] * r2[1];
    r3[1] = r1[2] * r2[0] - r1[0] * r2[2];
    r3[2] = r1[0] * r2[1] - r1[1] * r2[0];

    // Extract Euler Angles (Ry(yaw) * Rx(pitch) * Rz(roll) convention)
    // Rotation Matrix R = [r1; r2; r3]
    // r1 = R[0], r2 = R[1], r3 = R[2]
    float pitch_rad = asinf(-fmaxf(-1.0f, fminf(1.0f, r2[2]))); // -R[1][2]
    float yaw_rad = 0.0f, roll_rad = 0.0f;

    if (cosf(pitch_rad) > 1e-4f) {
        yaw_rad = atan2f(r1[2], r3[2]); // R[0][2], R[2][2]
        roll_rad = atan2f(r2[0], r2[1]); // R[1][0], R[1][1]
    } else {
        // Gimbal lock
        yaw_rad = atan2f(-r1[1], r1[0]);
        roll_rad = 0.0f;
    }

    out_pose->yaw = RAD_TO_DEG(yaw_rad);
    out_pose->pitch = RAD_TO_DEG(pitch_rad);
    out_pose->roll = RAD_TO_DEG(roll_rad);
    out_pose->tz = Tz;
    out_pose->tx = u[0] * Tz / f;
    out_pose->ty = v[0] * Tz / f;
    out_pose->is_valid = true;

    return true;
}
