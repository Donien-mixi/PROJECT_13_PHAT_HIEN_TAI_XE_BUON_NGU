#ifndef FACE_DETECTOR_H_
#define FACE_DETECTOR_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * [D2] Bộ phát hiện khuôn mặt BlazeFace short-range INT8 128x128 (TFLM + esp-nn).
 * Nguồn độc lập với TinyDriverNet (chỉ dùng để XÁC ĐỊNH VÙNG CROP).
 * Chạy THƯA (khởi động / mất dấu) vì khá nặng trên ESP32-S3.
 */

typedef struct {
    float cx;      // tâm mặt (pixel khung gốc)
    float cy;
    float size;    // max(width,height) (pixel khung gốc)
    float score;   // độ tin cậy 0..1
    bool  valid;
    bool  has_kp;      // [D3] có 6 keypoints hợp lệ
    float kp[6][2];    // [D3] 6 điểm: 0,1 = 2 mắt; 2 = chóp mũi; (3 miệng; 4,5 tai) - pixel khung gốc
} face_box_t;

/** Khởi tạo detector (nạp model + arena PSRAM). Trả false nếu lỗi. */
bool face_detector_init(void);

/**
 * Phát hiện khuôn mặt lớn nhất trong khung xám.
 * @param gray   ảnh xám toàn khung (uint8)
 * @param src_w,src_h kích thước khung
 * @param out    hộp mặt (toạ độ khung gốc)
 * @return true nếu tìm thấy (out->valid = true)
 */
bool face_detector_run(const uint8_t* gray, int src_w, int src_h, face_box_t* out);

/** Ngưỡng conf hiện hành (mặc định 0.75). */
void face_detector_set_threshold(float thr);

#ifdef __cplusplus
}
#endif

#endif // FACE_DETECTOR_H_
