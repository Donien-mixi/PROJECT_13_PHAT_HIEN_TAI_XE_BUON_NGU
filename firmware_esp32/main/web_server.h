#ifndef WEB_SERVER_H_
#define WEB_SERVER_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "adas_controller.h"
#include "pnp_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Khởi tạo và chạy HTTP Web Server nhúng trên cổng 80.
 * Cung cấp giao diện Dashboard tại '/', luồng video MJPEG tại '/stream',
 * và dữ liệu thời gian thực dạng JSON tại '/status'.
 * @return true nếu thành công, false nếu thất bại.
 */
bool web_server_init(void);

/**
 * @brief Cập nhật khung hình JPEG mới nhất để phục vụ luồng video web '/stream'.
 * An toàn khi gọi từ bất kỳ tác vụ hoặc nhân nào (Core 0 / Core 1).
 *
 * @param jpeg_data Con trỏ tới vùng đệm JPEG.
 * @param jpeg_len Kích thước vùng đệm JPEG (bytes).
 */
void web_server_update_frame(const uint8_t* jpeg_data, size_t jpeg_len);

/**
 * @brief Cập nhật số liệu telemetry, trạng thái ADAS và 22 điểm mốc khuôn mặt.
 *
 * @param metrics Con trỏ tới struct chỉ số ADAS hiện hành.
 * @param landmarks_22 Mảng 22 điểm mốc chuẩn hoá [0.0, 1.0].
 * @param roi_active Chế độ crop (0: giữa khung, 1: cố định, 2: ROI bám mặt).
 * @param roi_x0 Toạ độ X vùng crop.
 * @param roi_y0 Toạ độ Y vùng crop.
 * @param roi_size Kích thước vùng crop.
 * @param dec_ms Thời gian giải mã ảnh (ms).
 * @param ai_ms Thời gian suy luận mô hình AI (ms).
 * @param pnp_ms Thời gian giải POSIT Head Pose 3D (ms).
 * @param total_ms Tổng thời gian xử lý chu kỳ (ms).
 * @param frame_id [P1] ID frame nguồn (camera seq / decode counter).
 * @param img_age_ms [P1] Tuổi ảnh tại thời điểm publish (capture -> publish, ms).
 * @param roi_age_ms [P1] Tuổi ROI khi áp dụng cho frame (ROI tạo -> capture, ms).
 * @param core1_ms [P1] Thời gian Core 1 xử lý frame (nhận slot -> publish, ms).
 */
void web_server_update_telemetry(const adas_metrics_t* metrics,
                                const point2d_t landmarks_22[22],
                                int roi_active, int roi_x0, int roi_y0, int roi_size,
                                float dec_ms, float ai_ms, float pnp_ms, float total_ms,
                                uint32_t frame_id, float img_age_ms, float roi_age_ms,
                                float core1_ms);

/** Nguồn ROI (ghi vào /debug/snapshot) */
typedef enum {
    WEB_ROI_SRC_CENTER          = 0,  // crop giữa khung
    WEB_ROI_SRC_FIXED           = 1,  // crop cố định theo % (Kconfig)
    WEB_ROI_SRC_DETECTOR_PREV   = 2,  // detector của frame trước
    WEB_ROI_SRC_TRACKER         = 3,  // roi_tracker fallback
    WEB_ROI_SRC_DETECTOR_SAME   = 4,  // detector chạy trên CHÍNH frame này (P4)
} web_roi_source_t;

/**
 * @brief [COMPARE] Metadata chẩn đoán đi kèm một kết quả publish.
 * Tất cả trường thuộc CÙNG frame với JPEG/landmark. Có thể truyền NULL.
 * tensor: snapshot INT8 96x96 TRƯỚC Invoke (byte gốc), NULL nếu không thu.
 */
typedef struct {
    const int8_t* tensor;
    int      tensor_w;
    int      tensor_h;
    float    input_scale;
    int32_t  input_zero_point;
    int      roi_source;        // web_roi_source_t
    bool     face_valid;
    float    face_score;        // 0 nếu không có score
    float    face_cx, face_cy, face_size;   // pixel khung gốc
    head_pose_t pose_raw;       // trước EMA (từ landmark frame này)
    head_pose_t pose_smoothed;  // sau EMA, input của ADAS
    head_pose_t pose_published; // sau trừ bias (giá trị web/telemetry)
    float    bias_yaw, bias_pitch;          // bias hiệu chuẩn (độ)
    uint32_t cam_seq;           // sequence camera (nếu có)
    int64_t  cap_us;            // esp_timer lúc capture
    int64_t  publish_us;        // esp_timer lúc publish
} web_diag_meta_t;

/**
 * @brief [P3] Publish ảnh JPEG + telemetry + 22 landmark của CÙNG một frame.
 *
 * Khác web_server_update_frame()/web_server_update_telemetry(): hai hàm cũ được
 * gọi riêng nên /debug/snapshot có thể đọc ảnh frame N với landmark frame N+1.
 * Hàm này cập nhật cả hai dưới cùng một khóa (frame -> telemetry) để endpoint
 * /debug/snapshot luôn trả về một cặp ảnh <-> landmark nhất quán.
 *
 * @param jpeg_data  Ảnh JPEG nguồn (NULL nếu frame này không có ảnh).
 * @param jpeg_len   Độ dài JPEG (0 nếu không có ảnh).
 * @param src_w/src_h Kích thước ảnh gốc (px) để trình duyệt map tọa độ.
 * @param diag Metadata chẩn đoán của CÙNG frame (có thể NULL).
 * @return true nếu ảnh được cập nhật (telemetry vẫn được cập nhật nếu ảnh NULL).
 */
bool web_server_publish_result(const uint8_t* jpeg_data, size_t jpeg_len,
                               const adas_metrics_t* metrics,
                               const point2d_t landmarks_22[22],
                               int roi_active, int roi_x0, int roi_y0, int roi_size,
                               int src_w, int src_h,
                               float dec_ms, float ai_ms, float pnp_ms, float total_ms,
                               uint32_t frame_id, float img_age_ms, float roi_age_ms,
                               float core1_ms,
                               const web_diag_meta_t* diag);

/**
 * @brief Kiểm tra xem người dùng có vừa bấm nút "Hiệu Chuẩn Lại" trên Web UI không.
 * Tự động xoá cờ sau khi đọc.
 * @return true nếu có yêu cầu hiệu chuẩn lại.
 */
bool web_server_is_recalibrate_requested(void);

#ifdef __cplusplus
}
#endif

#endif // WEB_SERVER_H_
