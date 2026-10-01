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
 */
void web_server_update_telemetry(const adas_metrics_t* metrics,
                                const point2d_t landmarks_22[22],
                                int roi_active, int roi_x0, int roi_y0, int roi_size,
                                float dec_ms, float ai_ms, float pnp_ms, float total_ms);

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
