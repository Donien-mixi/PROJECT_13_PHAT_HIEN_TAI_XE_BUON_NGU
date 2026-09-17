#ifndef ROI_TRACKER_H_
#define ROI_TRACKER_H_

#include <stdbool.h>
#include "pnp_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * [v2.9.0 - FACE ROI CROP]
 * Bám vùng khuôn mặt (ROI vuông) từ 22 landmark của frame trước, theo đúng
 * công thức "canonical anatomical anchor" lúc TRAIN (isomorphic_transform.py),
 * để đầu vào 96x96 là CROP SÁT MẶT (giống laptop/Colab) thay vì crop giữa khung.
 *
 * Mục tiêu: cứu độ nhạy MAR (ngáp) — miệng được phóng đủ lớn như trên laptop.
 * KHÔNG đụng model / EAR / MAR / ADAS / PnP.
 */

typedef struct {
    int x0;      // góc trên-trái (pixel khung gốc)
    int y0;
    int size;    // cạnh vuông (pixel)
} face_roi_t;

/** Xoá trạng thái (gọi khi khởi động / mất mặt). */
void roi_tracker_reset(void);

/** ROI hiện tại có dùng được không (nếu false -> decoder tự crop giữa khung). */
bool roi_tracker_is_valid(void);

/** Lấy ROI hiện tại (đã kẹp trong khung). */
face_roi_t roi_tracker_get(void);

/**
 * Cập nhật ROI từ landmark chuẩn hoá của frame vừa suy luận.
 * @param landmarks  22 điểm (normalized [0,1] so với crop vừa dùng).
 * @param src_w,src_h Kích thước khung đã giải mã (vd 320x240).
 * @param crop_x0,crop_y0,crop_size  Crop đã dùng để tạo ra landmark trên.
 */
void roi_tracker_update_from_landmarks(const point2d_t landmarks[22],
                                       int src_w, int src_h,
                                       int crop_x0, int crop_y0, int crop_size);

/**
 * [D7] NEO ROI từ detector (thay Haar/MediaPipe của laptop) — chống trôi dạt.
 * @param cx,cy,size  Tâm & cạnh crop canonical do detector xác định (pixel khung gốc).
 *                    Khi tracker chưa bám -> đặt cứng; khi đang bám -> hiệu chỉnh qua bộ lọc.
 */
void roi_tracker_anchor(int cx, int cy, int size, int src_w, int src_h);

#ifdef __cplusplus
}
#endif

#endif // ROI_TRACKER_H_
