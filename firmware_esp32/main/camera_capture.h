#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// [14/11 - SPEED] 1 = the OV5640 is configured for YUV422/GRAYSCALE so the pipeline can read the
// Y plane directly and skip the JPEG decode (~0.60 s/frame on QVGA). 0 = hardware JPEG (old path).
#ifndef TD_CAM_RAW_GRAY
#define TD_CAM_RAW_GRAY 0
#endif


// A single captured frame. For the OV5640 we request the sensor's hardware
// JPEG output, so `data` points to a baseline JPEG byte stream.
typedef struct {
    const uint8_t* data;
    size_t length;
    int width;
    int height;
    // [P1 - TRUY VET] So thu tu frame (tang dan theo moi lan acquire thanh cong).
    uint32_t seq;
    // [P1 - TRUY VET] Thoi diem bat dau nhan frame tu DMA (esp_timer, us).
    // Lay tu camera_fb_t.timestamp (cam_hal.c set khi VSYNC frame) -> gan dung
    // thoi diem "chup"; dung de do tuoi anh capture -> publish.
    int64_t timestamp_us;
} camera_frame_t;

// Initializes the onboard OV5640 via the DVP 24-pin interface.
// Pin mapping and format come from Kconfig (TinyDriver ADAS Configuration).
bool camera_capture_init(void);

// Grabs the newest frame from the camera. The returned data pointer stays valid
// until camera_capture_release() is called.
bool camera_capture_acquire(camera_frame_t* out_frame);

// Releases the frame obtained with camera_capture_acquire().
void camera_capture_release(void);

#endif  // CAMERA_CAPTURE_H
