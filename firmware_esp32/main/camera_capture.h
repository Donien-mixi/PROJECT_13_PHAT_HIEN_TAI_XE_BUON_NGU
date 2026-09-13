#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// A single captured frame. For the OV5640 we request the sensor's hardware
// JPEG output, so `data` points to a baseline JPEG byte stream.
typedef struct {
    const uint8_t* data;
    size_t length;
    int width;
    int height;
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
