#ifndef WIFI_STREAM_CLIENT_H_
#define WIFI_STREAM_CLIENT_H_

#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TCP_STREAM_MAGIC 0xAA55AA55
#define MAX_JPEG_BUFFER_SIZE (64 * 1024) // 64 KB per buffer in PSRAM

typedef struct {
    uint8_t* buffer;
    size_t length;
    uint32_t frame_index;
    int64_t timestamp_us;
} frame_buffer_t;

/**
 * Initializes Wi-Fi Station and Double Buffers in Octal PSRAM.
 * @return true on success, false on error.
 */
bool wifi_stream_init(void);

/**
 * FreeRTOS Task running on Core 0 for receiving video stream from Laptop.
 */
void wifi_stream_receiver_task(void* pvParameters);

/**
 * Called by AI Task (Core 1) to acquire the latest ready frame from PSRAM.
 * Blocks up to timeout_ms waiting for a new frame.
 * @param out_frame Pointer to store the frame buffer descriptor.
 * @param timeout_ms Timeout in milliseconds.
 * @return true if a new frame is acquired, false on timeout.
 */
bool wifi_stream_acquire_latest_frame(frame_buffer_t* out_frame, uint32_t timeout_ms);

/**
 * Releases the acquired frame back to the double buffer manager.
 */
void wifi_stream_release_frame(void);

/**
 * Returns whether the TCP connection to the Laptop Host is currently active.
 */
bool wifi_stream_is_connected(void);

#ifdef __cplusplus
}
#endif

#endif // WIFI_STREAM_CLIENT_H_
