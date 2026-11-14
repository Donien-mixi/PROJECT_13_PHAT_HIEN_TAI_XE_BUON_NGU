idf_component_register(
    SRCS
        "main.cpp"
        "wifi_stream_client.cpp"
        "image_decoder.cpp"
        "roi_tracker.cpp"
        "face_detector.cpp"
        "blaze_tracker.cpp"
        "obs_server.cpp"
        "face_detector_lite.cpp"
        "ai_inference.cpp"
        "face_espdl.cpp"
        "face_det_espdl.cpp"
        "camera_capture.cpp"
        "pnp_solver.cpp"
        "adas_controller.cpp"
        "telemetry_sender.cpp"
        "web_server.cpp"
    INCLUDE_DIRS
        "."
    REQUIRES
        esp_wifi
        esp_event
        nvs_flash
        esp_psram
        esp_timer
        esp_netif
        esp_http_server
        lwip
        driver
        mbedtls
        esp_new_jpeg
        esp32-camera
        esp-tflite-micro
        esp-dl
)






