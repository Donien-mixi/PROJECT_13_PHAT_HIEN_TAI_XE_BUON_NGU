file(GLOB_RECURSE ESP_NN_SRCS
    "esp_nn/src/*.c"
    "esp_nn/src/*.S"
)

idf_component_register(
    SRCS
        "main.cpp"
        "wifi_stream_client.cpp"
        "image_decoder.cpp"
        "ai_inference.cpp"
        "pnp_solver.cpp"
        "adas_controller.cpp"
        "telemetry_sender.cpp"
        "esp_nn_glue.cpp"
        ${ESP_NN_SRCS}
    INCLUDE_DIRS
        "."
        "esp_nn/include"
        "esp_nn/src/common"
    REQUIRES
        esp_wifi
        esp_event
        nvs_flash
        esp_psram
        esp_timer
        esp_netif
        lwip
        driver
        mbedtls
        esp_new_jpeg
)

