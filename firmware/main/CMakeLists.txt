idf_component_register(
    SRCS
        main.c
        audio_capture.c
        frontend.c
        frontend_fft_ref.c
        model_runner.cc
        decision.c
        sd_logger.c
        eval_mode.c
    INCLUDE_DIRS . model
    PRIV_REQUIRES esp_driver_i2s esp_driver_spi esp_driver_sdspi fatfs sdmmc vfs esp_timer
)
