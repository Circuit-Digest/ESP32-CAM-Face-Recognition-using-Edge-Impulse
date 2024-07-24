/**
 * @brief   This program initializes a camera on an ESP32, captures images, and runs inference using a pre-trained model.
 *          The results of the inference, such as object detection or classification, are then displayed on an OLED display.
 *          The program handles various tasks like memory allocation, image capture, and result display.
 *
 * Features:
 * - Initializes the camera and handles errors if initialization fails.
 * - Captures images from the camera and processes them for inference.
 * - Runs a classifier to predict objects or categories in the captured images.
 * - Displays the inference results on an OLED display.
 * - Handles memory allocation and deallocation for image buffers.
 * - Provides debug information if enabled.
 */

/* Includes ---------------------------------------------------------------- */
#include <xxxx.h>  // Include Edge Impulse inferencing header - replace xxxx with your Project Name created in edge impulse.
#include "edge-impulse-sdk/dsp/image/image.hpp"  // Include Edge Impulse image processing

#include "esp_camera.h"  // Include ESP32 camera library

// Select camera model - find more camera models in camera_pins.h file here
// https://github.com/espressif/arduino-esp32/blob/master/libraries/ESP32/examples/Camera/CameraWebServer/camera_pins.h

// Uncomment the correct camera model
// #define CAMERA_MODEL_ESP_EYE // Has PSRAM
#define CAMERA_MODEL_AI_THINKER  // Has PSRAM

// Define the GPIO pins for the selected camera model
#if defined(CAMERA_MODEL_ESP_EYE)
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 4
#define SIOD_GPIO_NUM 18
#define SIOC_GPIO_NUM 23

#define Y9_GPIO_NUM 36
#define Y8_GPIO_NUM 37
#define Y7_GPIO_NUM 38
#define Y6_GPIO_NUM 39
#define Y5_GPIO_NUM 35
#define Y4_GPIO_NUM 14
#define Y3_GPIO_NUM 13
#define Y2_GPIO_NUM 34
#define VSYNC_GPIO_NUM 5
#define HREF_GPIO_NUM 27
#define PCLK_GPIO_NUM 25

#elif defined(CAMERA_MODEL_AI_THINKER)
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

#else
#error "Camera model not selected"
#endif

/* Constant defines -------------------------------------------------------- */
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS 320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS 240
#define EI_CAMERA_FRAME_BYTE_SIZE 3  // 3 bytes per pixel (RGB)

/* I2C and OLED Display Includes ------------------------------------------- */
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ESP32-CAM doesn't have dedicated I2C pins, so we define our own
#define I2C_SDA 15
#define I2C_SCL 14
TwoWire I2Cbus = TwoWire(0);

// Display defines
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &I2Cbus, OLED_RESET);

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false;  // Set to true to see features generated from the raw signal
static bool is_initialised = false;
uint8_t *snapshot_buf;  // Points to the output of the capture

// Camera configuration structure
static camera_config_t camera_config = {
  .pin_pwdn = PWDN_GPIO_NUM,
  .pin_reset = RESET_GPIO_NUM,
  .pin_xclk = XCLK_GPIO_NUM,
  .pin_sscb_sda = SIOD_GPIO_NUM,
  .pin_sscb_scl = SIOC_GPIO_NUM,

  .pin_d7 = Y9_GPIO_NUM,
  .pin_d6 = Y8_GPIO_NUM,
  .pin_d5 = Y7_GPIO_NUM,
  .pin_d4 = Y6_GPIO_NUM,
  .pin_d3 = Y5_GPIO_NUM,
  .pin_d2 = Y4_GPIO_NUM,
  .pin_d1 = Y3_GPIO_NUM,
  .pin_d0 = Y2_GPIO_NUM,
  .pin_vsync = VSYNC_GPIO_NUM,
  .pin_href = HREF_GPIO_NUM,
  .pin_pclk = PCLK_GPIO_NUM,

  // XCLK 20MHz or 10MHz for OV2640 double FPS (Experimental)
  .xclk_freq_hz = 20000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,

  .pixel_format = PIXFORMAT_JPEG,  // YUV422, GRAYSCALE, RGB565, JPEG
  .frame_size = FRAMESIZE_QVGA,    // QQVGA-UXGA. Do not use sizes above QVGA when not JPEG

  .jpeg_quality = 12,  // 0-63 lower number means higher quality
  .fb_count = 1,       // if more than one, i2s runs in continuous mode. Use only with JPEG
  .fb_location = CAMERA_FB_IN_PSRAM,
  .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

/* Function definitions ------------------------------------------------------- */
bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);

/**
 * @brief      Arduino setup function
 */
void setup() {
  Serial.begin(115200);  // Initialize serial communication

  // Initialize I2C with our defined pins
  I2Cbus.begin(I2C_SDA, I2C_SCL, 100000);

  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.printf("SSD1306 OLED display failed to initialize.\nCheck that display SDA is connected to pin %d and SCL connected to pin %d\n", I2C_SDA, I2C_SCL);
    while (true);
  }

  Serial.println("Edge Impulse Inferencing Demo");
  if (ei_camera_init() == false) {
    ei_printf("Failed to initialize Camera!\r\n");
  } else {
    ei_printf("Camera initialized\r\n");
  }

  ei_printf("\nStarting continuous inference in 2 seconds...\n");
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.print("Starting continuous\n inference in\n 2 seconds...");
  display.display();
  ei_sleep(2000);
  display.clearDisplay();
}

/**
 * @brief      Main loop to get data and run inferencing
 *
 * @param[in]  debug  Get debug info if true
 */
void loop() {
    display.clearDisplay();

    // Instead of wait_ms, we'll wait on the signal, this allows threads to cancel us...
    if (ei_sleep(5) != EI_IMPULSE_OK) {
        return;
    }

    // Allocate memory for the snapshot buffer
    snapshot_buf = (uint8_t *)malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_FRAME_BYTE_SIZE);

    // Check if allocation was successful
    if (snapshot_buf == nullptr) {
        ei_printf("ERR: Failed to allocate snapshot buffer!\n");
        return;
    }

    // Signal structure to pass data to the classifier
    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data;

    // Capture image using the camera
    if (ei_camera_capture((size_t)EI_CLASSIFIER_INPUT_WIDTH, (size_t)EI_CLASSIFIER_INPUT_HEIGHT, snapshot_buf) == false) {
        ei_printf("Failed to capture image\r\n");
        free(snapshot_buf);  // Free the allocated memory
        return;
    }

    // Run the classifier on the captured image
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        free(snapshot_buf);  // Free the allocated memory
        return;
    }

    // Print the prediction results
    ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
              result.timing.dsp, result.timing.classification, result.timing.anomaly);

    // Object detection mode
    #if EI_CLASSIFIER_OBJECT_DETECTION == 1
        ei_printf("Object detection bounding boxes:\r\n");
        bool bb_found = result.bounding_boxes[0].value > 0;
        for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
            ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
            if (bb.value == 0) {
                continue;
            }
            ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                      bb.label,
                      bb.value,
                      bb.x,
                      bb.y,
                      bb.width,
                      bb.height);

            // Display results on OLED
            display.setCursor(0, 16 * i);
            display.setTextSize(2);
            display.setTextColor(SSD1306_WHITE);

            if (strcmp(bb.label, "cristiano ronaldo") == 0) {
                display.print("CR7");
            } else if (strcmp(bb.label, "elon musk") == 0) {
                display.print("Elon");
            } else if (strcmp(bb.label, "robert downey jr") == 0) {
                display.print("RDj");
            }

            display.print("-");
            display.print(int((bb.value) * 100));
            display.print("%");
            display.display();
        }
        if (!bb_found) {
            ei_printf("    No objects found\n");
            display.setCursor(0, 2);
            display.setTextSize(2);
            display.setTextColor(SSD1306_WHITE);
            display.print("No objects  found");
            display.display();
        }
    // Classification mode
    #else
        ei_printf("Predictions:\r\n");
        for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
            ei_printf("%.5f\r\n", result.classification[i].value);
        }
    #endif

    // Print anomaly result if it exists
    #if EI_CLASSIFIER_HAS_ANOMALY
        ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
    #endif

    // Print visual anomaly detection results if it exists
    #if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
        ei_printf("Visual anomalies:\r\n");
        for (uint32_t i = 0; i < result.visual_ad_count; i++) {
            ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
            if (bb.value == 0) {
                continue;
            }
            ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                      bb.label,
                      bb.value,
                      bb.x,
                      bb.y,
                      bb.width,
                      bb.height);
        }
    #endif

    // Free the allocated memory
    free(snapshot_buf);
}

/**
 * @brief   Setup image sensor & start streaming
 *
 * @retval  false if initialization failed
 */
bool ei_camera_init(void) {
    if (is_initialised) return true;

    #if defined(CAMERA_MODEL_ESP_EYE)
        pinMode(13, INPUT_PULLUP);
        pinMode(14, INPUT_PULLUP);
    #endif

    // Initialize the camera
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    // Get the sensor and configure settings
    sensor_t *s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);       // Flip the image vertically
        s->set_brightness(s, 1);  // Increase brightness
        s->set_saturation(s, 0);  // Decrease saturation
    }

    #if defined(CAMERA_MODEL_M5STACK_WIDE)
        s->set_vflip(s, 1);
        s->set_hmirror(s, 1);
    #elif defined(CAMERA_MODEL_ESP_EYE)
        s->set_vflip(s, 1);
        s->set_hmirror(s, 1);
        s->set_awb_gain(s, 1);
    #endif

    is_initialised = true;
    return true;
}

/**
 * @brief   Stop streaming of sensor data
 */
void ei_camera_deinit(void) {
    // Deinitialize the camera
    esp_err_t err = esp_camera_deinit();
    if (err != ESP_OK) {
        ei_printf("Camera deinit failed\n");
        return;
    }
    is_initialised = false;
}

/**
 * @brief      Capture, rescale, and crop image
 *
 * @param[in]  img_width     Width of output image
 * @param[in]  img_height    Height of output image
 * @param[in]  out_buf       Pointer to store output image, NULL may be used
 *                           if ei_camera_frame_buffer is to be used for capture and resize/cropping.
 *
 * @retval     false if not initialized, image capture, rescale, or crop failed
 */
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    bool do_resize = false;

    if (!is_initialised) {
        ei_printf("ERR: Camera is not initialized\r\n");
        return false;
    }

    // Capture image frame buffer
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ei_printf("Camera capture failed\n");
        return false;
    }

    // Convert the image format to RGB888
    bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);
    esp_camera_fb_return(fb);

    if (!converted) {
        ei_printf("Conversion failed\n");
        return false;
    }

    // Check if resizing is needed
    if ((img_width != EI_CAMERA_RAW_FRAME_BUFFER_COLS) || (img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        do_resize = true;
    }

    // Resize the image if needed
    if (do_resize) {
        ei::image::processing::crop_and_interpolate_rgb888(
            out_buf,
            EI_CAMERA_RAW_FRAME_BUFFER_COLS,
            EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
            out_buf,
            img_width,
            img_height);
    }

    return true;
}

/**
 * @brief   Get data from the camera for inference
 *
 * @param[in]  offset    Offset in the data buffer
 * @param[in]  length    Length of data to retrieve
 * @param[out] out_ptr   Pointer to store the data
 *
 * @retval  0 on success
 */
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
    // We already have an RGB888 buffer, so recalculate offset into pixel index
    size_t pixel_ix = offset * 3;
    size_t pixels_left = length;
    size_t out_ptr_ix = 0;

    while (pixels_left != 0) {
        // Swap BGR to RGB here
        // due to https://github.com/espressif/esp32-camera/issues/379
        out_ptr[out_ptr_ix] = (snapshot_buf[pixel_ix + 2] << 16) + (snapshot_buf[pixel_ix + 1] << 8) + snapshot_buf[pixel_ix];

        // Go to the next pixel
        out_ptr_ix++;
        pixel_ix += 3;
        pixels_left--;
    }
    // Done!
    return 0;
}

// Ensure the model is compatible with the current sensor
#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor"
#endif
