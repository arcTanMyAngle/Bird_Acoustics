#include "model_runner.h"
#include "model_data.h"

#include <cmath>
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
// 64-mel model needs ~262 KB (AllocateTensors). Too large for internal .bss
// (dram0_0_seg overflows), so the arena lives in PSRAM. See arena_used_bytes().
constexpr int kArenaSize = 320 * 1024;
uint8_t *arena = nullptr;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *input, *output;
}  // namespace

extern "C" int model_runner_init(void)
{
    const tflite::Model *model = tflite::GetModel(bird_classifier_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("schema %lu != %d", model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    if (!arena) {
        arena = (uint8_t *)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM);
        if (!arena) {
            MicroPrintf("arena alloc failed (%d B PSRAM)", kArenaSize);
            return -4;
        }
    }

    // Exact op set of the exported graph (MEAN lowers to SUM+MUL)
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddDequantize();
    resolver.AddFullyConnected();
    resolver.AddMaxPool2D();
    resolver.AddMul();
    resolver.AddQuantize();
    resolver.AddReshape();
    resolver.AddSum();
    resolver.AddTranspose();

    static tflite::MicroInterpreter static_interpreter(model, resolver, arena, kArenaSize);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors failed");
        return -2;
    }
    input = interpreter->input(0);
    output = interpreter->output(0);
    if (input->bytes != N_MELS * N_FRAMES || output->bytes != N_CLASSES) {
        MicroPrintf("tensor size mismatch: in %d out %d", input->bytes, output->bytes);
        return -3;
    }
    MicroPrintf("model ready: arena %d/%d B, in scale=%f zp=%d",
                (int)interpreter->arena_used_bytes(), kArenaSize,
                (double)input->params.scale, (int)input->params.zero_point);
    return 0;
}

extern "C" int model_runner_invoke(const float *spec, float *logits_out)
{
    const float inv_scale = 1.0f / input->params.scale;
    const int zp = input->params.zero_point;
    int8_t *in = input->data.int8;
    for (int i = 0; i < N_MELS * N_FRAMES; i++) {
        int q = (int)lrintf(spec[i] * inv_scale) + zp;
        in[i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }

    if (interpreter->Invoke() != kTfLiteOk)
        return -1;

    const float out_scale = output->params.scale;
    const int out_zp = output->params.zero_point;
    for (int c = 0; c < N_CLASSES; c++)
        logits_out[c] = (output->data.int8[c] - out_zp) * out_scale;
    return 0;
}

extern "C" int model_runner_arena_used(void)
{
    return interpreter ? (int)interpreter->arena_used_bytes() : 0;
}
