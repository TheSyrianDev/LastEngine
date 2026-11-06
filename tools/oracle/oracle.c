// Quantizes random data with ik_llama's libggml and dumps quant bytes + dequantized floats for the Rust port tests.
// build: gcc oracle.c -I$IK/ggml/include -L$IK/build/ggml/src -lggml -Wl,-rpath,$IK/build/ggml/src -lm -o oracle
// usage: oracle TYPE_ID OUT_PREFIX    -> OUT_PREFIX.q (bytes), OUT_PREFIX.f (f32), prints "rows cols row_bytes"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
int main(int argc, char** argv) {
    struct ggml_init_params ip = { 1 << 20, NULL, true }; ggml_init(ip); // fills the f16->f32 lookup table the dequantizers use
    int ty = atoi(argv[1]);
    ggml_type_traits_t t = ggml_internal_get_type_traits(ty);
    int rows = 8, cols = 1024;
    if (t.blck_size > 256) cols = t.blck_size * 4;
    float* x = malloc(sizeof(float) * rows * cols); float* im = malloc(sizeof(float) * cols);
    srand(7);
    for (int i = 0; i < rows * cols; i++) { float a = (rand() + 1.f) / (RAND_MAX + 2.f), b = (rand() + 1.f) / (RAND_MAX + 2.f); x[i] = 0.05f * sqrtf(-2 * logf(a)) * cosf(6.2831853f * b); }
    for (int i = 0; i < cols; i++) im[i] = 1.f;
    size_t rb = ggml_row_size(ty, cols);
    char* q = calloc(rows, rb);
    ggml_quantize_chunk(ty, x, q, 0, rows, cols, im, NULL);
    float* y = malloc(sizeof(float) * rows * cols);
    for (int r = 0; r < rows; r++) {
        if (ty == 134 || ty == 135) { // bn: to_float expects the blocks only; the row scale (f16 / f32) sits in front and is applied by the matmul
            const char* p = q + r * rb; float sc;
            if (ty == 134) { unsigned short h; memcpy(&h, p, 2); sc = ggml_fp16_to_fp32(h); } else memcpy(&sc, p, 4);
            t.to_float(p + t.row_meta_size, y + r * cols, cols);
            for (int i = 0; i < cols; i++) y[r * cols + i] *= sc;
        } else t.to_float(q + r * rb, y + r * cols, cols);
    }
    char n[512]; sprintf(n, "%s.q", argv[2]); FILE* f = fopen(n, "wb"); fwrite(q, rb, rows, f); fclose(f);
    sprintf(n, "%s.f", argv[2]); f = fopen(n, "wb"); fwrite(y, 4, rows * cols, f); fclose(f);
    printf("%d %d %zu %s blck=%ld tsz=%zu meta=%ld\n", rows, cols, rb, t.type_name, t.blck_size, t.type_size, t.row_meta_size);
}
