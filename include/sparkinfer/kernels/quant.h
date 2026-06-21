#pragma once
#include <cuda_runtime.h>

// int8/uint8 spelled as signed/unsigned char so this header is safe to include
// in nvcc's device compilation pass (libstdc++ <cstdint> is not device-parseable).

namespace sparkinfer { namespace kernels {

// Per-tensor symmetric int8 quantize: out = round(in / scale), scale = max|in|/127.
// scale is computed on device and written to *scale (1 float).
void launch_quantize_i8(const void* in_bf16, signed char* out, float* scale, int n,
                        cudaStream_t stream = nullptr);

// Inverse: out = in * scale.
void launch_dequantize_i8(const signed char* in, const float* scale, void* out_bf16, int n,
                          cudaStream_t stream = nullptr);

// Symmetric int4 block dequant. Two 4-bit signed values are packed per byte
// (low nibble first). Each block of `block` values shares one bf16 scale.
//   packed:  [n/2] bytes,  scales: [n/block] bf16,  out: [n] bf16
void launch_dequant_int4_block(const unsigned char* packed, const void* scales_bf16,
                               void* out_bf16, int n, int block,
                               cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
