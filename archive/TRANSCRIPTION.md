**English** | [简体中文](TRANSCRIPTION.zh-CN.md)

# Source transcription

`source/transcribed/dgemm.cpp` was prepared on 2026-11-01 by comparing the local draft with all 11 code images. It is a transcription of the screenshot listing. Comments and spacing have been normalized. `<cstdint>` is retained from the local draft to define fixed-width types.

The original `.cpp` draft ends after the SME function, lacks closing conditional directives and the outer dgemm implementation, and contains typographical errors. The screenshots provide the remaining dispatcher and entry point and use valid spellings for the NEON loads/stores. These versions are preserved separately.

Source-image map:

- image1: includes and GEMM_P/Q/R environment settings.
- image2–3: packing and scalar operations; AVX microkernel.
- image4: AVX-512 and NEON initialization.
- image5–6: NEON multiply-accumulate/store and SME function.
- image7–8: compile-time architecture branches and environment-controlled dispatch.
- image9: beta/NUMA operations and dgemm entry.
- image10–11: OpenMP scheduling, packing, prefetching, dispatch and cleanup.

The SME path is conditional on `__ARM_FEATURE_SME` at compile time and `USE_SME=1` at execution. It is not runtime CPU feature detection.

SIMD tail handling, packed-panel strides across multiple microtiles, allocation failure handling and the char-versus-enum public signature need attention before using it as a replacement BLAS. The portable 4×4 check covers only the isolated tile math.
