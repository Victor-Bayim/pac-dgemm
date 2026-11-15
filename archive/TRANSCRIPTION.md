# Source transcription

`source/transcribed/dgemm.cpp` was prepared on 2026-11-01 by comparing the local draft with all 11 code images. It is a transcription of the screenshot listing, not the recovered original submission file. Comments and spacing have been normalized. `<cstdint>` is retained from the local draft to define fixed-width types. No claim of a successful build or benchmark is attached to this transcription.

The original `.cpp` draft ends after the SME function, lacks closing conditional directives and the outer dgemm implementation, and contains typographical errors. The screenshots provide the remaining dispatcher and entry point and use valid spellings for the NEON loads/stores. These versions are preserved separately.

Source-image map:

- image1: includes and GEMM_P/Q/R environment settings.
- image2–3: packing and scalar operations; AVX microkernel.
- image4: AVX-512 and NEON initialization.
- image5–6: NEON multiply-accumulate/store and SME function.
- image7–8: compile-time architecture branches and environment-controlled dispatch.
- image9: beta/NUMA operations and dgemm entry.
- image10–11: OpenMP scheduling, packing, prefetching, dispatch and cleanup.

The SME path is conditional on `__ARM_FEATURE_SME` at compile time and `USE_SME=1` at execution. It is not runtime CPU feature detection and does not establish SME execution on the competition machine.

The screenshot listing does not establish a safe general-purpose GEMM: SIMD tail handling, packed-panel strides across multiple microtiles, allocation failure handling and the char-versus-enum public signature need attention before using it as a replacement BLAS. These assumptions are deliberately retained in the transcript rather than silently rewritten and presented as historical code. The portable 4×4 check covers only the isolated tile math.
