# Maintenance implementation, 2026-11-15

The public repository previously contained a 26-line placeholder. The new root
`dgemm_final.cpp` is a maintenance reconstruction of the documented
Packing + 4x4 kernel + OpenMP design. It is not represented as the original
submission, a recovered final executable or the source that produced historical
11.07 s / 0.040 s measurements.

## Explicit API and layout

`include/pac_dgemm.h` declares one C-compatible API with fixed-width dimensions:
`pac_dgemm_colmajor`. It supports column-major, non-transposed matrices only,
with arbitrary valid leading dimensions. A/B/C must not overlap. It validates
dimensions, leading dimensions, required pointers and representable spans before
modifying C. The caller still owns buffer sizes and storage validity.

## Corrections from the historical listing

1. Each A micro-panel is `[depth][4 rows]`, and each B micro-panel is
   `[depth][4 columns]`. Both packing and dispatch use the same panel offsets.
2. M/N tails are padded during packing; partial output tiles are computed in a
   local 4x4 array, then only valid elements are copied to C. No fixed-width
   load/store reaches a caller's row or column tail.
3. Beta is applied once, before all K blocks. Beta=0 overwrites old C without
   reading it (including NaNs); alpha=0 and K=0 skip A/B reads.
4. Constant positive block sizes replace unchecked environment variables.
   Workspace allocation happens before output changes and outside OpenMP.
5. Optional OpenMP partitions disjoint row blocks, each with private workspace.
   The serial path has no OpenMP runtime dependency.
6. The portable and guarded AArch64 NEON kernels share the panel contract.
   `PAC_FORCE_SCALAR=1` selects the portable kernel even on AArch64.
7. The new driver uses the same column-major independent reference; every warmup
   and timed round restores C. Warmup is separate; median and minimum are labelled
   separately and all durations are printed.

## Validation scope

The reference suite covers rectangular matrices, multiple microtiles, K blocking,
M/N tails, padded lda/ldb/ldc, alpha/beta combinations, NaN C with beta=0, empty
dimensions, rejected parameters and guard/padding preservation. The four contest
shapes' final block extents are included, not confused with full contest reruns.

The benchmark checks every output on small workloads, and explicitly reports 64
sampled outputs on larger workloads. Sampling is not full correctness proof.
`--contest` selects the four historical dimensions but produces *new maintenance*
measurements on the actual current machine. Its output must never replace archived
CSV values or be described as historical contest data.

No original kblas SDK is required for the maintenance API. No CBLAS compatibility,
arbitrary transpose support, SME execution or competitive BLAS performance is
claimed. A same-machine BLAS baseline and an actual AArch64/NEON/OpenMP run remain
useful target-machine work; a cross-compile alone does not validate execution.
