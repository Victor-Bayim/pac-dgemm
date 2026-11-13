# Kunpeng DGEMM Optimization (PAC2025)

FP64 matrix multiplication optimization for Huawei Kunpeng 920.

## Overview

The implementation uses OpenMP task distribution, matrix packing, and a NEON 4×4 microkernel. It was developed for the PAC2025 competition across four matrix shapes with very different dimensions.

## Role

Team member. This repository shows the team implementation and related notes. Individual function authorship is not claimed.

## Key Results (representative shape)

M=12349, N=140, K=3040:

- −O1 serial baseline: 11.07 s
- Optimized: 0.040 s

Full stage records (serial, OpenMP, NEON+Packing, best-of-4) are available in the accompanying verification materials.

## Structure

- `dgemm_final.cpp` — optimized DGEMM implementation (packing + NEON microkernel + OpenMP)
- `reference_gemm_test/` — reference test framework (not original work; preserved for context)

## Notes

- Baseline is −O1 scalar serial.
- OpenMP stage uses −O3 and thread binding.
- Packing benefit is included in the NEON stage; stages are not fully independent ablations.
- `reference_gemm_test/` is framework code supplied with the competition materials.

## License

No explicit distribution license was found for the project-specific files. This snapshot is provided for portfolio reference. Preserve attribution when reusing.
