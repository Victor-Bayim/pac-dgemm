**English** | [简体中文](README.zh-CN.md)

# PAC-DGEMM: High-Performance FP64 GEMM on Kunpeng 920

A hand-tuned, production-grade FP64 DGEMM implementation for Huawei Kunpeng 920. It delivers a measured **276×** speedup on the representative contest shape (M=12349, N=140, K=3040), reducing runtime from a serial `-O1` baseline of **11.07 s** to **0.040 s** under the optimized configuration.

## What Was Built

The implementation focuses on the critical path of matrix multiplication on AArch64:

- Consistent micro-panel packing with carefully chosen panel strides to maximize cache reuse and vector load efficiency.
- A 4×4 portable, bounds-checked AArch64 NEON kernel that processes the main computation with guarded edge handling.
- Safe tail processing for arbitrary matrix dimensions, block boundaries, and padded leading dimensions.
- Optional OpenMP parallelization across available cores, with correct work distribution and synchronization.
- A clean column-major, NoTrans API that correctly handles alpha/beta scaling, NaN-safe `beta=0`, overlapping-storage rejection, and full error reporting for invalid dimensions, strides, or pointers.

The code is completely self-contained — no third-party BLAS library or historical SDK is required. Correctness is verified against an independent long-double column-major reference covering rectangular matrices, multiple microtiles, padded strides, empty dimensions, edge cases, and output guards.

## Results

| Shape              | Serial -O1 | Optimized | Speedup |
|--------------------|------------|-----------|---------|
| 12349 × 140 × 3040 | 11.07 s    | 0.040 s   | 276×    |

All four contest shapes and the corresponding 16-stage throughput records are preserved and independently verifiable via `tools/verify_archive.py`. OpenMP runs used `-O3` with 38 bound threads. Packing and NEON effects are combined in the final optimized configuration.

## Build & Verify

```bash
python tools/build_and_test.py
python tools/build_and_test.py --openmp
build/pac-bench --contest
python tools/verify_archive.py
```

## Attribution

Developed as part of the PAC2025 team effort on Kunpeng 920. This repository maintains the optimized implementation together with the complete historical record and verification tooling.
