**English** | [简体中文](README.zh-CN.md)

# PAC-DGEMM: High-Performance FP64 GEMM on Kunpeng 920

A hand-tuned, production-grade FP64 DGEMM for Huawei Kunpeng 920. On the representative contest shape (M=12349, N=140, K=3040), it reduces runtime from a serial `-O1` baseline of **11.07 s** to **0.040 s** under the optimized configuration — a measured **276×** speedup.

## Core Techniques

- Consistent micro-panel packing with carefully chosen panel strides
- 4×4 portable, bounds-checked AArch64 NEON kernel
- Safe tail handling for arbitrary matrix dimensions and block boundaries
- Optional OpenMP parallelization across available cores
- Full column-major, NoTrans API supporting padded leading dimensions, alpha/beta scaling, and NaN-safe `beta=0`

The implementation is completely self-contained — no third-party BLAS, no historical SDK. Correctness is verified against an independent long-double column-major reference that covers rectangular matrices, multiple microtiles, padded strides, edge cases, empty dimensions, and output guards.

## Results

| Shape              | Serial -O1 | Optimized | Speedup |
|--------------------|------------|-----------|---------|
| 12349 × 140 × 3040 | 11.07 s    | 0.040 s   | 276×    |

All four contest shapes and the corresponding 16-stage throughput records are preserved and independently verifiable. OpenMP runs used `-O3` with 38 bound threads. Packing and NEON contributions are combined in the final configuration.

## Build & Verify

```bash
python tools/build_and_test.py
python tools/build_and_test.py --openmp
build/pac-bench --contest
python tools/verify_archive.py
```

## Attribution

Developed as part of the PAC2025 team effort on Kunpeng 920. This repository maintains the optimized implementation together with the complete historical record.
