**English** | [简体中文](README.zh-CN.md)

# PAC-DGEMM: High-Performance FP64 GEMM on Kunpeng 920

FP64 DGEMM for Huawei Kunpeng 920. On the representative contest shape (M=12349, N=140, K=3040) the optimized configuration finishes in **0.040 s** versus **11.07 s** for a serial `-O1` baseline (**276×**).

The implementation is a self-contained column-major, NoTrans DGEMM (`pac_dgemm_colmajor`). It packs A and B into micro-panels with fixed strides, runs a 4×4 AArch64 NEON kernel for the main tiles, and handles tails, padded leading dimensions, and block boundaries without falling back to a scalar path for the common case. Alpha/beta scaling, including the NaN-safe `beta=0` case, is done inside the kernel. Invalid dimensions, strides, or pointers return an error; the caller is still responsible for buffer sizes.

OpenMP is optional and only active when compiled in. The default build needs only a C++17 compiler and Python 3; no third-party BLAS or historical SDK is required. `PAC_FORCE_SCALAR=1` disables NEON for comparison.

Correctness is checked against an independent long-double column-major reference. The test suite covers rectangular matrices, multiple microtiles, M/N/K block boundaries, padded leading dimensions, alpha/beta combinations, empty dimensions, rejected parameters, and output guards/padding. The four archived contest shapes are included.

The benchmark driver restores C outside the timed region, runs a separate warmup, and reports every duration along with median and minimum. Small cases check the full output; large cases check 64 sampled positions. All four contest shapes and the 16-stage throughput records are kept in `archive/data/` and can be re-verified with `tools/verify_archive.py`. OpenMP runs used `-O3` and 38 bound threads.

```bash
python tools/build_and_test.py
python tools/build_and_test.py --openmp
build/pac-bench --contest
python tools/verify_archive.py
```

Developed as part of the PAC2025 team effort. This repository keeps the maintained implementation, the historical data, and the verification tooling.
