# PAC2025 DGEMM: historical records and a maintained implementation

FP64 matrix multiplication work associated with a PAC2025 team project on Huawei
Kunpeng 920. Victor was a **team member**. See [NOTICE.md](NOTICE.md) for attribution.

The root implementation is a **2026-11-15 maintenance reconstruction**. It demonstrates consistent micro-panel packing, a 4x4
portable/guarded AArch64 NEON kernel, safe tails and optional OpenMP.

## Build and test

Requires Python 3 and a C++17 compiler. The default build has no third-party
library or OpenMP requirement. It does not require `kblas.h` or the historical SDK.

```sh
python tools/build_and_test.py
# Explicit compiler, if not on PATH:
python tools/build_and_test.py --cxx /path/to/clang++
python tools/build_and_test.py --force-scalar
# Only where the compiler's OpenMP headers and runtime are installed:
python tools/build_and_test.py --openmp
```

On Windows, run from a configured Visual Studio developer terminal for `cl`, or
pass an installed GCC/Clang compiler. Zig is also accepted by `--cxx`; on Windows
the helper uses its `x86_64-windows-gnu` target, without requiring an MSVC SDK.
For an intentional cross-build, use `--target TRIPLE --compile-only` with Zig or
Clang. That mode **does not execute** tests.

The test executable compares actual C++ output against an independent long-double
column-major reference. It checks rectangular matrices, multiple microtiles,
M/N/K block boundaries, padded leading dimensions, alpha/beta, beta=0 with NaN C,
empty dimensions, rejected parameters, unchanged inputs and output guards/padding.
It also includes the four archived shapes' final block extents.

## API and source map

| File | Purpose |
|---|---|
| [dgemm_final.cpp](dgemm_final.cpp) | Current maintenance Packing + scalar/NEON + OpenMP implementation |
| [include/pac_dgemm.h](include/pac_dgemm.h) | Matching C-compatible API, status values and memory contract |
| [tests/correctness.cpp](tests/correctness.cpp) | Independent reference and bounds/semantics tests |
| [bench/benchmark.cpp](bench/benchmark.cpp) | New measurement driver; restores C before every call |
| [tools/build_and_test.py](tools/build_and_test.py) | Compiler detection, build, tests and a smoke benchmark |
| [archive/](archive/) | Historical source drafts, labelled transcription, framework and data |

`pac_dgemm_colmajor` computes `C = alpha*A*B + beta*C` for **column-major,
NoTrans** matrices. lda/ldb/ldc may include padding. A/B/C storage must not overlap.
There is no row-major or transpose option and this API is not a CBLAS ABI wrapper.
Invalid dimensions/strides/pointers and representational overflow return an error;
the caller remains responsible for actual buffer sizes. `PAC_FORCE_SCALAR=1`
disables NEON. OpenMP is used only when explicitly enabled at compilation.

[Maintenance notes](docs/MAINTENANCE.md) explain the panel-stride, tail, beta,
interface and driver corrections.

## New measurements

After building, run `build/pac-bench` (or `build/pac-bench.exe` on Windows):

```sh
build/pac-bench --m 129 --n 131 --k 133 --rounds 5 --warmup 1
# Optional full historical dimensions; potentially expensive:
build/pac-bench --contest --rounds 5 --warmup 1
```

The driver uses fixed-seed nontrivial inputs and alpha=0.75, beta=-0.25. It restores
C **outside** every timed interval, runs a separate warmup, prints every duration,
and labels median and minimum separately. Timing includes workspace/packing/kernel
work within the API call, but excludes input generation/reset and reference checking.
Small workloads check every output; large workloads explicitly check only 64
sampled positions. Record compiler, source hash, CPU, NUMA/thread
settings and actual backend when collecting target-machine results.

## Historical results

The **report-transcribed** representative result for M=12349, N=140, K=3040 is
11.07 s (serial `-O1`) versus 0.040 s (optimized configuration). All four shapes
are retained in [elapsed-summary.csv](archive/data/elapsed-summary.csv); 16
stage throughput records are in [stages.csv](archive/data/stages.csv).

- OpenMP also changed the compiler level to `-O3` and used 38 bound threads.
- Packing and NEON effects were not independently measured.
- The retained driver's final `-r=4` statistic selects the **minimum of four**,
  not an average and not a separately discarded warmup.
- The files are historical report/worksheet transcriptions, not per-run raw logs.
  Rounded-time ratios differ from ratios of six-decimal GFLOPS; both are preserved.

```sh
python tools/verify_archive.py
```

This command checks the archived byte manifest and recomputes data ratios.
