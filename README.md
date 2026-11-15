# PAC2025 DGEMM: historical records and a maintained implementation

FP64 matrix multiplication work associated with a PAC2025 team project on Huawei
Kunpeng 920. Victor was a **team member**; sole authorship of team functions is
not claimed. See [NOTICE.md](NOTICE.md) for attribution.

The root implementation is a **2026-11-15 maintenance reconstruction**, replacing
the earlier placeholder. It demonstrates consistent micro-panel packing, a 4x4
portable/guarded AArch64 NEON kernel, safe tails and optional OpenMP. It is **not
the recovered competition executable**, and its test results are not evidence
that the historical performance has been reproduced.

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
Clang. That mode **does not execute** tests. Tools and binaries are not vendored.

The test executable compares actual C++ output against an independent long-double
column-major reference. It checks rectangular matrices, multiple microtiles,
M/N/K block boundaries, padded leading dimensions, alpha/beta, beta=0 with NaN C,
empty dimensions, rejected parameters, unchanged inputs and output guards/padding.
It also includes the four archived shapes' final block extents. These small
tests are not full-size contest benchmarks. See [validation scope](docs/VALIDATION.md).

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
interface and driver corrections. The previous nonfunctional `cachecount.py`
placeholder was removed; no cache-counter measurement is claimed.

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
sampled positions. Every output file is a new maintenance measurement, never a
replacement for archived contest CSVs. Record compiler, source hash, CPU, NUMA/thread
settings and actual backend when collecting target-machine results.

## Historical results and their limits

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
- Historical incomplete drafts and the screenshot transcription are archived
  unchanged and excluded from the build. The old framework requires an external
  SDK, has signature/layout/repeated-C issues, and is not a validated reproduction.

```sh
python tools/verify_archive.py
```

This command checks the archived byte manifest and recomputes data ratios. It
does not execute NEON, reconstruct a competition binary, or rerun performance.
Same-machine BLAS comparisons and actual Kunpeng/NEON/OpenMP execution remain
target-machine work. No current license is inferred for historical team/framework
material; existing attribution and source status remain intact.
