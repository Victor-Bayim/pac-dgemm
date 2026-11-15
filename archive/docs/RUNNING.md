# Running and interpreting the measurements

## Archived configuration

The historical run script uses a serial `-O1` binary and an optimized `-O3` binary, with these settings for the optimized stages:

```sh
export USE_OMP=1
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export SIMD=1
export REGISTER=1
unset NUMA USE_SME
./dgemmbench_opt -m 12349 -n 140 -k 3040 -c 1 -r 4
./dgemmbench_opt -m 24698 -n 8 -k 3040 -c 1 -r 4
./dgemmbench_opt -m 256 -n 3040 -k 3040 -c 1 -r 4
./dgemmbench_opt -m 3040 -n 16 -k 12349 -c 1 -r 4
```

The retained framework needs a Linux/AArch64 environment and the `kblas.h` dependency from its original toolchain. The screenshot source and framework also differ in their public dgemm signature.

For a new run, first establish correct scalar and SIMD results on small shapes, including tails, then record compiler/version, CPU model, NUMA binding, thread count, source hash and all individual timings.

## Timing method

`source/reference/test/gemmtest.cpp` loops R times, sorts the durations and selects `v_time[0]`. Throughput is `2*M*N*K/(seconds*1e9)`. The historical script scans repeat counts 1 through 5 and reports the highest observed throughput. Hence the recorded final row is a best-duration measurement, not an average or a confidence interval.

The spreadsheet's rounded speedup text does not exactly match ratios of its retained six-decimal GFLOPS values in every cell. `tools/verify.py` reports both. Rounded report times lead to another small difference; neither should be silently substituted for the other.
