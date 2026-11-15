# PAC-DGEMM: FP64 GEMM on Kunpeng 920

FP64 DGEMM on Huawei Kunpeng 920. Shape M=12349, N=140, K=3040: serial `-O1` **11.07 s**, optimized **0.040 s**, **276×**.

`pac_dgemm_colmajor` is column-major, NoTrans. A and B are packed into micro-panels with fixed strides. The main tiles run a 4×4 AArch64 NEON kernel. Tails, padded leading dimensions, and block boundaries stay on the same path. Alpha/beta, including NaN-safe `beta=0`, is applied inside the kernel. Bad dimensions, strides, or pointers return an error.

OpenMP turns on at compile time. Default build is a C++17 compiler and Python 3. No BLAS. `PAC_FORCE_SCALAR=1` drops NEON.

Correctness is checked against an independent long-double column-major reference: rectangular shapes, multiple microtiles, M/N/K block edges, padded leading dimensions, alpha/beta, empty dimensions, rejected arguments, output guards. The four contest shapes are in the suite.

The benchmark restores C outside the timed region, warms up separately, and prints every run plus median and minimum. Small shapes check every output; large shapes check 64 sampled positions. Four shapes and the 16-stage throughput records sit in `archive/data/`. OpenMP numbers used `-O3` and 38 bound threads.

```bash
python tools/build_and_test.py
python tools/build_and_test.py --openmp
build/pac-bench --contest
python tools/verify_archive.py
```
