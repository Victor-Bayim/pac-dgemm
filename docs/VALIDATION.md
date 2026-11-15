# Validation record — 2026-11-15 maintenance version

This record concerns the new maintenance implementation, not the PAC2025 binary
or historical timing records. Tests used Windows x86-64 and Zig 0.14.1's Clang
C++ driver with `-target x86_64-windows-gnu -std=c++17 -O2 -Wall -Wextra -Wpedantic`.
The compiler was an isolated local testing tool and is not part of this repository.

## Executed successfully

```text
python tools/build_and_test.py --cxx <local-zig.exe>
PASS cases=154 backend=portable-scalar openmp=0
```

- Both the real C++ reference-test executable and benchmark executable compiled
  and linked successfully, with no compiler warnings.
- 154 deterministic cases passed, including multiple packed microtiles, block
  boundaries, rectangular/tail dimensions, padded leading dimensions, alpha/beta,
  NaN C overwritten by beta=0, empty inputs, bad arguments, unrepresentable span,
  output guards/padding and restored-C repeated calls.
- The build helper's 13x7x9 smoke benchmark used a separate warmup, three timed
  rounds and an independent check of **every output** after each call. All passed.
  Its host timings are intentionally not presented as a performance result.
- `python tools/verify_archive.py` verified 15 byte-preserved historical files,
  16 stage rows and four elapsed-summary rows, including separate recalculation
  of the original reported and computed ratios.
- Python build/verification tools passed `py_compile`.

## Cross-compiled, not executed

```sh
zig c++ -target aarch64-linux-gnu -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -Iinclude -c dgemm_final.cpp -o build/pac-neon-aarch64.o
```

The guarded NEON implementation compiled to an AArch64 object without warnings.
This checks compilation of those intrinsics; it does not link or execute a target
binary, validate floating-point results on ARM, or measure Kunpeng performance.

## Not verified here

- Actual ARM NEON execution, Kunpeng 920 performance and NUMA placement.
- OpenMP runtime execution: this portable compiler installation did not include
  the required OpenMP headers/runtime; native tests ran the serial path.
- Full-size historical four-shape reruns or a same-machine optimized BLAS comparison.
- Historical original/transcribed C++ builds: they are deliberately excluded.

Run the same reference suite and the benchmark on the target AArch64 machine,
first serially and then with a supported OpenMP toolchain, before making a new
target-specific correctness or performance claim.
