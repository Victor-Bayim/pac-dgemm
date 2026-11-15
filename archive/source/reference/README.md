**English** | [简体中文](README.zh-CN.md)

# Directory structure
- include
- - Header files
- src
- - Source code files
- tests
- - Test code files
- Makefile

# File overview
gemm.cpp: implements a simple dgemm with tiling and a kernel
gemmtest.cpp: tests gemm correctness and performance
help.cpp: prints help information

# Building
source /PATH_TO_HPCKit/latest/servars.sh # Load HPCKit environment variables
make all
make lib
make test

# Optimization ideas
- Use more efficient matrix multiplication algorithms, such as Strassen or Coppersmith-Winograd
- Parallelize computation with multiple threads or processes
- Optimize memory access patterns to reduce cache misses
- Vectorize operations to use SIMD instructions
