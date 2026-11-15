[English](README.md) | **简体中文**

# PAC-DGEMM：鲲鹏 920 上的 FP64 GEMM

华为鲲鹏 920 上的 FP64 DGEMM。规模 M=12349, N=140, K=3040：串行 `-O1` **11.07 s**，优化后 **0.040 s**，**276×**。

`pac_dgemm_colmajor` 是列主序、NoTrans。A 和 B 打成固定 stride 的微面板。主块走 4×4 AArch64 NEON 内核。尾部、填充 leading dimension、块边界走同一条路径。Alpha/beta（含 NaN 安全的 `beta=0`）在内核里完成。维度、步长或指针非法直接返回错误。

OpenMP 编译时打开。默认构建只要 C++17 编译器和 Python 3，不依赖 BLAS。`PAC_FORCE_SCALAR=1` 关掉 NEON。

正确性对照独立的 long-double 列主序参考：矩形、多微块、M/N/K 块边界、填充 leading dimension、alpha/beta、空维度、非法参数、输出保护。四个竞赛形状在测试里。

基准驱动在计时区间外恢复 C，单独预热，打出每一轮、中位数和最小值。小规模全量校验，大规模抽 64 个点。四个形状和 16 阶段吞吐在 `archive/data/`。OpenMP 数字用 `-O3`、38 个绑定线程。

```bash
python tools/build_and_test.py
python tools/build_and_test.py --openmp
build/pac-bench --contest
python tools/verify_archive.py
```
