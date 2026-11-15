[English](README.md) | **简体中文**

# PAC-DGEMM：鲲鹏 920 上的高性能 FP64 GEMM

面向华为鲲鹏 920 的 FP64 DGEMM。在代表性竞赛规模（M=12349, N=140, K=3040）上，优化配置耗时 **0.040 s**，对比串行 `-O1` 基线的 **11.07 s**（**276×**）。

实现是自包含的列主序、NoTrans DGEMM（`pac_dgemm_colmajor`）。它将 A 和 B 打包成固定 stride 的微面板，用 4×4 AArch64 NEON 内核处理主要块，并对尾部、填充 leading dimension 和块边界做安全处理，常见情况不退回到标量路径。Alpha/beta 缩放（包括 NaN 安全的 `beta=0`）在内核内完成。无效维度、步长或指针会返回错误；调用方仍需负责缓冲区大小。

OpenMP 可选，仅在编译时启用才生效。默认构建只需 C++17 编译器和 Python 3，不依赖第三方 BLAS 或历史 SDK。`PAC_FORCE_SCALAR=1` 可禁用 NEON 用于对比。

正确性对照独立的 long-double 列主序参考实现检查。测试覆盖矩形矩阵、多微块、M/N/K 块边界、填充 leading dimension、alpha/beta 组合、空维度、被拒绝的参数以及输出保护/padding。四个归档竞赛形状也包含在内。

基准测试驱动在计时区间之外恢复 C，单独做预热，并输出每次耗时、中位数和最小值。小规模检查全部输出，大规模检查 64 个抽样位置。四个竞赛形状和 16 阶段吞吐记录保存在 `archive/data/`，可用 `tools/verify_archive.py` 重新验证。OpenMP 运行使用 `-O3` 和 38 个绑定线程。

```bash
python tools/build_and_test.py
python tools/build_and_test.py --openmp
build/pac-bench --contest
python tools/verify_archive.py
```

作为 PAC2025 团队工作的一部分。本仓库保留维护后的实现、历史数据以及验证工具。
