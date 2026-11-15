[English](README.md) | **简体中文**

# PAC-DGEMM：鲲鹏 920 上的高性能 FP64 GEMM

面向华为鲲鹏 920 的手工调优 FP64 DGEMM 实现。在代表性竞赛规模（M=12349, N=140, K=3040）上，将串行 `-O1` 基线的 **11.07 s** 降至优化配置下的 **0.040 s**，实测加速比 **276×**。

## 核心技术

- 一致的微面板 packing，配合精心选择的 panel stride
- 4×4 可移植、带边界保护的 AArch64 NEON 内核
- 对任意矩阵维度与块边界的安全尾部处理
- 可选的 OpenMP 多核并行
- 完整的列主序、NoTrans API，支持 leading dimension 填充、alpha/beta 缩放以及 NaN 安全的 `beta=0`

实现完全自包含，不依赖第三方 BLAS 或历史 SDK。正确性由独立的 long-double 列主序参考实现验证，覆盖矩形矩阵、多微块、填充步长、边界情况、空维度以及输出保护。

## 结果

| 规模               | 串行 -O1 | 优化后   | 加速比 |
|--------------------|----------|----------|--------|
| 12349 × 140 × 3040 | 11.07 s  | 0.040 s  | 276×   |

四个竞赛规模及对应的 16 阶段吞吐数据均完整保留并可独立验证。OpenMP 运行使用 `-O3` 与 38 个绑定线程。Packing 与 NEON 的贡献合并在最终配置中。

## 构建与验证

```bash
python tools/build_and_test.py
python tools/build_and_test.py --openmp
build/pac-bench --contest
python tools/verify_archive.py
```

## 归属

作为 PAC2025 团队在鲲鹏 920 上的工作成果。本仓库维护优化后的实现及完整历史记录。
