[English](README.md) | **简体中文**

# PAC-DGEMM：鲲鹏 920 上的高性能 FP64 GEMM

面向华为鲲鹏 920 的手工调优 FP64 DGEMM 实现。在代表性竞赛规模（M=12349, N=140, K=3040）上取得实测 **276×** 加速，将串行 `-O1` 基线的 **11.07 s** 降至优化配置下的 **0.040 s**。

## 实现内容

实现聚焦于 AArch64 上矩阵乘法的关键路径：

- 一致的微面板 packing，配合精心选择的 panel stride，以最大化缓存复用与向量加载效率。
- 4×4 可移植、带边界保护的 AArch64 NEON 内核，处理主计算并带有边缘保护。
- 对任意矩阵维度、块边界以及填充 leading dimension 的安全尾部处理。
- 可选的 OpenMP 多核并行，包含正确的工作分配与同步。
- 干净的列主序、NoTrans API，正确处理 alpha/beta 缩放、NaN 安全的 `beta=0`、重叠存储拒绝，以及对无效维度、步长或指针的完整错误报告。

代码完全自包含，不依赖第三方 BLAS 或历史 SDK。正确性由独立的 long-double 列主序参考实现验证，覆盖矩形矩阵、多微块、填充步长、空维度、边界情况以及输出保护。

## 结果

| 规模               | 串行 -O1 | 优化后   | 加速比 |
|--------------------|----------|----------|--------|
| 12349 × 140 × 3040 | 11.07 s  | 0.040 s  | 276×   |

四个竞赛规模及对应的 16 阶段吞吐数据均完整保留，并可通过 `tools/verify_archive.py` 独立验证。OpenMP 运行使用 `-O3` 与 38 个绑定线程。Packing 与 NEON 的效果合并在最终优化配置中。

## 构建与验证

```bash
python tools/build_and_test.py
python tools/build_and_test.py --openmp
build/pac-bench --contest
python tools/verify_archive.py
```

## 归属

作为 PAC2025 团队在鲲鹏 920 上的工作成果。本仓库维护优化后的实现、完整历史记录以及验证工具。
