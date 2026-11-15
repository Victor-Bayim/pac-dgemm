[English](MAINTENANCE.md) | **简体中文**

# 维护版实现，2026-11-15

## 显式 API 与布局

`include/pac_dgemm.h` 声明使用固定宽度维度的 C 兼容 API：`pac_dgemm_colmajor`。
它只支持列主序、非转置矩阵，允许任意有效 leading dimensions。A/B/C 不得重叠。
修改 C 前会校验维度、leading dimensions、必需指针和可表示地址跨度；调用方仍负责缓冲区大小和存储有效性。

## 相对历史清单的修正

1. 每个 A 微面板为 `[depth][4 rows]`，B 微面板为 `[depth][4 columns]`；packing 与分发使用相同面板偏移。
2. M/N 尾部在 packing 时补齐；部分输出 tile 先在本地 4×4 数组中计算，再仅将有效元素复制到 C。固定宽度 load/store 不会越过调用方行列尾部。
3. Beta 在全部 K 分块前仅应用一次。Beta=0 直接覆盖旧 C 而不读取它（包括 NaN）；alpha=0 和 K=0 不读取 A/B。
4. 使用固定正分块尺寸代替未校验环境变量。在输出修改前、OpenMP 区域之外分配工作空间。
5. 可选 OpenMP 划分互不重叠的行块，每个线程使用私有工作空间；串行路径不依赖 OpenMP 运行时。
6. 可移植与带条件保护的 AArch64 NEON kernel 共用面板合同。`PAC_FORCE_SCALAR=1` 即使在 AArch64 上也选择可移植 kernel。
7. 新驱动使用同一列主序独立参考实现；每轮预热和计时前恢复 C。预热单独执行，分别标记中位数与最小值，并打印全部耗时。

## 验证范围

参考测试覆盖矩形矩阵、多微块、K 分块、M/N 尾部、带 padding 的 lda/ldb/ldc、alpha/beta 组合、beta=0 且 C 含 NaN、空维度、参数拒绝和边界/padding 保持。
也覆盖四种比赛形状的最后分块范围。

Benchmark 对小工作负载检查全部输出，对大工作负载明确报告 64 个抽样输出。
`--contest` 选择四种历史维度，但产生的是当前实际机器上的*新维护版测量*。
