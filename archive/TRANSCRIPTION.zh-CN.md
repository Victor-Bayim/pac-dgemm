[English](TRANSCRIPTION.md) | **简体中文**

# 源码转录

`source/transcribed/dgemm.cpp` 于 2026-11-01 对照本地草稿与全部 11 张代码图片制作，是截图代码清单的文字转录。注释和空格已规范化。
`<cstdint>` 从本地草稿保留，用于定义固定宽度类型。

原始 `.cpp` 草稿在 SME 函数后结束，缺少条件编译结束指令和外层 dgemm 实现，且含拼写错误。
截图补充了剩余 dispatcher 和入口，并使用正确的 NEON load/store 拼写。这些版本分开保留。

源码图片对应关系：

- image1：头文件与 GEMM_P/Q/R 环境设置。
- image2–3：packing、标量操作与 AVX 微内核。
- image4：AVX-512 与 NEON 初始化。
- image5–6：NEON 乘加/存储与 SME 函数。
- image7–8：编译期架构分支与环境变量控制的分发。
- image9：beta/NUMA 操作与 dgemm 入口。
- image10–11：OpenMP 调度、packing、prefetch、分发与清理。

SME 路径以编译期 `__ARM_FEATURE_SME` 和运行时 `USE_SME=1` 为条件，不是运行时 CPU 特性检测。

作为替代 BLAS 使用前，需处理 SIMD 尾块、多个微块间的 packed-panel 步长、分配失败，以及 char 与 enum 不一致的公开签名。
可移植 4×4 检查仅覆盖独立 tile 的数学运算。
