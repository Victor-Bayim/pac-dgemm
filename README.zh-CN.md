[English](README.md) | **简体中文**

# PAC2025 DGEMM：历史记录与维护版实现

华为鲲鹏 920 上 PAC2025 团队项目的 FP64 矩阵乘法工作。Victor 为**参赛队员**，归属见 [NOTICE.md](NOTICE.zh-CN.md)。

根目录实现是 **2026-11-15 维护重建版本**，包含一致的微面板 packing、可移植/带 AArch64 条件保护的 4×4 NEON kernel、安全尾块和可选 OpenMP。

## 构建与测试

需要 Python 3 和 C++17 编译器。默认构建不要求第三方库或 OpenMP，也不需要 `kblas.h` 或历史 SDK。

```sh
python tools/build_and_test.py
# Explicit compiler, if not on PATH:
python tools/build_and_test.py --cxx /path/to/clang++
python tools/build_and_test.py --force-scalar
# Only where the compiler's OpenMP headers and runtime are installed:
python tools/build_and_test.py --openmp
```

Windows 下可在配置好的 Visual Studio 开发者终端中使用 `cl`，也可指定已安装的 GCC/Clang。
`--cxx` 也接受 Zig；Windows 下构建助手使用其 `x86_64-windows-gnu` 目标，无需 MSVC SDK。
交叉构建可结合 Zig 或 Clang 使用 `--target TRIPLE --compile-only`，该模式**不会执行**测试。

测试程序将实际 C++ 输出与独立的 long-double 列主序参考实现比较，覆盖矩形矩阵、多微块、M/N/K 分块边界、带 padding 的 leading dimensions、alpha/beta、beta=0 且 C 含 NaN、空维度、参数拒绝、输入不变和输出边界/padding。
还覆盖四个归档形状的最后分块范围。

## API 与源码导航

| 文件 | 用途 |
|---|---|
| [dgemm_final.cpp](dgemm_final.cpp) | 当前维护版 Packing + scalar/NEON + OpenMP 实现 |
| [include/pac_dgemm.h](include/pac_dgemm.h) | 配套 C 兼容 API、状态值和内存合同 |
| [tests/correctness.cpp](tests/correctness.cpp) | 独立参考实现与边界/语义测试 |
| [bench/benchmark.cpp](bench/benchmark.cpp) | 新测量驱动，每次调用前恢复 C |
| [tools/build_and_test.py](tools/build_and_test.py) | 编译器检测、构建、测试和性能烟测 |
| [archive/](archive) | 历史源码草稿、标记转录、框架和数据 |

`pac_dgemm_colmajor` 对**列主序、NoTrans** 矩阵计算 `C = alpha*A*B + beta*C`。
lda/ldb/ldc 可含 padding，A/B/C 存储不得重叠。API 不提供行主序或转置选项，也不是 CBLAS ABI 包装。
无效维度、步长、指针和表示范围溢出会返回错误；调用方仍负责实际缓冲区大小。
`PAC_FORCE_SCALAR=1` 禁用 NEON。只有编译时显式启用才使用 OpenMP。

[维护说明](docs/MAINTENANCE.zh-CN.md) 介绍面板步长、尾块、beta、接口和驱动修正。

## 新测量

构建后运行 `build/pac-bench`（Windows 下为 `build/pac-bench.exe`）：

```sh
build/pac-bench --m 129 --n 131 --k 133 --rounds 5 --warmup 1
# Optional full historical dimensions; potentially expensive:
build/pac-bench --contest --rounds 5 --warmup 1
```

驱动使用固定随机种子的非平凡输入，以及 alpha=0.75、beta=-0.25。
每次计时区间**之外**恢复 C，单独预热，打印所有耗时，并分别标记中位数与最小值。
计时包含 API 调用内的工作空间、packing 和 kernel 工作，排除输入生成、重置与参考校验。
小规模工作负载检查全部输出；大规模工作负载明确仅检查 64 个抽样位置。
在目标机器采集结果时，记录编译器、源码哈希、CPU、NUMA/线程设置和实际后端。

## 历史结果

从报告转录的代表性结果为 M=12349、N=140、K=3040：串行 `-O1` 耗时 11.07 s，优化配置耗时 0.040 s。
四种形状保留在 [elapsed-summary.csv](archive/data/elapsed-summary.csv)，16 条阶段吞吐记录见 [stages.csv](archive/data/stages.csv)。

- OpenMP 阶段同时将编译级别改为 `-O3`，使用 38 个绑定线程。
- Packing 与 NEON 的效果没有分别独立测量。
- 保留驱动的最终 `-r=4` 统计取**四次最小值**，不是平均值，也不是另行丢弃预热。
- 文件是历史报告/工作表转录，不是逐轮原始日志。四舍五入耗时的比值与六位小数 GFLOPS 的比值不同，两者均保留。

```sh
python tools/verify_archive.py
```

该命令校验归档文件清单的字节信息，并复算数据比值。
