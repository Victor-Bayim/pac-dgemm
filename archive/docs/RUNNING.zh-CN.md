[English](RUNNING.md) | **简体中文**

# 运行与测量解读

## 归档配置

历史运行脚本使用串行 `-O1` 和优化 `-O3` 二进制，优化阶段配置如下：

```sh
export USE_OMP=1
export OMP_NUM_THREADS=38
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export SIMD=1
export REGISTER=1
unset NUMA USE_SME
./dgemmbench_opt -m 12349 -n 140 -k 3040 -c 1 -r 4
./dgemmbench_opt -m 24698 -n 8 -k 3040 -c 1 -r 4
./dgemmbench_opt -m 256 -n 3040 -k 3040 -c 1 -r 4
./dgemmbench_opt -m 3040 -n 16 -k 12349 -c 1 -r 4
```

保留的框架需要 Linux/AArch64 环境及原工具链中的 `kblas.h` 依赖。截图源码与框架的公开 dgemm 签名也存在差异。

新运行应先在包括尾块的小形状上确认 scalar 与 SIMD 结果正确，再记录编译器/版本、CPU 型号、NUMA 绑定、线程数、源码哈希与全部逐轮耗时。

## 计时方法

`source/reference/test/gemmtest.cpp` 循环 R 次，将耗时排序后选择 `v_time[0]`。
吞吐量为 `2*M*N*K/(seconds*1e9)`。历史脚本扫描重复次数 1 至 5，报告最高观测吞吐量。
因此最终记录取最佳耗时，不是平均值或置信区间。

工作表中四舍五入的加速比文字与保留的六位小数 GFLOPS 比值并非每格完全一致。
`tools/verify.py` 会报告两者。报告耗时的取整会带来另一处小差异，两者不应被静默替换。
