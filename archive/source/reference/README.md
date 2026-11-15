# 目录结构：
- include
- - 包含头文件的目录
- src
- - 源代码文件的目录
- tests
- - 测试代码文件的目录
- Makefile

# 文件解析：
gemm.cpp:实现了一个简易的dgemm，包含tiling与kernel
gemmtest.cpp:测试gemm函数的正确性，与性能
help.cpp:帮助信息的输出

# 编译方法
source /PATH_TO_HPCKit/latest/servars.sh #加载HPCKit环境变量
make all
make lib
make test

# 优化思路
- 使用更高效的矩阵乘法算法（如Strassen算法、Coppersmith-Winograd算法等）
- 并行化计算以提高效率（使用多线程或多进程）
- 优化内存访问模式以减少缓存未命中率
- 向量化操作以利用SIMD指令集加速计算