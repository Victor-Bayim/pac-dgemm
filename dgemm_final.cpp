#include <cstdint>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<string.h>
#include<algorithm> //for std::min
#include<omp.h> //for opemmp

//平台的定的SIMD头文件
#if defined (__x86_64__) || defined (_M_X64)
#include <immintrin.h> //for AVX,AVX2,AVX512
#elif defined(__aarch64__)
#include<arm_neon.h> //for neon
#if defined(__ARM_FEATURE_SME)
#include<arm_sme.h>//for SME
#endif
#endif

// 运行时配置函数
static int get_gemm_p(){
    const char* env_p = getenv("GEMM_P");
    return env_p ? atoi(env_p) : 128;
}

// ... (core packing + NEON microkernel + OpenMP scheduling follows in full source)
// Full file is 221 lines; key parts include matrix packing and 4x4 NEON kernel.
