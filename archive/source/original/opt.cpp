#include <bits/stdc++.h>
#include <stdlib.h>
#include <unistd.h>
#include <cmath>
#include "gemm.h"

#define GEMM_P 128
#define GEMM_Q 128
#define GEMM_R 128

// 打包 B 子块（m 行 × n 列）到连续缓冲区 bufB
void pack_B(const int32_t m, const int32_t n,
            const double *a, const int32_t ldb,
            double *bufB)
{
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            bufB[j * m + i] = a[j * ldb + i];
        }
    }
}

// 优化后：按行 memcpy，将 A 的子块（m 行 × n 列）复制到连续缓冲区 bufA
void pack_A(const int32_t m, const int32_t n,
            const double *a, const int32_t lda,
            double *bufA)
{
    for (int i = 0; i < m; ++i) {
        const double *src = a + i * lda;    // 源矩阵第 i 行
        double       *dst = bufA + i * n;   // 目标缓冲区第 i 行
        std::memcpy(dst, src, n * sizeof(double));
    }
}

// 内核计算：将 j→i→k 循环顺序，保证对 C 的写入为单位步长连续访问
void KernelOperate(const int32_t m, const int32_t n, const int32_t k,
                   const double alpha,
                   const double *A, const double *B,
                   double *C, const int32_t ldc)
{
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            const double *pA = A + i * k;
            const double *pB = B + j * k;
            for (int l = 0; l < k; ++l) {
                sum += pA[l] * pB[l];
            }
            C[j * ldc + i] += alpha * sum;
        }
    }
}

// 按 beta 缩放 C 矩阵
void betaOperate(const int32_t M, const int32_t N,
                 const double beta,
                 double *C, const int32_t ldc)
{
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C[j * ldc + i] *= beta;
        }
    }
}

// 优化后的分块 DGEMM 实现
void dgemm(const enum CBLAS_ORDER order,
           const enum CBLAS_TRANSPOSE transA,
           const enum CBLAS_TRANSPOSE transB,
           const int32_t M, const int32_t N, const int32_t K,
           const double alpha, const double *A, const int32_t lda,
           const double *B, const int32_t ldb,
           const double beta, double *C, const int32_t ldc)
{
    (void)order;
    (void)transA;
    (void)transB;

    // 对 C 矩阵执行 beta 缩放
    betaOperate(M, N, beta, C, ldc);

    // 分配对齐的打包缓冲区（64 字节对齐，匹配缓存行大小）
    double *bufA = nullptr, *bufB = nullptr;
    posix_memalign((void**)&bufA, 64, GEMM_P * GEMM_Q * sizeof(double));
    posix_memalign((void**)&bufB, 64, GEMM_Q * GEMM_R * sizeof(double));

    int mini = GEMM_P;
    int minl = GEMM_Q;
    int minj = GEMM_R;

    // 三层阻塞循环结构保持不变
    for (int i0 = 0; i0 < M; i0 += mini) {
        mini = std::min(GEMM_P, M - i0);
        for (int l0 = 0; l0 < K; l0 += minl) {
            minl = std::min(GEMM_Q, K - l0);
            // 打包 A 子块至 bufA
            pack_A(mini, minl,
                   A + l0 * lda + i0, lda,
                   bufA);
            for (int j0 = 0; j0 < N; j0 += minj) {
                minj = std::min(GEMM_R, N - j0);
                // 打包 B 子块至 bufB
                pack_B(minl, minj,B + l0 + j0 * ldb,ldb,
                        bufB);
                // 核心计算，保证对 C 的连续写
                KernelOperate(mini, minj, minl,
                              alpha,
                              bufA, bufB,
                              C + j0 * ldc + i0, ldc);
        }
    }

    free(bufA);
    free(bufB);
}
