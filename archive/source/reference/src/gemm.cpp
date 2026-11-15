#include <bits/stdc++.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <unistd.h>
#include <cmath>
#include "gemm.h"

#define GEMM_P 128
#define GEMM_Q 128
#define GEMM_R 128

void pack_B(const int32_t m, const int32_t n, const double *a, const int32_t ldb, double *bufB)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            bufB[j * m + i] = a[j * ldb + i];
        }
    }
}

void pack_A(const int32_t m, const int32_t n, const double *a, const int32_t lda, double *bufA)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j) {
            *bufA = a[j * lda + i];
            bufA++;
        }
    }
}


void KernelOperate(const int32_t m, const int32_t n, const int32_t k, const double alpha, const double *A, const double *B,
                   double *C, const int32_t ldc)
{
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            double sum = 0;
            for(int l=0;l<k;l++)
            {
                sum += A[i*k+l]*B[j*k+l];
            }
            C[j * ldc + i] += alpha*sum;
        }
    }
}

void betaOperate(const int32_t M, const int32_t N, const double beta, double *C, const int32_t ldc) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C[j * ldc + i] *= beta;
        }
    }
}

void dgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB, const int32_t M, const int32_t N, const int32_t K,
           const double alpha, const double *A, const int32_t lda,
           const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc)
{
    (void)order;
    (void)transA;
    (void)transB;
    double * bufA = (double*)malloc(GEMM_P*GEMM_Q*sizeof(double));
    double * bufB = (double*)malloc(GEMM_Q*GEMM_R*sizeof(double));

    int mini, minj, minl;

    minl = GEMM_P; // Typo in original code, likely should be mini = GEMM_P
    minl = GEMM_Q;
    minj = GEMM_R;

    for(int i=0; i< M; i+=mini) {
        mini = std::min(GEMM_P, M-i);
        for(int l=0; l<K; l+=minl) {
            minl = std::min(GEMM_Q, K-l);
            pack_A(mini, minl, A+l*lda + i, lda, bufA);
            for(int j=0; j<N; j+=minj) {
                minj = std::min(GEMM_R, N-j);
                pack_B(minl, minj, B+j*ldb + l, ldb, bufB);

                KernelOperate(mini, minj, minl, alpha, bufA, bufB, C+j*ldc+i, ldc);
            }
        }
    }
}