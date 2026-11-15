#ifndef __DGEMM_H__
#define __DGEMM_H__
#include <stdint.h>
#include "kblas.h"
#ifdef __cplusplus
extern "C" {
#endif

void dgemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transA, const enum CBLAS_TRANSPOSE transB, const int32_t M,
    const int32_t N, const int32_t K, const double alpha, const double *A, const int32_t lda,
    const double *B, const int32_t ldb, const double beta, double *C, const int32_t ldc);

#ifdef __cplusplus
}
#endif
#endif