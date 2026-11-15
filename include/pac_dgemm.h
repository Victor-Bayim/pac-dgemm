#ifndef PAC_DGEMM_H
#define PAC_DGEMM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum pac_gemm_status {
    PAC_GEMM_SUCCESS = 0,
    PAC_GEMM_INVALID_ARGUMENT = 1,
    PAC_GEMM_ALLOCATION_FAILURE = 2
};
/* Maintenance API: column-major, NoTrans only. C = alpha*A*B + beta*C.
 * Nonnegative dimensions; lda>=max(1,m), ldb>=max(1,k), ldc>=max(1,m).
 * Storage must be sufficiently large and A/B/C must not overlap.
 * A/B may be NULL when alpha==0 or k==0; all may be NULL when m==0 or n==0.
 * beta==0 does not read old C. Padding is never modified.
 * Invalid arguments/allocation failure return before modifying C.
 * Not a CBLAS ABI replacement: no row-major or transpose flags exist.
 */
int pac_dgemm_colmajor(int32_t m, int32_t n, int32_t k, double alpha,
    const double* a, int32_t lda, const double* b, int32_t ldb,
    double beta, double* c, int32_t ldc);
const char* pac_dgemm_backend(void);
int pac_dgemm_openmp_enabled(void);
#ifdef __cplusplus
}
#endif
#endif
