// Screenshot transcription prepared 2026-11-01; see TRANSCRIPTION.md.
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
static int get_gemm_q(){
    const char* env_q = getenv("GEMM_Q");
    return env_q ? atoi(env_q) : 128;
}
static int get_gemm_r(){
    const char* env_r = getenv("GEMM_R");
    return env_r ? atoi(env_r) : 128;
}

//数据打包函数（已适配列主序）

//pack_A :将A的一个m x n 子块打包到bufA(列面板)
void pack_A(const int32_t m, const int32_t n, const double* a, const int32_t lda, double* bufA){
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < m; ++i){
            bufA[j * m + i] = a[i + j * lda];
        }
    }
}

//pack_B :将B的一个m x n 子块打包到bufB(行面板)
void pack_B(const int32_t m, const int32_t n, const double* b, const int32_t lda, double* bufB){
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < m; ++i){
            bufB[i * n + j] = b[i + j * lda];
        }
    }
}

//朴素内核（用于对比和回退）
//朴素的矩阵乘法内核（列主序）
void KernelOperate(const int32_t m, const int32_t n, const int32_t k, const double alpha, const double* A, const double*B, double* C, const int32_t ldc){
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < m; ++i){
            double sum = 0.0;
            for(int l = 0; l < k; ++l){
                sum += A[i + l * m] * B[l * n + j];
            }
            C[i + j * ldc] +=alpha * sum;
        }
    }
}
//朴素的beta操作（列主序）
void betaOperate(const int32_t M, const int32_t N, const double beta, double* C, const int32_t ldc){
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < M; ++i){
            C[i + j * ldc] *= beta;
        }
    }
}

//---SIMD微内核（高性能核心）---
#if defined (__x86_64__) || defined (_M_X64)
//AVX/AVX2微内核（4x6）
static inline void micro_kernel_4x6_avx(const int32_t k,const double alpha, const double* A, const double* B,double* C, const int32_t ldc){
    __m256d c0 = _mm256_setzero_pd(), c1 = _mm256_setzero_pd(), c2 = _mm256_setzero_pd(),
    c3 = _mm256_setzero_pd(), c4 = _mm256_setzero_pd(), c5 = _mm256_setzero_pd();
    for(int l = 0; l<k; ++l){
        __m256d a = _mm256_load_pd(A + l * 4);
        c0 = _mm256_fmadd_pd(a, _mm256_broadcast_sd(B + l * 6 + 0), c0);
        c1 = _mm256_fmadd_pd(a, _mm256_broadcast_sd(B + l * 6 + 1), c1);
        c2 = _mm256_fmadd_pd(a, _mm256_broadcast_sd(B + l * 6 + 2), c2);
        c3 = _mm256_fmadd_pd(a, _mm256_broadcast_sd(B + l * 6 + 3), c3);
        c4 = _mm256_fmadd_pd(a, _mm256_broadcast_sd(B + l * 6 + 4), c4);
        c5 = _mm256_fmadd_pd(a, _mm256_broadcast_sd(B + l * 6 + 5), c5);
    }
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    _mm256_storeu_pd(C + 0 * ldc, _mm256_add_pd(_mm256_loadu_pd(C + 0 * ldc), _mm256_mul_pd(c0, alpha_vec)));
    _mm256_storeu_pd(C + 1 * ldc, _mm256_add_pd(_mm256_loadu_pd(C + 1 * ldc), _mm256_mul_pd(c1, alpha_vec)));
    _mm256_storeu_pd(C + 2 * ldc, _mm256_add_pd(_mm256_loadu_pd(C + 2 * ldc), _mm256_mul_pd(c2, alpha_vec)));
    _mm256_storeu_pd(C + 3 * ldc, _mm256_add_pd(_mm256_loadu_pd(C + 3 * ldc), _mm256_mul_pd(c3, alpha_vec)));
    _mm256_storeu_pd(C + 4 * ldc, _mm256_add_pd(_mm256_loadu_pd(C + 4 * ldc), _mm256_mul_pd(c4, alpha_vec)));
    _mm256_storeu_pd(C + 5 * ldc, _mm256_add_pd(_mm256_loadu_pd(C + 5 * ldc), _mm256_mul_pd(c5, alpha_vec)));
}

#if defined (__AVX512F__)
// AVX-512 微内核（8x6)
static inline void micro_kernel_8x6_avx512(const int32_t k,const double alpha, const double* A, const double* B, double* C, const int32_t ldc){
    __m512d c0 = _mm512_setzero_pd(), c1 = _mm512_setzero_pd(), c2 = _mm512_setzero_pd(),
    c3 = _mm512_setzero_pd(), c4 = _mm512_setzero_pd(), c5 = _mm512_setzero_pd();
    for(int l = 0; l < k; ++l){
        __m512d a = _mm512_load_pd(A + l * 8);
        c0 = _mm512_fmadd_pd(a, _mm512_set1_pd(*(B + l * 6 + 0)), c0);
        c1 = _mm512_fmadd_pd(a, _mm512_set1_pd(*(B + l * 6 + 1)), c1);
        c2 = _mm512_fmadd_pd(a, _mm512_set1_pd(*(B + l * 6 + 2)), c2);
        c3 = _mm512_fmadd_pd(a, _mm512_set1_pd(*(B + l * 6 + 3)), c3);
        c4 = _mm512_fmadd_pd(a, _mm512_set1_pd(*(B + l * 6 + 4)), c4);
        c5 = _mm512_fmadd_pd(a, _mm512_set1_pd(*(B + l * 6 + 5)), c5);
    }
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    _mm512_storeu_pd(C + 0 * ldc, _mm512_add_pd(_mm512_loadu_pd(C + 0 * ldc), _mm512_mul_pd(c0, alpha_vec)));
    _mm512_storeu_pd(C + 1 * ldc, _mm512_add_pd(_mm512_loadu_pd(C + 1 * ldc), _mm512_mul_pd(c1, alpha_vec)));
    _mm512_storeu_pd(C + 2 * ldc, _mm512_add_pd(_mm512_loadu_pd(C + 2 * ldc), _mm512_mul_pd(c2, alpha_vec)));
    _mm512_storeu_pd(C + 3 * ldc, _mm512_add_pd(_mm512_loadu_pd(C + 3 * ldc), _mm512_mul_pd(c3, alpha_vec)));
    _mm512_storeu_pd(C + 4 * ldc, _mm512_add_pd(_mm512_loadu_pd(C + 4 * ldc), _mm512_mul_pd(c4, alpha_vec)));
    _mm512_storeu_pd(C + 5 * ldc, _mm512_add_pd(_mm512_loadu_pd(C + 5 * ldc), _mm512_mul_pd(c5, alpha_vec)));
}
#endif // AVX512
#endif // x86

#if defined (__aarch64__)
// NEON 微内核（4x4）
static inline void micro_kernel_4x4_neon(const int32_t k, const double alpha, const double* A, const double* B, double* C, const int32_t ldc){
    //为一个4x4的块需要8个float64x2_t累加器（每个存2个double）
    float64x2_t c_col0_r01 = vdupq_n_f64(0.0), c_col0_r23 = vdupq_n_f64(0.0);
    float64x2_t c_col1_r01 = vdupq_n_f64(0.0), c_col1_r23 = vdupq_n_f64(0.0);
    float64x2_t c_col2_r01 = vdupq_n_f64(0.0), c_col2_r23 = vdupq_n_f64(0.0);
    float64x2_t c_col3_r01 = vdupq_n_f64(0.0), c_col3_r23 = vdupq_n_f64(0.0);

    for(int l = 0; l < k; ++l){
        // A的打包方式是列面板，一次加载A的一列（4个元素）
        float64x2_t a_r01 = vld1q_f64(A + l * 4 + 0); //A(0,l),A(1,l)
        float64x2_t a_r23 = vld1q_f64(A + l * 4 + 2); //A(2,l),A(3,l)

        // B的打包方式是行面板，广播B的一行中的每个元素
        //第0列
        float64x2_t b_val0 = vld1q_dup_f64(B + l * 4 + 0);
        c_col0_r01 = vfmaq_f64(c_col0_r01, a_r01, b_val0);
        c_col0_r23 = vfmaq_f64(c_col0_r23, a_r23, b_val0);

        //第1列
        float64x2_t b_val1 = vld1q_dup_f64(B + l * 4 + 1);
        c_col1_r01 = vfmaq_f64(c_col1_r01, a_r01, b_val1);
        c_col1_r23 = vfmaq_f64(c_col1_r23, a_r23, b_val1);

        //第2列
        float64x2_t b_val2 = vld1q_dup_f64(B + l * 4 + 2);
        c_col2_r01 = vfmaq_f64(c_col2_r01, a_r01, b_val2);
        c_col2_r23 = vfmaq_f64(c_col2_r23, a_r23, b_val2);

        //第3列
        float64x2_t b_val3 = vld1q_dup_f64(B + l * 4 + 3);
        c_col3_r01 = vfmaq_f64(c_col3_r01, a_r01, b_val3);
        c_col3_r23 = vfmaq_f64(c_col3_r23, a_r23, b_val3);
    }

    //将结果写回C矩阵： C = C + alpha * C_accum
    float64x2_t alpha_vec = vdupq_n_f64(alpha);

    //写回第0列
    vst1q_f64(C + 0 * ldc + 0, vfmaq_f64(vld1q_f64(C + 0 * ldc + 0), c_col0_r01, alpha_vec));
    vst1q_f64(C + 0 * ldc + 2, vfmaq_f64(vld1q_f64(C + 0 * ldc + 2), c_col0_r23, alpha_vec));

    //写回第1列
    vst1q_f64(C + 1 * ldc + 0, vfmaq_f64(vld1q_f64(C + 1 * ldc + 0), c_col1_r01, alpha_vec));
    vst1q_f64(C + 1 * ldc + 2, vfmaq_f64(vld1q_f64(C + 1 * ldc + 2), c_col1_r23, alpha_vec));

    //写回第2列
    vst1q_f64(C + 2 * ldc + 0, vfmaq_f64(vld1q_f64(C + 2 * ldc + 0), c_col2_r01, alpha_vec));
    vst1q_f64(C + 2 * ldc + 2, vfmaq_f64(vld1q_f64(C + 2 * ldc + 2), c_col2_r23, alpha_vec));

    //写回第3列
    vst1q_f64(C + 3 * ldc + 0, vfmaq_f64(vld1q_f64(C + 3 * ldc + 0), c_col3_r01, alpha_vec));
    vst1q_f64(C + 3 * ldc + 2, vfmaq_f64(vld1q_f64(C + 3 * ldc + 2), c_col3_r23, alpha_vec));
}

#if defined (__ARM_FEATURE_SME)
// SME 微内核（可伸缩）
static inline void micro_kernel_sme(const int32_t k, const double alpha, const double* A, const double* B, double* C, const int32_t ldc, const int32_t m_tile, const int32_t n_tile){
    const uint64_t VLEN_D = svcntd();
    svbool_t pg_m = svwhilelt_b64(0, 0, m_tile);
    svbool_t pg_n = svwhilelt_b64(0, 0, n_tile);
    __arm_smstart_za();
    svzero_za();
    for(int l = 0; l < k; ++l){
        svfloat64_t a_vec =  svld1_f64(pg_m, &A[l * VLEN_D]);
        svfloat64_t b_vec =  svld1_f64(pg_n, &B[l * VLEN_D]);
        svfmopa_za(0, pg_m, pg_n, a_vec, b_vec);
    }
    for(uint64_t j = 0; j < n_tile; ++j){
        svfloat64_t c_vec = svld1_f64(pg_m, &C[j * ldc]);
        svfloat64_t za_vec = svld1_za(0, j, pg_m);
        c_vec = svmla_f64_x(c_vec, za_vec, alpha, pg_n);
        svst1_f64(pg_m, &C[j * ldc], c_vec);
    }
    __arm_smstop();
}
#endif // SME
#endif // ARM

// Transcribed from screenshots image7.png through image11.png.
void KernelOperate_SIMD(const int32_t m, const int32_t n, const int32_t k, const double alpha,
                        const double* A, const double* B, double* C, const int32_t ldc) {
    const char* use_register_env = getenv("REGISTER");
    bool use_register = (use_register_env != nullptr && strcmp(use_register_env, "1") == 0);
#if defined(__x86_64__) || defined(_M_X64)
    if (use_register) {
#if defined(__AVX512F__)
        const char* use_avx512_env = getenv("AVX512");
        bool use_avx512 = (use_avx512_env != nullptr && strcmp(use_avx512_env, "1") == 0);
        if (use_avx512) {
            for (int j = 0; j < n; j += 6)
                for (int i = 0; i < m; i += 8)
                    micro_kernel_8x6_avx512(k, alpha, &A[i * k], &B[j * k], &C[i + j * ldc], ldc);
            return;
        }
#endif
        for (int j = 0; j < n; j += 6)
            for (int i = 0; i < m; i += 4)
                micro_kernel_4x6_avx(k, alpha, &A[i * k], &B[j * k], &C[i + j * ldc], ldc);
        return;
    }
#elif defined(__aarch64__)
#if defined(__ARM_FEATURE_SME)
    const char* use_sme_env = getenv("USE_SME");
    bool use_sme = (use_sme_env != nullptr && strcmp(use_sme_env, "1") == 0);
    if (use_sme) {
        const int VLEN_D = svcntd();
        for (int j = 0; j < n; j += VLEN_D)
            for (int i = 0; i < m; i += VLEN_D)
                micro_kernel_sme(k, alpha, &A[i * k], &B[j * k], &C[i + j * ldc], ldc,
                                 std::min(m - i, VLEN_D), std::min(n - j, VLEN_D));
        return;
    }
#endif
    if (use_register) {
        for (int j = 0; j < n; j += 4)
            for (int i = 0; i < m; i += 4)
                micro_kernel_4x4_neon(k, alpha, &A[i * k], &B[j * k], &C[i + j * ldc], ldc);
        return;
    }
#endif
    KernelOperate(m, n, k, alpha, A, B, C, ldc);
}

void betaOperate_SIMD(const int32_t M, const int32_t N, const double beta,
                      double* C, const int32_t ldc) {
    if (beta == 1.0) return;
    if (beta == 0.0) {
        for (int j = 0; j < N; ++j) for (int i = 0; i < M; ++i) C[i + j * ldc] = 0.0;
        return;
    }
#if defined(__x86_64__) || defined(_M_X64)
    __m256d beta_vec = _mm256_set1_pd(beta);
    for (int j = 0; j < N; ++j) for (int i = 0; i < M; i += 4)
        _mm256_storeu_pd(&C[i+j*ldc], _mm256_mul_pd(_mm256_loadu_pd(&C[i+j*ldc]), beta_vec));
#elif defined(__aarch64__)
    float64x2_t beta_vec = vdupq_n_f64(beta);
    for (int j = 0; j < N; ++j) for (int i = 0; i < M; i += 2)
        vst1q_f64(&C[i+j*ldc], vmulq_f64(vld1q_f64(&C[i+j*ldc]), beta_vec));
#else
    betaOperate(M, N, beta, C, ldc);
#endif
}

void numa_aware_init_C(const int32_t M, const int32_t N, const double beta,
                       double* C, const int32_t ldc) {
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (beta == 0.0) C[i + j * ldc] = 0.0;
                else C[i + j * ldc] *= beta;
            }
        }
    }
}

extern "C" void dgemm(const char order, const char transA, const char transB,
                      const int32_t M, const int32_t N, const int32_t K,
                      const double alpha, const double* A, const int32_t lda,
                      const double* B, const int32_t ldb, const double beta,
                      double* C, const int32_t ldc) {
    const char* use_simd_env = getenv("SIMD");
    bool use_simd = (use_simd_env != nullptr && strcmp(use_simd_env, "1") == 0);
    const char* use_omp_env = getenv("USE_OMP");
    bool use_omp = (use_omp_env != nullptr && strcmp(use_omp_env, "1") == 0);
    const char* use_numa_env = getenv("NUMA");
    bool use_numa = (use_numa_env != nullptr && strcmp(use_numa_env, "1") == 0);
    if (use_numa && use_omp) {
        numa_aware_init_C(M, N, beta, C, ldc);
    } else {
        if (use_simd) betaOperate_SIMD(M, N, beta, C, ldc);
        else betaOperate(M, N, beta, C, ldc);
    }
    if (alpha == 0.0) return;
    const int GEMM_P = get_gemm_p();
    const int GEMM_Q = get_gemm_q();
    const int GEMM_R = get_gemm_r();
#pragma omp parallel if(use_omp)
    {
        double* bufA = nullptr;
        double* bufB = nullptr;
        if (posix_memalign((void**)&bufA, 64, GEMM_P * GEMM_Q * sizeof(double)) != 0 ||
            posix_memalign((void**)&bufB, 64, GEMM_Q * GEMM_R * sizeof(double)) != 0) {
            // Screenshot leaves allocation failure handling unimplemented.
        }
#pragma omp for schedule(static)
        for (int i = 0; i < M; i += GEMM_P) {
            int mini = std::min(GEMM_P, M - i);
            for (int l = 0; l < K; l += GEMM_Q) {
                int minl = std::min(GEMM_Q, K - l);
                pack_A(mini, minl, A + i + l * lda, lda, bufA);
                for (int j = 0; j < N; j += GEMM_R) {
                    int minj = std::min(GEMM_R, N - j);
                    pack_B(minl, minj, B + l + j * ldb, ldb, bufB);
#if defined(__x86_64__) || defined(_M_X64)
                    _mm_prefetch((const char*)(B + l + (j + GEMM_R) * ldb), _MM_HINT_T0);
#elif defined(__aarch64__)
                    __builtin_prefetch((const char*)(B + l + (j + GEMM_R) * ldb), 0, 3);
#endif
                    if (use_simd) KernelOperate_SIMD(mini, minj, minl, alpha, bufA, bufB, C + i + j * ldc, ldc);
                    else KernelOperate(mini, minj, minl, alpha, bufA, bufB, C + i + j * ldc, ldc);
                }
            }
        }
        free(bufA);
        free(bufB);
    }
}
