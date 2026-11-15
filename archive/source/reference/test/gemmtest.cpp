#include <bits/stdc++.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstdio>
#include <ctime>
#include <vector>
#include <kblas.h>
#include <arm_neon.h>
#include "help.h"
#include "gemm.h"

#ifdef FLOAT64
#define FLOAT double
#define GEMM cblas_dgemm
#define TESTED_GEMM dgemm
#endif

#define FMAX(x, y) (((x) > (y)) ? (x) : (y))
#define FABS(x) (((x) > 0) ? (x) : -(x))
std::vector<double> v_time;

// Reference GEMM implementation for correctness check
void RefGemm(int M, int N, int K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] *= beta;
            for (int k = 0; k < K; k++) {
                C[i * N + j] += alpha * A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Generates the L1 norm of the error matrix (C - ref_C)
FLOAT GenErrNorm1(FLOAT *C, FLOAT *ref_C, int M, int N)
{
    FLOAT err_norm = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            err_norm += fabs(C[i * N + j] - ref_C[i * N + j]);
        }
    }
    return err_norm;
}

// Generates the L1 norm of a matrix
FLOAT GenNorm1(FLOAT *A, int M, int N)
{
    FLOAT norm = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            norm += FABS(A[i * N + j]);
        }
    }
    return norm;
}

// Calculates machine epsilon
FLOAT GetEpsilon()
{
    static FLOAT eps;
    const FLOAT half = 0.5;
    volatile FLOAT maxval, f1 = 0.5;
    do {
        eps = f1;
        f1 *= half;
        maxval = 1.0 + f1;
    } while (maxval != 1.0);
    return eps;
}

// Gets current time for benchmarking
inline double now()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + (double)t.tv_usec * 1.e-6;
}

// Generates a matrix with random values
void gen_mat(int m, int n, FLOAT *mat)
{
    srand(time(NULL));
    for (int i = 0; i < m * n; ++i) {
        mat[i] = 0.5 - (FLOAT)rand() / (FLOAT)RAND_MAX;
    }
}

int main(int argc, char **argv)
{
    int M = 256;
    int N = 256;
    int K = 256;
    int R = 1;
    int check = 0;
    int threads = 24;
    int opt;
    FLOAT alpha = 1.0;
    FLOAT beta = 1.0;
    while ((opt = getopt(argc, argv, "m:n:k:r:c:t:h")) != -1) {
        switch (opt) {
        case 'm': M = atoi(optarg); break;
        case 'n': N = atoi(optarg); break;
        case 'k': K = atoi(optarg); break;
        case 'r': R = atoi(optarg); break;
        case 'c': check = atoi(optarg); break;
        case 't': threads = atoi(optarg); break;
        case 'h': helpmsg(argv[0]); exit(0); break;
        default: break;
        }
    }

    FLOAT *A = new FLOAT[M * K];
    FLOAT *B = new FLOAT[K * N];
    FLOAT *C = new FLOAT[M * N];
    FLOAT *ck_C = new FLOAT[M * N];
    FLOAT *ref_C = new FLOAT[M * N];
    FLOAT *init_C = new FLOAT[M * N];

    gen_mat(M, K, A);
    gen_mat(K, N, B);
    gen_mat(M, N, C);
    memcpy(ck_C, C, sizeof(FLOAT) * M * N);
    memcpy(ref_C, C, sizeof(FLOAT) * M * N);
    memcpy(init_C, C, sizeof(FLOAT) * M * N);

    double realtime = 0.0;
    for (int i = 0; i < R; ++i) {
        realtime = now();
        TESTED_GEMM(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
        v_time.push_back(now() - realtime);
    }

    sort(v_time.begin(), v_time.end());
    realtime = v_time[0]; // 多次跑取最快的一次
    double gfl = (2.0 * M * N * K) / (realtime * 1.e9);
    printf("%%f\t Type: %d", gfl);
    // resid C
    if (check) {
        // public: int errNorm1
        RefGemm(M, N, K, alpha, A, B, ref_C);
        FLOAT errNorm1 = GenErrNorm1(C, ref_C, M, N);
        FLOAT NormA = GenNorm1(A, M, K);
        FLOAT NormB = GenNorm1(B, K, N);
        FLOAT NormC = GenNorm1(init_C, M, N);
        FLOAT resid = errNorm1 / ((NormA * NormB + NormC) * FMAX(FMAX((FLOAT)M, (FLOAT)N), (FLOAT)K) * GetEpsilon());
        if (resid > 50.0) {
            printf("error! resid = %f\n", resid);
        } else {
            printf("corr!\n");
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] ck_C;
    delete[] ref_C;
    delete[] init_C;

    return 0;
}