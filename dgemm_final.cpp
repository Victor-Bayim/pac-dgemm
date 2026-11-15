// Maintenance implementation, 2026-11-15. Not the recovered competition binary.
// See NOTICE.md and docs/MAINTENANCE.md for provenance and changes.
#include "pac_dgemm.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#if defined(__aarch64__) && !defined(PAC_FORCE_SCALAR)
#include <arm_neon.h>
#define PAC_NEON 1
#else
#define PAC_NEON 0
#endif
#if defined(_OPENMP) && !defined(PAC_DISABLE_OPENMP)
#include <omp.h>
#define PAC_OPENMP 1
#else
#define PAC_OPENMP 0
#endif

namespace {
constexpr int tile = 4;
constexpr int block_m = 128, block_n = 128, block_k = 128;
struct Workspace {
    std::array<double, block_m * block_k> a;
    std::array<double, block_k * block_n> b;
};
bool valid_span(int32_t rows, int32_t cols, int32_t ld) {
    if (rows == 0 || cols == 0) return true;
    const auto limit = static_cast<std::uint64_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    return static_cast<std::uint64_t>(cols - 1) * static_cast<std::uint64_t>(ld)
           + static_cast<std::uint64_t>(rows) <= limit;
}
// Consecutive [depth][4 rows] micro-panels, including zero-padded tails.
void pack_a(int rows, int depth, const double* a, int32_t lda, double* packed) {
    for (int base = 0; base < rows; base += tile) {
        double* panel = packed + (base / tile) * depth * tile;
        for (int l = 0; l < depth; ++l)
            for (int r = 0; r < tile; ++r)
                panel[l * tile + r] = base + r < rows
                    ? a[base + r + static_cast<std::ptrdiff_t>(l) * lda] : 0.0;
    }
}
// Consecutive [depth][4 columns] micro-panels, not whole-panel row strides.
void pack_b(int depth, int cols, const double* b, int32_t ldb, double* packed) {
    for (int base = 0; base < cols; base += tile) {
        double* panel = packed + (base / tile) * depth * tile;
        for (int l = 0; l < depth; ++l)
            for (int c = 0; c < tile; ++c)
                panel[l * tile + c] = base + c < cols
                    ? b[l + static_cast<std::ptrdiff_t>(base + c) * ldb] : 0.0;
    }
}
void kernel_4x4(int depth, double alpha, const double* a, const double* b,
                double* c, int32_t ldc) {
#if PAC_NEON
    float64x2_t lo[4], hi[4];
    for (int col = 0; col < tile; ++col) {
        lo[col] = vdupq_n_f64(0.0);
        hi[col] = vdupq_n_f64(0.0);
    }
    for (int l = 0; l < depth; ++l) {
        const float64x2_t a_lo = vld1q_f64(a + l * tile);
        const float64x2_t a_hi = vld1q_f64(a + l * tile + 2);
        for (int col = 0; col < tile; ++col) {
            const float64x2_t v = vdupq_n_f64(b[l * tile + col]);
            lo[col] = vfmaq_f64(lo[col], a_lo, v);
            hi[col] = vfmaq_f64(hi[col], a_hi, v);
        }
    }
    const float64x2_t scale = vdupq_n_f64(alpha);
    for (int col = 0; col < tile; ++col) {
        double* dst = c + static_cast<std::ptrdiff_t>(col) * ldc;
        vst1q_f64(dst, vfmaq_f64(vld1q_f64(dst), lo[col], scale));
        vst1q_f64(dst + 2, vfmaq_f64(vld1q_f64(dst + 2), hi[col], scale));
    }
#else
    double sums[tile * tile] = {};
    for (int l = 0; l < depth; ++l)
        for (int col = 0; col < tile; ++col)
            for (int row = 0; row < tile; ++row)
                sums[row + col * tile] += a[l * tile + row] * b[l * tile + col];
    for (int col = 0; col < tile; ++col)
        for (int row = 0; row < tile; ++row)
            c[row + static_cast<std::ptrdiff_t>(col) * ldc] += alpha * sums[row + col * tile];
#endif
}
void scale_c(int32_t rows, int32_t cols, double beta, double* c, int32_t ldc) {
    if (beta == 1.0) return;
    for (int32_t j = 0; j < cols; ++j)
        for (int32_t i = 0; i < rows; ++i) {
            double& value = c[i + static_cast<std::ptrdiff_t>(j) * ldc];
            value = beta == 0.0 ? 0.0 : beta * value;
        }
}
} // namespace

extern "C" const char* pac_dgemm_backend(void) {
    return PAC_NEON ? "aarch64-neon" : "portable-scalar";
}
extern "C" int pac_dgemm_openmp_enabled(void) { return PAC_OPENMP; }
extern "C" int pac_dgemm_colmajor(int32_t m, int32_t n, int32_t k, double alpha,
    const double* a, int32_t lda, const double* b, int32_t ldb,
    double beta, double* c, int32_t ldc) {
    if (m < 0 || n < 0 || k < 0 || lda < std::max<int32_t>(1, m)
        || ldb < std::max<int32_t>(1, k) || ldc < std::max<int32_t>(1, m))
        return PAC_GEMM_INVALID_ARGUMENT;
    if (m == 0 || n == 0) return PAC_GEMM_SUCCESS;
    if (!c || !valid_span(m, n, ldc)) return PAC_GEMM_INVALID_ARGUMENT;
    if (alpha == 0.0 || k == 0) {
        scale_c(m, n, beta, c, ldc);
        return PAC_GEMM_SUCCESS;
    }
    if (!a || !b || !valid_span(m, k, lda) || !valid_span(k, n, ldb))
        return PAC_GEMM_INVALID_ARGUMENT;
    const int64_t row_blocks = (static_cast<int64_t>(m) + block_m - 1) / block_m;
    int threads = 1;
#if PAC_OPENMP
    threads = static_cast<int>(std::min<int64_t>(row_blocks, omp_get_max_threads()));
#endif
    // Allocate before changing C; no allocation or exception crosses an OpenMP region.
    std::vector<Workspace> workspace;
    try { workspace.resize(static_cast<std::size_t>(threads)); }
    catch (const std::bad_alloc&) { return PAC_GEMM_ALLOCATION_FAILURE; }
#if PAC_OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t block = 0; block < row_blocks; ++block) {
        int thread = 0;
#if PAC_OPENMP
        thread = omp_get_thread_num();
#endif
        double* packed_a = workspace[static_cast<std::size_t>(thread)].a.data();
        double* packed_b = workspace[static_cast<std::size_t>(thread)].b.data();
        const int32_t i0 = static_cast<int32_t>(block * block_m);
        const int rows = std::min<int32_t>(block_m, m - i0);
        scale_c(rows, n, beta, c + i0, ldc);
        for (int64_t k0 = 0; k0 < k; k0 += block_k) {
            const int depth = static_cast<int>(std::min<int64_t>(block_k, k - k0));
            pack_a(rows, depth, a + i0 + k0 * lda, lda, packed_a);
            for (int64_t j0 = 0; j0 < n; j0 += block_n) {
                const int cols = static_cast<int>(std::min<int64_t>(block_n, n - j0));
                pack_b(depth, cols, b + k0 + j0 * ldb, ldb, packed_b);
                for (int j = 0; j < cols; j += tile)
                    for (int i = 0; i < rows; i += tile) {
                        const double* ap = packed_a + (i / tile) * depth * tile;
                        const double* bp = packed_b + (j / tile) * depth * tile;
                        double* dst = c + i0 + i + (j0 + j) * ldc;
                        if (i + tile <= rows && j + tile <= cols) {
                            kernel_4x4(depth, alpha, ap, bp, dst, ldc);
                        } else {
                            double tail[tile * tile] = {};
                            kernel_4x4(depth, alpha, ap, bp, tail, tile);
                            for (int col = 0; col < std::min(tile, cols - j); ++col)
                                for (int row = 0; row < std::min(tile, rows - i); ++row)
                                    dst[row + static_cast<std::ptrdiff_t>(col) * ldc] += tail[row + col * tile];
                        }
                    }
            }
        }
    }
    return PAC_GEMM_SUCCESS;
}
