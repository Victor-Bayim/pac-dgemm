#include "pac_dgemm.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr std::size_t guard = 16;
constexpr double sentinel = 987654.125;
int cases = 0;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(int m, int n, int k, double alpha, double beta, int padding,
           bool nan_c = false) {
    const int lda = std::max(1, m) + padding;
    const int ldb = std::max(1, k) + padding;
    const int ldc = std::max(1, m) + padding;
    std::vector<double> a(2 * guard + static_cast<std::size_t>(lda) * k, sentinel);
    std::vector<double> b(2 * guard + static_cast<std::size_t>(ldb) * n, sentinel);
    std::vector<double> c(2 * guard + static_cast<std::size_t>(ldc) * n, sentinel);
    std::mt19937_64 rng(20261115 + cases);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (int l = 0; l < k; ++l) for (int i = 0; i < m; ++i)
        a[guard + i + static_cast<std::size_t>(l) * lda] = dist(rng);
    for (int j = 0; j < n; ++j) for (int l = 0; l < k; ++l)
        b[guard + l + static_cast<std::size_t>(j) * ldb] = dist(rng);
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i)
        c[guard + i + static_cast<std::size_t>(j) * ldc] = nan_c
            ? std::numeric_limits<double>::quiet_NaN() : dist(rng);
    const auto original_a = a, original_b = b, original_c = c;
    const int status = pac_dgemm_colmajor(m, n, k, alpha, a.data() + guard, lda,
        b.data() + guard, ldb, beta, c.data() + guard, ldc);
    require(status == PAC_GEMM_SUCCESS, "valid call rejected");
    require(a == original_a && b == original_b, "input modified");
    for (std::size_t p = 0; p < c.size(); ++p) {
        const bool outer = p < guard || p >= c.size() - guard;
        const bool pad = !outer && ((p - guard) % static_cast<std::size_t>(ldc))
            >= static_cast<std::size_t>(m);
        if (outer || pad) require(c[p] == sentinel, "C guard or padding overwritten");
    }
    // Independent column-major reference: no packing, blocking, SIMD or shared kernel.
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
        long double sum = 0.0L;
        for (int l = 0; l < k; ++l)
            sum += static_cast<long double>(a[guard + i + static_cast<std::size_t>(l) * lda])
                   * b[guard + l + static_cast<std::size_t>(j) * ldb];
        const std::size_t offset = guard + i + static_cast<std::size_t>(j) * ldc;
        const long double want = static_cast<long double>(alpha) * sum
            + (beta == 0.0 ? 0.0L : static_cast<long double>(beta) * original_c[offset]);
        require(std::isfinite(c[offset]), "non-finite result");
        require(std::abs(static_cast<long double>(c[offset]) - want)
            <= 2e-11L * std::max(1.0L, std::abs(want)), "reference mismatch");
    }
    ++cases;
}
void api_cases() {
    double a = 2.0, b = 3.0, c = 7.0;
    require(pac_dgemm_colmajor(-1, 1, 1, 1, &a, 1, &b, 1, 1, &c, 1)
        == PAC_GEMM_INVALID_ARGUMENT && c == 7.0, "negative dimension");
    require(pac_dgemm_colmajor(1, 1, 1, 1, &a, 0, &b, 1, 1, &c, 1)
        == PAC_GEMM_INVALID_ARGUMENT && c == 7.0, "invalid lda");
    require(pac_dgemm_colmajor(1, 1, 1, 1, &a, 1, &b, 0, 1, &c, 1)
        == PAC_GEMM_INVALID_ARGUMENT && c == 7.0, "invalid ldb");
    require(pac_dgemm_colmajor(1, 1, 1, 1, &a, 1, &b, 1, 1, &c, 0)
        == PAC_GEMM_INVALID_ARGUMENT && c == 7.0, "invalid ldc");
    require(pac_dgemm_colmajor(1, 1, 1, 1, nullptr, 1, &b, 1, 1, &c, 1)
        == PAC_GEMM_INVALID_ARGUMENT && c == 7.0, "missing A");
    require(pac_dgemm_colmajor(1, 1, 1, 1, &a, 1, nullptr, 1, 1, &c, 1)
        == PAC_GEMM_INVALID_ARGUMENT && c == 7.0, "missing B");
    require(pac_dgemm_colmajor(1, 1, 1, 1, &a, 1, &b, 1, 1, nullptr, 1)
        == PAC_GEMM_INVALID_ARGUMENT, "missing C");
    const int huge = std::numeric_limits<int32_t>::max();
    require(pac_dgemm_colmajor(huge, huge, 1, 1, &a, huge, &b, 1, 1, &c, huge)
        == PAC_GEMM_INVALID_ARGUMENT && c == 7.0, "unrepresentable C span");
    require(pac_dgemm_colmajor(0, 2, 1, 1, nullptr, 1, nullptr, 1, 1, nullptr, 1)
        == PAC_GEMM_SUCCESS, "empty M must not access storage");
    require(pac_dgemm_colmajor(2, 0, 1, 1, nullptr, 2, nullptr, 1, 1, nullptr, 2)
        == PAC_GEMM_SUCCESS, "empty N must not access storage");
    require(pac_dgemm_colmajor(1, 1, 5, 0, nullptr, 1, nullptr, 5, -2, &c, 1)
        == PAC_GEMM_SUCCESS && c == -14.0, "alpha zero must ignore A/B");
    c = std::numeric_limits<double>::quiet_NaN();
    require(pac_dgemm_colmajor(1, 1, 0, 1, nullptr, 1, nullptr, 1, 0, &c, 1)
        == PAC_GEMM_SUCCESS && c == 0.0, "empty K beta zero");
    cases += 12;
}
void repeat_reset() {
    const double a[] = {1, 2, 3, 4, 5, 6}; // 3x2, column-major
    const double b[] = {2, -1, 3, 4};       // 2x2
    const std::vector<double> initial = {1, 2, 3, 4, 5, 6};
    std::vector<double> first;
    for (int repeat = 0; repeat < 4; ++repeat) {
        auto c = initial; // Each measurement must compute the same mathematical input.
        require(pac_dgemm_colmajor(3, 2, 2, 0.75, a, 3, b, 2, -0.25, c.data(), 3)
            == PAC_GEMM_SUCCESS, "repeat call failed");
        if (repeat == 0) first = c;
        else require(c == first, "repeated calls with restored C differ");
    }
    ++cases;
}
} // namespace

int main() {
    try {
        const int shapes[][3] = {
            {1,1,1}, {3,5,7}, {8,4,2}, {5,9,11}, {31,17,33},
            {129,131,133}, {257,9,65}, {4,4,0}, {0,5,7}, {5,0,7},
            // Final block extents of the four archived contest shapes.
            {61,12,96}, {122,8,96}, {128,96,96}, {96,16,61}
        };
        const double scales[][2] = {{1,0}, {1,1}, {-0.5,2}, {0,-1}, {0.75,-0.25}};
        for (const auto& shape : shapes) for (const auto& scale : scales)
            for (const int padding : {0, 3})
                check(shape[0], shape[1], shape[2], scale[0], scale[1], padding);
        check(7, 9, 5, 1.0, 0.0, 5, true);
        api_cases();
        repeat_reset();
        std::cout << "PASS cases=" << cases << " backend=" << pac_dgemm_backend()
                  << " openmp=" << pac_dgemm_openmp_enabled() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
