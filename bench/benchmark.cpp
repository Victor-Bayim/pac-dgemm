// New maintenance measurements only. Archived competition CSVs are never overwritten.
#include "pac_dgemm.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void run(int m, int n, int k, int rounds, int warmup) {
    const double alpha = 0.75, beta = -0.25;
    std::vector<double> a(static_cast<std::size_t>(m) * k);
    std::vector<double> b(static_cast<std::size_t>(k) * n);
    std::vector<double> initial(static_cast<std::size_t>(m) * n), c(initial.size());
    std::mt19937_64 rng(20261115);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& value : a) value = dist(rng);
    for (auto& value : b) value = dist(rng);
    for (auto& value : initial) value = dist(rng);
    const auto invoke = [&]() {
        if (pac_dgemm_colmajor(m,n,k,alpha,a.data(),m,b.data(),k,beta,c.data(),m)
            != PAC_GEMM_SUCCESS) throw std::runtime_error("DGEMM call failed");
    };
    const bool full_check = 2.0 * m * n * k <= 2e7;
    std::vector<std::pair<int,int>> points;
    if (full_check) {
        for (int j=0; j<n; ++j) for (int i=0; i<m; ++i) points.emplace_back(i,j);
    } else {
        for (int i : {0,m-1}) for (int j : {0,n-1}) points.emplace_back(i,j);
        for (int count=0; count<60; ++count)
            points.emplace_back(static_cast<int>(rng()%m),static_cast<int>(rng()%n));
    }
    std::vector<long double> expected;
    for (const auto& point : points) {
        long double sum=0;
        for (int l=0; l<k; ++l)
            sum += static_cast<long double>(a[point.first+static_cast<std::size_t>(l)*m])
                   * b[l+static_cast<std::size_t>(point.second)*k];
        expected.push_back(alpha*sum+beta*initial[point.first+static_cast<std::size_t>(point.second)*m]);
    }
    const auto check = [&]() {
        for (std::size_t p=0;p<points.size();++p) {
            const auto point=points[p];
            const double got=c[point.first+static_cast<std::size_t>(point.second)*m];
            if (!std::isfinite(got) || std::abs(got-expected[p])>2e-10L*std::max(1.0L,std::abs(expected[p])))
                throw std::runtime_error("benchmark reference check failed");
        }
    };
    for (int i=0;i<warmup;++i) { c=initial; invoke(); check(); }
    std::cout << "# shape=" << m << 'x' << n << 'x' << k << " check="
              << (full_check?"all outputs":"64 sampled outputs")
              << " alpha=0.75 beta=-0.25 warmup=" << warmup << '\n';
    std::vector<double> durations;
    for (int i=0;i<rounds;++i) {
        c=initial; // Restore C outside the measured interval, including for beta != 0.
        const auto start=std::chrono::steady_clock::now();
        invoke();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        check();
        durations.push_back(seconds);
        std::cout << "sample," << m << ',' << n << ',' << k << ',' << i+1 << ','
                  << seconds << ',' << (2.0*m*n*k/(seconds*1e9)) << '\n';
    }
    std::sort(durations.begin(),durations.end());
    const auto count=durations.size();
    const double median=count%2?durations[count/2]:(durations[count/2-1]+durations[count/2])/2;
    std::cout << "median," << m << ',' << n << ',' << k << ",0," << median << ','
              << (2.0*m*n*k/(median*1e9)) << '\n';
    std::cout << "minimum," << m << ',' << n << ',' << k << ",0," << durations[0] << ','
              << (2.0*m*n*k/(durations[0]*1e9)) << '\n';
}
int main(int argc,char**argv) {
    try {
        int m=129,n=131,k=133,rounds=5,warmup=1;
        bool contest=false;
        for(int i=1;i<argc;++i) {
            const std::string option=argv[i];
            if(option=="--contest") { contest=true; continue; }
            if(option!="--m" && option!="--n" && option!="--k" && option!="--rounds" && option!="--warmup")
                throw std::runtime_error("use --m N --n N --k N --rounds N --warmup N or --contest");
            if(++i==argc) throw std::runtime_error("missing numeric argument");
            std::size_t end=0;
            const auto value=std::stoll(argv[i],&end);
            if(end!=std::string(argv[i]).size() || value<0 || value>std::numeric_limits<int32_t>::max())
                throw std::runtime_error("invalid integer argument");
            if(option=="--m")m=static_cast<int>(value);
            if(option=="--n")n=static_cast<int>(value);
            if(option=="--k")k=static_cast<int>(value);
            if(option=="--rounds")rounds=static_cast<int>(value);
            if(option=="--warmup")warmup=static_cast<int>(value);
        }
        if(m<1||n<1||k<1||rounds<1||rounds>1000||warmup>1000)
            throw std::runtime_error("dimensions must be positive; rounds 1..1000, warmup 0..1000");
        std::cout << std::setprecision(12)
                  << "# origin=maintenance implementation; not historical PAC results\n# backend="
                  << pac_dgemm_backend() << " openmp=" << pac_dgemm_openmp_enabled()
                  << "\nrecord,m,n,k,round,seconds,gflops\n";
        if(contest) {
            const int shapes[][3]={{12349,140,3040},{24698,8,3040},{256,3040,3040},{3040,16,12349}};
            for(const auto& shape:shapes)run(shape[0],shape[1],shape[2],rounds,warmup);
        } else run(m,n,k,rounds,warmup);
        return 0;
    } catch(const std::exception& error) { std::cerr << "ERROR: " << error.what() << '\n'; return 1; }
}
