// Simple complex tridiagonal solver (Thomas algorithm)
#pragma once

#include <complex>
#include <vector>

namespace sim {

// Solves Ax = d for a tridiagonal A given by (a,b,c) where:
// a[0] is unused (0), c[N-1] is unused (0)
// b is main diagonal, a is sub-diagonal, c is super-diagonal
// All vectors length N. d is overwritten with the solution x.
inline void solve_tridiagonal(
    std::vector<std::complex<double>>& a,
    std::vector<std::complex<double>>& b,
    std::vector<std::complex<double>>& c,
    std::vector<std::complex<double>>& d)
{
    const int n = static_cast<int>(b.size());
    for (int i = 1; i < n; ++i) {
        std::complex<double> w = a[i] / b[i - 1];
        b[i] = b[i] - w * c[i - 1];
        d[i] = d[i] - w * d[i - 1];
    }
    d[n - 1] /= b[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        d[i] = (d[i] - c[i] * d[i + 1]) / b[i];
    }
}

} // namespace sim

