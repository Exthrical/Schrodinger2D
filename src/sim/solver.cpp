#include "solver.hpp"
#include "tridiag.hpp"

#include <algorithm>
#include <cmath>

namespace sim {

static inline int idx(int i, int j, int Nx) { return j * Nx + i; }

void CrankNicolsonADI::step(std::vector<std::complex<double>>& psi,
                            int Nx, int Ny, double dx, double dy, double dt,
                            const std::vector<std::complex<double>>& V)
{
    using cd = std::complex<double>;
    const cd I(0.0, 1.0);

    // Potential half-step: psi <- exp(-i V dt/2) psi
    const double half = 0.5;
    const double half_dt = half * dt;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const int k = idx(i, j, Nx);
            psi[k] *= std::exp(-I * V[k] * half_dt);
        }
    }

    // ADI for kinetic term (CN): alpha = i dt / 4
    const cd alpha = I * (dt * 0.25);
    const cd ax = alpha / (dx * dx);
    const cd ay = alpha / (dy * dy);

    // Temporary buffer
    std::vector<cd> phi(psi.size());

    // 1) Solve along x: (I - alpha D_x) phi = (I + alpha D_y) psi
    std::vector<cd> a(Nx), b(Nx), c(Nx), d(Nx);
    for (int j = 0; j < Ny; ++j) {
        // Build RHS: (I + ay * D_y) psi
        for (int i = 0; i < Nx; ++i) {
            const cd center = psi[idx(i, j, Nx)];
            cd up = (j > 0) ? psi[idx(i, j - 1, Nx)] : cd(0.0, 0.0);
            cd dn = (j < Ny - 1) ? psi[idx(i, j + 1, Nx)] : cd(0.0, 0.0);
            cd Dy_center = (up - cd(2.0, 0.0) * center + dn) / (dy * dy);
            d[i] = center + alpha * Dy_center; // (I + alpha D_y)
        }

        // Build tridiagonal coefficients for (I - alpha D_x)
        for (int i = 0; i < Nx; ++i) {
            a[i] = (i == 0) ? cd(0.0, 0.0) : -ax; // sub-diagonal
            b[i] = cd(1.0, 0.0) + cd(2.0, 0.0) * ax; // main diag
            c[i] = (i == Nx - 1) ? cd(0.0, 0.0) : -ax; // super-diagonal
        }
        // Solve row
        solve_tridiagonal(a, b, c, d);
        // Store into phi
        for (int i = 0; i < Nx; ++i) {
            phi[idx(i, j, Nx)] = d[i];
        }
    }

    // 2) Solve along y: (I - alpha D_y) psi_new = (I + alpha D_x) phi
    std::vector<cd> ay_a(Ny), ay_b(Ny), ay_c(Ny), rhs(Ny);
    for (int i = 0; i < Nx; ++i) {
        // RHS: (I + ax * D_x) phi
        for (int j = 0; j < Ny; ++j) {
            const cd center = phi[idx(i, j, Nx)];
            cd lf = (i > 0) ? phi[idx(i - 1, j, Nx)] : cd(0.0, 0.0);
            cd rt = (i < Nx - 1) ? phi[idx(i + 1, j, Nx)] : cd(0.0, 0.0);
            cd Dx_center = (lf - cd(2.0, 0.0) * center + rt) / (dx * dx);
            rhs[j] = center + alpha * Dx_center; // (I + alpha D_x)
        }
        // Tridiagonal for y
        for (int j = 0; j < Ny; ++j) {
            ay_a[j] = (j == 0) ? cd(0.0, 0.0) : -ay;
            ay_b[j] = cd(1.0, 0.0) + cd(2.0, 0.0) * ay;
            ay_c[j] = (j == Ny - 1) ? cd(0.0, 0.0) : -ay;
        }
        solve_tridiagonal(ay_a, ay_b, ay_c, rhs);
        for (int j = 0; j < Ny; ++j) {
            psi[idx(i, j, Nx)] = rhs[j];
        }
    }

    // Potential half-step again
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            const int k = idx(i, j, Nx);
            psi[k] *= std::exp(-I * V[k] * half_dt);
        }
    }
}

} // namespace sim

