#include "potential.hpp"

#include <algorithm>
#include <cmath>

namespace sim {

static inline int idx(int i, int j, int Nx) { return j * Nx + i; }

void PotentialField::build(std::vector<std::complex<double>>& V) const {
    using cd = std::complex<double>;
    V.assign(static_cast<size_t>(Nx*Ny), cd(0.0, 0.0));

    // Add rectangular boxes (real potential)
    for (const auto& b : boxes) {
        int ix0 = std::max(0, std::min(Nx-1, (int)std::floor(b.x0 * Nx)));
        int ix1 = std::max(0, std::min(Nx-1, (int)std::floor(b.x1 * Nx)));
        int iy0 = std::max(0, std::min(Ny-1, (int)std::floor(b.y0 * Ny)));
        int iy1 = std::max(0, std::min(Ny-1, (int)std::floor(b.y1 * Ny)));
        if (ix1 < ix0) std::swap(ix0, ix1);
        if (iy1 < iy0) std::swap(iy0, iy1);
        for (int j = iy0; j <= iy1; ++j) {
            for (int i = ix0; i <= ix1; ++i) {
                V[idx(i,j,Nx)] += cd(b.height, 0.0);
            }
        }
    }

    // Add smooth radial wells
    const double minLength = std::min(Lx, Ly);

    for (const auto& w : wells) {
        double r0 = std::max(1e-4, w.radius * minLength);
        double r0sq = r0 * r0;
        double cx = w.cx * Lx;
        double cy = w.cy * Ly;
        for (int j = 0; j < Ny; ++j) {
            double y = (j + 0.5) * (Ly / Ny);
            double dy = y - cy;
            for (int i = 0; i < Nx; ++i) {
                double x = (i + 0.5) * (Lx / Nx);
                double dx = x - cx;
                double r2 = dx * dx + dy * dy;
                double contrib = 0.0;
                switch (w.profile) {
                case RadialWell::Profile::Gaussian: {
                    double t = r2 / r0sq;
                    contrib = w.strength * std::exp(-t);
                    break;
                }
                case RadialWell::Profile::SoftCoulomb: {
                    contrib = w.strength / std::sqrt(r2 + r0sq);
                    break;
                }
                case RadialWell::Profile::InverseSquare: {
                    contrib = w.strength / (r2 + r0sq);
                    break;
                }
                }
                V[idx(i,j,Nx)] += cd(contrib, 0.0);
            }
        }
    }

    // Add complex absorbing potential (CAP) sponge near boundaries
    // A smooth polynomial ramp: w(s) = s^2 (3 - 2s) in [0,1]
    const int wx = std::max(1, (int)std::round(cap_ratio * Nx));
    const int wy = std::max(1, (int)std::round(cap_ratio * Ny));
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            double sx = 0.0;
            if (i < wx) sx = double(wx - i) / double(wx);
            else if (i >= Nx - wx) sx = double(i - (Nx - wx - 1)) / double(wx);
            double sy = 0.0;
            if (j < wy) sy = double(wy - j) / double(wy);
            else if (j >= Ny - wy) sy = double(j - (Ny - wy - 1)) / double(wy);
            double s = std::max(sx, sy);
            if (s > 0.0) {
                double ramp = s * s * (3.0 - 2.0 * s); // smoothstep
                double absorb = cap_strength * ramp * ramp; // stronger near edges
                V[idx(i,j,Nx)] += cd(0.0, -absorb); // -i*absorb (imaginary negative)
            }
        }
    }
}

} // namespace sim
