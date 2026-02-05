#include "simulation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace sim {

namespace {

static inline bool is_finite_complex(const std::complex<double>& z) {
    return std::isfinite(std::real(z)) && std::isfinite(std::imag(z));
}

} // namespace

Simulation::Simulation() {
    resize(Nx, Ny);
}

void Simulation::resize(int newNx, int newNy) {
    Nx = std::max(8, newNx);
    Ny = std::max(8, newNy);
    const int minDim = std::max(8, std::min(Nx, Ny));
    const double cell = 1.0 / static_cast<double>(minDim);
    Lx = Nx * cell;
    Ly = Ny * cell;
    dx = cell;
    dy = cell;
    psi.assign(static_cast<size_t>(Nx*Ny), std::complex<double>(0.0, 0.0));
    pfield.Nx = Nx;
    pfield.Ny = Ny;
    pfield.Lx = Lx;
    pfield.Ly = Ly;
    pfield.build(V);
    reset();
}

void Simulation::clearPsi() {
    std::fill(psi.begin(), psi.end(), std::complex<double>(0.0, 0.0));
}

void Simulation::reset() {
    clearPsi();
    pfield.Nx = Nx;
    pfield.Ny = Ny;
    pfield.Lx = Lx;
    pfield.Ly = Ly;
    pfield.build(V);
    for (const auto& p : packets) injectGaussian(p);
    refresh_diagnostics_baseline();
}

void Simulation::injectGaussian(const Packet& p) {
    // Convert normalized parameters to physical coordinates
    const double cx_phys = p.cx * Lx;
    const double cy_phys = p.cy * Ly;
    const double sig_base = std::max(1e-12, p.sigma * std::min(Lx, Ly));
    const double sigx = sig_base;
    const double sigy = sig_base;
    const std::complex<double> I(0.0, 1.0);
    for (int j = 0; j < Ny; ++j) {
        double y = (j + 0.5) * dy;
        double dyc = (y - cy_phys) / sigy;
        for (int i = 0; i < Nx; ++i) {
            double x = (i + 0.5) * dx;
            double dxc = (x - cx_phys) / sigx;
            double g = std::exp(-0.5 * (dxc * dxc + dyc * dyc));
            // plane-wave factor exp(i k·r). Interpret k in radians per unit length.
            double phase = p.kx * (x - cx_phys) + p.ky * (y - cy_phys);
            std::complex<double> w = p.amplitude * g * std::exp(I * phase);
            psi[idx(i, j)] += w;
        }
    }
    update_diagnostics(false);
}

void Simulation::addBox(const Box& b) {
    pfield.boxes.push_back(b);
    pfield.build(V);
    update_diagnostics(false);
}

void Simulation::addWell(const RadialWell& w) {
    pfield.wells.push_back(w);
    pfield.build(V);
    update_diagnostics(false);
}

void Simulation::step() {
    solver.step(psi, Nx, Ny, dx, dy, dt, V);
    update_diagnostics(true);
    if (diagnostics.unstable && stability.auto_pause_on_instability) {
        running = false;
    }
}

void Simulation::stepN(int n) {
    for (int k = 0; k < n; ++k) step();
}

double Simulation::mass() const {
    double sum = 0.0;
    for (const auto& z : psi) sum += std::norm(z);
    return sum * dx * dy;
}

double Simulation::interior_mass() const {
    const int wx = std::max(1, static_cast<int>(std::round(pfield.cap_ratio * Nx)));
    const int wy = std::max(1, static_cast<int>(std::round(pfield.cap_ratio * Ny)));

    int i0 = wx;
    int i1 = Nx - wx;
    int j0 = wy;
    int j1 = Ny - wy;
    if (i1 <= i0 || j1 <= j0) {
        i0 = 0;
        i1 = Nx;
        j0 = 0;
        j1 = Ny;
    }

    double sum = 0.0;
    for (int j = j0; j < j1; ++j) {
        for (int i = i0; i < i1; ++i) {
            sum += std::norm(psi[idx(i, j)]);
        }
    }
    return sum * dx * dy;
}

void Simulation::mass_split(double& left, double& right) const {
    left = diagnostics.left_mass;
    right = diagnostics.right_mass;
}

void Simulation::refresh_diagnostics_baseline() {
    diagnostics = StabilityDiagnostics{};
    update_diagnostics(false);
    diagnostics.initial_mass = diagnostics.current_mass;
    diagnostics.initial_interior_mass = diagnostics.current_interior_mass;
    diagnostics.rel_mass_drift = 0.0;
    diagnostics.rel_interior_mass_drift = 0.0;
    diagnostics.steps_since_baseline = 0;
    diagnostics.unstable = false;
    diagnostics.reason.clear();
}

void Simulation::update_diagnostics(bool is_time_step) {
    const int mid = Nx / 2;
    const int wx = std::max(1, static_cast<int>(std::round(pfield.cap_ratio * Nx)));
    const int wy = std::max(1, static_cast<int>(std::round(pfield.cap_ratio * Ny)));
    int i0 = wx;
    int i1 = Nx - wx;
    int j0 = wy;
    int j1 = Ny - wy;
    if (i1 <= i0 || j1 <= j0) {
        i0 = 0;
        i1 = Nx;
        j0 = 0;
        j1 = Ny;
    }

    bool finite = true;
    double total = 0.0;
    double interior = 0.0;
    double left = 0.0;
    double right = 0.0;
    for (int j = 0; j < Ny; ++j) {
        const bool insideY = (j >= j0 && j < j1);
        for (int i = 0; i < Nx; ++i) {
            const auto z = psi[idx(i, j)];
            finite = finite && is_finite_complex(z);
            const double w = std::norm(z) * dx * dy;
            total += w;
            if (i < mid) left += w; else right += w;
            if (insideY && i >= i0 && i < i1) interior += w;
        }
    }

    diagnostics.current_mass = total;
    diagnostics.current_interior_mass = interior;
    diagnostics.left_mass = left;
    diagnostics.right_mass = right;
    diagnostics.has_non_finite = !finite;

    const double massDenom = std::max(1e-15, diagnostics.initial_mass);
    const double interiorDenom = std::max(1e-15, diagnostics.initial_interior_mass);
    diagnostics.rel_mass_drift = std::fabs(total - diagnostics.initial_mass) / massDenom;
    diagnostics.rel_interior_mass_drift = std::fabs(interior - diagnostics.initial_interior_mass) / interiorDenom;
    if (is_time_step) {
        diagnostics.steps_since_baseline += 1;
    }

    if (diagnostics.unstable) return;

    if (!finite) {
        diagnostics.unstable = true;
        diagnostics.reason = "psi contains NaN/Inf";
        return;
    }

    if (diagnostics.steps_since_baseline <= std::max(0, stability.warmup_steps)) {
        return;
    }

    const bool capEnabled = pfield.cap_strength > 1e-12 && pfield.cap_ratio > 0.0;
    const double totalTol = std::max(0.0, stability.rel_mass_drift_tol);
    const double interiorTol = std::max(0.0, stability.rel_interior_mass_drift_tol);

    if (capEnabled) {
        const double allowed = diagnostics.initial_mass * (1.0 + totalTol);
        if (diagnostics.current_mass > allowed) {
            diagnostics.unstable = true;
            diagnostics.reason = "total mass grew unexpectedly with CAP";
            return;
        }
        if (diagnostics.rel_interior_mass_drift > interiorTol) {
            diagnostics.unstable = true;
            diagnostics.reason = "interior mass drift exceeded tolerance";
            return;
        }
    } else {
        if (diagnostics.rel_mass_drift > totalTol) {
            diagnostics.unstable = true;
            diagnostics.reason = "mass drift exceeded tolerance";
            return;
        }
        if (diagnostics.rel_interior_mass_drift > interiorTol) {
            diagnostics.unstable = true;
            diagnostics.reason = "interior mass drift exceeded tolerance";
            return;
        }
    }
}

static void tridiagonal_eigen(std::vector<double> diag, std::vector<double> off, std::vector<double>& evals, std::vector<std::vector<double>>& evecs) {
    const int n = static_cast<int>(diag.size());
    evals = diag;
    evecs.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) evecs[i][i] = 1.0;
    if (n == 0) return;
    off.push_back(0.0); // pad
    const double eps = std::numeric_limits<double>::epsilon();
    for (int l = 0; l < n; ++l) {
        int iter = 0;
        while (true) {
            int m = l;
            for (; m < n - 1; ++m) {
                double sum = std::fabs(evals[m]) + std::fabs(evals[m + 1]);
                if (std::fabs(off[m]) <= eps * sum) break;
            }
            if (m == l) break;
            if (++iter > 60) break;
            double g = (evals[l + 1] - evals[l]) / (2.0 * off[l]);
            double r = std::hypot(g, 1.0);
            g = evals[m] - evals[l] + off[l] / (g + std::copysign(r, g));
            double s = 1.0, c = 1.0, p = 0.0;
            for (int i = m - 1; i >= l; --i) {
                double f = s * off[i];
                double b = c * off[i];
                if (std::fabs(f) >= std::fabs(g)) {
                    c = g / f;
                    r = std::hypot(c, 1.0);
                    off[i + 1] = f * r;
                    s = 1.0 / r;
                    c *= s;
                } else {
                    s = f / g;
                    r = std::hypot(s, 1.0);
                    off[i + 1] = g * r;
                    c = 1.0 / r;
                    s *= c;
                }
                g = evals[i + 1] - p;
                r = (evals[i] - g) * s + 2.0 * b * c;
                p = s * r;
                evals[i + 1] = g + p;
                g = c * r - b;
                for (int k = 0; k < n; ++k) {
                    double fz = evecs[k][i + 1];
                    evecs[k][i + 1] = s * evecs[k][i] + c * fz;
                    evecs[k][i] = c * evecs[k][i] - s * fz;
                }
            }
            evals[l] -= p;
            off[l] = g;
            off[m] = 0.0;
        }
    }
}

std::vector<EigenState> Simulation::compute_eigenstates(int modes, int maxBasis, int maxIter, double tol) const {
    const int N = Nx * Ny;
    modes = std::max(1, modes);
    maxBasis = std::max(modes, maxBasis);
    maxIter = std::max(maxIter, modes);
    const int maxSteps = std::min(maxBasis, maxIter);
    const double h = dx; // dx == dy by construction
    const double vol = dx * dy;

    auto applyH = [&](const std::vector<double>& x, std::vector<double>& y) {
        y.assign(N, 0.0);
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const int k = idx(i, j);
                double center = x[k];
                double lap = 0.0;
                if (i > 0) lap += x[idx(i - 1, j)];
                if (i < Nx - 1) lap += x[idx(i + 1, j)];
                if (j > 0) lap += x[idx(i, j - 1)];
                if (j < Ny - 1) lap += x[idx(i, j + 1)];
                lap -= 4.0 * center;
                lap /= (h * h);
                double v = std::real(V[k]);
                y[k] = -0.5 * lap + v * center;
            }
        }
    };

    auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += a[i] * b[i];
        return s * vol;
    };

    // initial vector
    std::vector<double> q(N, 0.0), q_prev(N, 0.0);
    for (int i = 0; i < N; i += std::max(1, N / 50)) q[i] = 1.0;
    double nrm = std::sqrt(dot(q, q));
    if (nrm < 1e-12) q[0] = 1.0, nrm = 1.0;
    for (double& v : q) v /= nrm;

    std::vector<double> alphas;
    std::vector<double> betas;
    std::vector<std::vector<double>> basis;
    basis.reserve(maxBasis);
    basis.push_back(q);

    std::vector<double> w(N, 0.0);
    double beta = 0.0;
    for (int iter = 0; iter < maxSteps; ++iter) {
        applyH(q, w);
        for (int i = 0; i < N; ++i) w[i] -= beta * q_prev[i];
        double alpha = dot(q, w);
        for (int i = 0; i < N; ++i) w[i] -= alpha * q[i];
        double nextBeta = std::sqrt(dot(w, w));
        alphas.push_back(alpha);
        if (iter > 0) betas.push_back(beta);
        if (nextBeta < tol) {
            break;
        }
        q_prev.swap(q);
        q.swap(w);
        for (double& v : q) v /= nextBeta;
        basis.push_back(q);
        beta = nextBeta;
    }
    const int m = static_cast<int>(alphas.size());
    if (m == 0) return {};

    std::vector<double> evals;
    std::vector<std::vector<double>> evecsSmall;
    tridiagonal_eigen(alphas, betas, evals, evecsSmall);

    std::vector<int> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return evals[a] < evals[b]; });

    int take = std::min(modes, m);
    std::vector<EigenState> results;
    results.reserve(take);
    for (int t = 0; t < take; ++t) {
        int idxMode = order[t];
        std::vector<double> phi(N, 0.0);
        for (int b = 0; b < m; ++b) {
            double coeff = evecsSmall[b][idxMode];
            const auto& qb = basis[b];
            for (int i = 0; i < N; ++i) {
                phi[i] += coeff * qb[i];
            }
        }
        double norm = std::sqrt(dot(phi, phi));
        if (norm < 1e-12) continue;
        double invNorm = 1.0 / norm;
        EigenState es;
        es.energy = evals[idxMode];
        es.psi.resize(N);
        for (int i = 0; i < N; ++i) es.psi[i] = phi[i] * invNorm;
        results.push_back(std::move(es));
    }
    return results;
}

void Simulation::apply_eigenstate(const EigenState& state) {
    if (static_cast<int>(state.psi.size()) != Nx * Ny) return;
    psi = state.psi;
    packets.clear();
    running = false;
    refresh_diagnostics_baseline();
}

} // namespace sim
