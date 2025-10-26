#include "simulation.hpp"

#include <algorithm>
#include <cmath>

namespace sim {

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
}

void Simulation::addBox(const Box& b) {
    pfield.boxes.push_back(b);
    pfield.build(V);
}

void Simulation::addWell(const RadialWell& w) {
    pfield.wells.push_back(w);
    pfield.build(V);
}

void Simulation::step() {
    solver.step(psi, Nx, Ny, dx, dy, dt, V);
}

void Simulation::stepN(int n) {
    for (int k = 0; k < n; ++k) step();
}

double Simulation::mass() const {
    double sum = 0.0;
    for (const auto& z : psi) sum += std::norm(z);
    return sum * dx * dy;
}

void Simulation::mass_split(double& left, double& right) const {
    int mid = Nx / 2;
    double l = 0.0, r = 0.0;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            double w = std::norm(psi[idx(i,j)]) * dx * dy;
            if (i < mid) l += w; else r += w;
        }
    }
    left = l; right = r;
}

} // namespace sim
