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
    dx = Lx / Nx;
    dy = Ly / Ny;
    psi.assign(static_cast<size_t>(Nx*Ny), std::complex<double>(0.0, 0.0));
    pfield.Nx = Nx;
    pfield.Ny = Ny;
    pfield.build(V);
}

void Simulation::clearPsi() {
    std::fill(psi.begin(), psi.end(), std::complex<double>(0.0, 0.0));
}

void Simulation::reset() {
    clearPsi();
    pfield.Nx = Nx;
    pfield.Ny = Ny;
    pfield.build(V);
    for (const auto& p : packets) injectGaussian(p);
}

void Simulation::injectGaussian(const Packet& p) {
    // Convert normalized center to indices
    double cx = p.cx * (Nx - 1);
    double cy = p.cy * (Ny - 1);
    double sigx = p.sigma * Nx; // interpret sigma in normalized domain units
    double sigy = p.sigma * Ny;
    const std::complex<double> I(0.0, 1.0);
    for (int j = 0; j < Ny; ++j) {
        double dyc = (j - cy) / sigy;
        for (int i = 0; i < Nx; ++i) {
            double dxc = (i - cx) / sigx;
            double g = std::exp(-0.5 * (dxc * dxc + dyc * dyc));
            // plane-wave factor exp(i k·r). Interpret k in radians per normalized unit.
            double x = double(i) / double(Nx - 1);
            double y = double(j) / double(Ny - 1);
            double phase = p.kx * (x - p.cx) + p.ky * (y - p.cy);
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
