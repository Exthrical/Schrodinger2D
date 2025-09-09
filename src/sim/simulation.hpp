#pragma once

#include <complex>
#include <vector>
#include <string>

#include "solver.hpp"
#include "potential.hpp"

namespace sim {

enum class ViewMode { MagnitudePhase, Real, Imag, Magnitude, Phase };

struct Simulation {
    int Nx{128}, Ny{128};
    double Lx{1.0}, Ly{1.0};      // physical domain size (arbitrary units)
    double dx{Lx/Nx}, dy{Ly/Ny};  // grid spacing
    double dt{0.001};
    bool running{false};

    // Fields
    std::vector<std::complex<double>> psi; // wavefunction
    std::vector<std::complex<double>> V;   // potential (real + i*imag for CAP)

    // Objects (for reconstructing initial conditions on reset)
    PotentialField pfield;   // includes boxes + CAP params
    std::vector<Packet> packets; // defined sources (for reset)

    // Numerics
    CrankNicolsonADI solver;

    Simulation();

    void resize(int newNx, int newNy);
    void reset();               // rebuild V and re-inject packets (psi from scratch)
    void clearPsi();            // set psi = 0
    void injectGaussian(const Packet& p); // add a Gaussian packet to psi
    void addBox(const Box& b);  // add a rectangle to potential & rebuild V

    void step();                // one CN-ADI step
    void stepN(int n);

    // Diagnostics
    double mass() const;        // discrete L2 norm integral sum |psi|^2 dx dy
    void mass_split(double& left, double& right) const; // split by vertical midline

    // Helpers
    inline int idx(int i, int j) const { return j * Nx + i; }
};

} // namespace sim

