#pragma once

#include <complex>
#include <vector>
#include <string>

#include "solver.hpp"
#include "potential.hpp"

namespace sim {

enum class ViewMode { MagnitudePhase, Real, Imag, Magnitude, Phase };

struct EigenState {
    double energy{0.0};
    std::vector<std::complex<double>> psi;
};

struct Simulation {
    int Nx{372}, Ny{300};
    double Lx{1.0}, Ly{1.0};      // physical domain size (arbitrary units)
    double dx{Lx/Nx}, dy{Ly/Ny};  // grid spacing
    double dt{0.0001};
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
    void addWell(const RadialWell& w); // add a smooth radial feature

    void step();                // one CN-ADI step
    void stepN(int n);

    // Diagnostics
    double mass() const;        // discrete L2 norm integral sum |psi|^2 dx dy
    void mass_split(double& left, double& right) const; // split by vertical midline

    // Eigenmodes of the current Hamiltonian (real part of V, Dirichlet boundary)
    std::vector<EigenState> compute_eigenstates(int modes, int maxBasis = 64, int maxIter = 200, double tol = 1e-6) const;
    void apply_eigenstate(const EigenState& state);

    // Helpers
    inline int idx(int i, int j) const { return j * Nx + i; }
};

} // namespace sim
