#pragma once

#include <complex>
#include <vector>

namespace sim {

// Crank–Nicolson ADI solver for i dpsi/dt = -(1/2) Laplacian(psi) + V psi
// Potential is allowed to be complex (for absorbing boundary sponge).
struct CrankNicolsonADI {
    // One time step in-place. psi and V are length Nx*Ny row-major.
    // dx, dy: grid spacing; dt: time step.
    void step(std::vector<std::complex<double>>& psi,
              int Nx, int Ny, double dx, double dy, double dt,
              const std::vector<std::complex<double>>& V);
};

} // namespace sim

