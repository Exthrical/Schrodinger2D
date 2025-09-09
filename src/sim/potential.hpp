#pragma once

#include <complex>
#include <vector>

namespace sim {

struct Box {
    // Normalized [0,1] screen space rectangle
    double x0, y0, x1, y1; // axis-aligned
    double height;         // potential height (positive=barrier, negative=well)
};

struct Packet {
    // Normalized [0,1] center, width in normalized units, amplitude, momentum vector in normalized grid units
    double cx, cy;
    double sigma;     // Gaussian width (relative to domain size)
    double amplitude; // amplitude of packet
    double kx, ky;    // initial momentum (radians per unit length)
};

// Potential field: sum of static boxes + absorbing boundary sponge
struct PotentialField {
    int Nx{128}, Ny{128};
    double cap_strength{1.0};  // absorption coefficient
    double cap_ratio{0.1};     // fraction of domain width used for sponge (each side)
    std::vector<Box> boxes;     // static rectangular features

    // compute complex potential V(i,j)
    void build(std::vector<std::complex<double>>& V) const;
};

} // namespace sim

