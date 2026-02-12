#pragma once

#include <string>
#include <vector>

#include "sim/simulation.hpp"

namespace io {

struct SceneBox { double x0,y0,x1,y1,height; };
struct ScenePacket { double cx,cy,sigma,amplitude,kx,ky; };
struct SceneWell { double cx,cy,strength,radius; int profile; };

struct Scene {
    int Nx{128}, Ny{128};
    double dt{0.001};
    double cap_strength{1.0};
    double cap_ratio{0.1};
    double rel_mass_drift_tol{0.15};
    double rel_cap_mass_growth_tol{0.01};
    double rel_interior_mass_drift_tol{1.0};
    double interior_mass_drift_vs_total_tol{0.05};
    double min_initial_interior_mass_fraction{0.05};
    double min_interior_area_fraction{0.01};
    int stability_warmup_steps{8};
    bool interior_drift_hard_fail{false};
    bool auto_pause_on_instability{true};
    std::vector<SceneBox> boxes;
    std::vector<SceneWell> wells;
    std::vector<ScenePacket> packets;
    int steps{600}; // for smoke example
};

// Serialize/deserialize (minimal JSON; assumes well-formed input from our own writer)
bool save_scene(const std::string& path, const Scene& s);
bool load_scene(const std::string& path, Scene& s);

// Conversion helpers
void from_simulation(const sim::Simulation& srcSim, Scene& s);
void to_simulation(const Scene& s, sim::Simulation& dstSim);

// CLI example runner
int run_example_cli(const std::string& scene_path);

} // namespace io

