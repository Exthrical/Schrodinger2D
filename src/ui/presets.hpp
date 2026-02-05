#pragma once

#if BUILD_GUI

#include "sim/simulation.hpp"

namespace ui::presets {

void load_default_twowall_scene(sim::Simulation& sim);
void load_default_doubleslit_scene(sim::Simulation& sim);
void load_default_doubleslit2_scene(sim::Simulation& sim);
void load_counterpropagating_scene(sim::Simulation& sim);
void load_waveguide_scene(sim::Simulation& sim);
void load_trap_scene(sim::Simulation& sim);
void load_central_well_scene(sim::Simulation& sim);
void load_central_well_2_scene(sim::Simulation& sim);
void load_central_well_3_scene(sim::Simulation& sim);
void load_well_lattice_scene(sim::Simulation& sim);
void load_ring_resonator_scene(sim::Simulation& sim);
void load_barrier_gauntlet_scene(sim::Simulation& sim);

} // namespace ui::presets

#endif
