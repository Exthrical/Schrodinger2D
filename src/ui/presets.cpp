#if BUILD_GUI

#include "presets.hpp"

#include <cmath>

namespace ui::presets {

static void clear_scene(sim::Simulation& sim) {
    sim.running = false;
    sim.pfield.boxes.clear();
    sim.pfield.wells.clear();
    sim.packets.clear();
}

void load_default_twowall_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.pfield.boxes.push_back(sim::Box{0.48, 0.0, 0.52, 1.0, 2400.0});
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.25, 0.75, 0.05, 1.0, 10.0, -1.0});
    sim.packets.push_back(sim::Packet{0.25, 0.25, 0.05, 1.0, 42.0, 4.0});
    sim.reset();
}

void load_default_doubleslit_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.pfield.boxes.push_back(sim::Box{0.48, 0.0, 0.52, 0.4, 2400.0});
    sim.pfield.boxes.push_back(sim::Box{0.48, 0.6, 0.52, 1.0, 2400.0});
    sim.pfield.boxes.push_back(sim::Box{0.48, 0.45, 0.52, 0.55, 2400.0});
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.25, 0.5, 0.05, 1.0, 24.0, 0.0});
    sim.reset();
}

void load_default_doubleslit2_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.dt = 0.00001;
    sim.pfield.boxes.push_back(sim::Box{0.48, 0.0, 0.52, 0.4, 100000.0});
    sim.pfield.boxes.push_back(sim::Box{0.48, 0.6, 0.52, 1.0, 100000.0});
    sim.pfield.boxes.push_back(sim::Box{0.48, 0.45, 0.52, 0.55, 100000.0});
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.25, 0.5, 0.05, 1.0, 192.0, 0.0});
    sim.reset();
}

void load_counterpropagating_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.28, 0.5, 0.045, 0.8, 22.0, 0.0});
    sim.packets.push_back(sim::Packet{0.72, 0.5, 0.045, 0.8, -22.0, 0.0});
    sim.packets.push_back(sim::Packet{0.5, 0.68, 0.035, 0.6, -6.0, -10.0});
    sim.reset();
}

void load_waveguide_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.pfield.boxes.push_back(sim::Box{0.0, 0.0, 1.0, 0.08, 2200.0});
    sim.pfield.boxes.push_back(sim::Box{0.0, 0.92, 1.0, 1.0, 2200.0});
    sim.pfield.boxes.push_back(sim::Box{0.36, 0.0, 0.44, 0.38, 2200.0});
    sim.pfield.boxes.push_back(sim::Box{0.56, 0.62, 0.64, 1.0, 2200.0});
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.12, 0.5, 0.05, 1.0, 28.0, 0.0});
    sim.reset();
}

void load_trap_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.pfield.boxes.push_back(sim::Box{0.1, 0.1, 0.9, 0.12, 3400.0});
    sim.pfield.boxes.push_back(sim::Box{0.1, 0.88, 0.9, 0.9, 3400.0});
    sim.pfield.boxes.push_back(sim::Box{0.1, 0.1, 0.12, 0.9, 3400.0});
    sim.pfield.boxes.push_back(sim::Box{0.88, 0.1, 0.9, 0.9, 3400.0});
    sim.pfield.boxes.push_back(sim::Box{0.43, 0.43, 0.57, 0.57, 2800.0});
    sim.pfield.wells.push_back(sim::RadialWell{0.5, 0.5, -320.0, 0.08, sim::RadialWell::Profile::SoftCoulomb});
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.3, 0.5, 0.04, 0.7, 12.0, 6.0});
    sim.packets.push_back(sim::Packet{0.7, 0.5, 0.04, 0.7, -12.0, -6.0});
    sim.packets.push_back(sim::Packet{0.5, 0.3, 0.035, 0.6, 0.0, 14.0});
    sim.reset();
}

void load_central_well_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.dt = 0.000025;
    sim::RadialWell well;
    well.cx = 0.5;
    well.cy = 0.5;
    well.strength = -260.0;
    well.radius = 0.075;
    well.profile = sim::RadialWell::Profile::Gaussian;
    sim.pfield.wells.push_back(well);
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.35, 0.5, 0.035, 0.85, 0.0, 14.0});
    sim.packets.push_back(sim::Packet{0.65, 0.5, 0.035, 0.85, 0.0, -14.0});
    sim.reset();
}

void load_central_well_2_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.dt = 0.000025;
    sim::RadialWell well;
    well.cx = 0.5;
    well.cy = 0.5;
    well.strength = -500.0;
    well.radius = 0.075;
    well.profile = sim::RadialWell::Profile::InverseSquare;
    sim.pfield.wells.push_back(well);
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.175, 0.5, 0.035, 0.85, 65.0, 25.0});
    sim.reset();
}

void load_central_well_3_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.dt = 0.000025;
    sim::RadialWell well;
    well.cx = 0.5;
    well.cy = 0.5;
    well.strength = -4000.0;
    well.radius = 0.18;
    well.profile = sim::RadialWell::Profile::HarmonicOscillator;
    sim.pfield.wells.push_back(well);
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.425, 0.5, 0.035, 0.85, 15.0, 0.0});
    sim.reset();
}

void load_well_lattice_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.dt = 0.00002;
    const int cols = 5;
    const int rows = 4;
    const double startX = 0.18;
    const double startY = 0.2;
    const double gapX = 0.14;
    const double gapY = 0.16;
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            sim::RadialWell w;
            w.cx = startX + i * gapX;
            w.cy = startY + j * gapY;
            w.radius = 0.05;
            const bool attractive = ((i + j) % 2) == 0;
            w.strength = attractive ? -320.0 : 320.0;
            w.profile = attractive ? sim::RadialWell::Profile::SoftCoulomb : sim::RadialWell::Profile::Gaussian;
            sim.pfield.wells.push_back(w);
        }
    }
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.08, 0.25, 0.03, 0.85, 60.0, 2.0});
    sim.packets.push_back(sim::Packet{0.08, 0.75, 0.03, 0.85, 55.0, -2.0});
    sim.reset();
}

void load_ring_resonator_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.dt = 0.00002;
    const int segments = 12;
    const double tau = 6.28318530717958647692;
    for (int i = 0; i < segments; ++i) {
        double angle = (tau / segments) * i;
        sim::RadialWell wall;
        wall.cx = 0.5 + 0.28 * std::cos(angle);
        wall.cy = 0.5 + 0.28 * std::sin(angle);
        wall.radius = 0.045;
        wall.strength = 900.0;
        wall.profile = sim::RadialWell::Profile::Gaussian;
        sim.pfield.wells.push_back(wall);
    }
    sim::RadialWell core;
    core.cx = 0.5;
    core.cy = 0.5;
    core.radius = 0.07;
    core.strength = -450.0;
    core.profile = sim::RadialWell::Profile::HarmonicOscillator;
    sim.pfield.wells.push_back(core);
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.35, 0.5, 0.035, 0.8, 0.0, 24.0});
    sim.packets.push_back(sim::Packet{0.65, 0.5, 0.035, 0.8, 0.0, -24.0});
    sim.packets.push_back(sim::Packet{0.5, 0.65, 0.03, 0.6, -18.0, 0.0});
    sim.reset();
}

void load_barrier_gauntlet_scene(sim::Simulation& sim) {
    clear_scene(sim);
    sim.dt = 0.00002;
    sim.pfield.boxes.push_back(sim::Box{0.12, 0.1, 0.88, 0.18, 3400.0});
    sim.pfield.boxes.push_back(sim::Box{0.12, 0.82, 0.88, 0.9, 3400.0});
    sim.pfield.boxes.push_back(sim::Box{0.12, 0.28, 0.32, 0.72, 3400.0});
    sim.pfield.boxes.push_back(sim::Box{0.68, 0.28, 0.88, 0.72, 3400.0});
    sim.pfield.boxes.push_back(sim::Box{0.44, 0.44, 0.56, 0.56, 4200.0});

    for (int i = 0; i < 3; ++i) {
        sim::RadialWell sink;
        sink.cx = 0.35 + 0.15 * i;
        sink.cy = (i % 2 == 0) ? 0.3 : 0.7;
        sink.radius = 0.06;
        sink.strength = -380.0;
        sink.profile = sim::RadialWell::Profile::InverseSquare;
        sim.pfield.wells.push_back(sink);
    }
    sim::RadialWell exit;
    exit.cx = 0.85;
    exit.cy = 0.5;
    exit.radius = 0.07;
    exit.strength = -520.0;
    exit.profile = sim::RadialWell::Profile::SoftCoulomb;
    sim.pfield.wells.push_back(exit);
    sim.pfield.build(sim.V);

    sim.packets.push_back(sim::Packet{0.18, 0.5, 0.035, 0.9, 48.0, 0.0});
    sim.packets.push_back(sim::Packet{0.22, 0.35, 0.025, 0.7, 60.0, 12.0});
    sim.reset();
}

} // namespace ui::presets

#endif
