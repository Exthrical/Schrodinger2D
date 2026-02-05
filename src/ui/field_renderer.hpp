#pragma once

#if BUILD_GUI

#include <vector>

#include "sim/simulation.hpp"

namespace ui {

using GLuint = unsigned int;

void ensure_texture(GLuint& tex, int& texW, int& texH, int w, int h);

void render_field_to_rgba(const sim::Simulation& sim,
                          std::vector<unsigned char>& outRGBA,
                          bool showPotential,
                          sim::ViewMode view,
                          bool normalizeView);

} // namespace ui

#endif
