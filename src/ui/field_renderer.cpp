#if BUILD_GUI

#include "field_renderer.hpp"

#include <algorithm>
#include <cmath>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <GL/gl.h>

namespace ui {

void ensure_texture(GLuint& tex, int& texW, int& texH, int w, int h) {
    if (tex == 0) {
        glGenTextures(1, &tex);
    }
    if (texW != w || texH != h) {
        texW = w;
        texH = h;
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void render_field_to_rgba(const sim::Simulation& sim,
                          std::vector<unsigned char>& outRGBA,
                          bool showPotential,
                          sim::ViewMode view,
                          bool normalizeView) {
    const int W = sim.Nx;
    const int H = sim.Ny;
    outRGBA.resize(static_cast<size_t>(W) * static_cast<size_t>(H) * 4);
    double maxmag = 1.0;
    if (normalizeView) {
        double m = 1e-12;
        for (const auto& z : sim.psi) {
            m = std::max(m, static_cast<double>(std::abs(z)));
        }
        maxmag = m;
    }
    double maxVre = 0.0;
    if (showPotential) {
        for (int j = 0; j < H; ++j) {
            for (int i = 0; i < W; ++i) {
                maxVre = std::max(maxVre, std::abs(static_cast<double>(std::real(sim.V[sim.idx(i, j)]))));
            }
        }
    }
    const double Vscale = (maxVre > 1e-12 ? 0.8 * maxVre : 20.0);
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            auto z = sim.psi[sim.idx(i, j)];
            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            if (view == sim::ViewMode::Real) {
                float v = static_cast<float>(0.5 + 0.5 * (std::real(z) / maxmag));
                r = g = b = v;
            } else if (view == sim::ViewMode::Imag) {
                float v = static_cast<float>(0.5 + 0.5 * (std::imag(z) / maxmag));
                r = g = b = v;
            } else if (view == sim::ViewMode::Magnitude) {
                float v = static_cast<float>(std::min(1.0, std::abs(z) / maxmag));
                r = g = b = v;
            } else {
                const double PI = 3.14159265358979323846;
                const double phase = std::atan2(std::imag(z), std::real(z));
                float h = static_cast<float>((phase + PI) / (2.0 * PI));
                float s = 1.0f;
                float v = 1.0f;
                if (view == sim::ViewMode::Phase) {
                    v = normalizeView ? 1.0f : static_cast<float>(std::min(1.0, std::abs(z) / maxmag));
                } else {
                    v = static_cast<float>(std::min(1.0, std::abs(z) / maxmag));
                }
                float c = v * s;
                float x = c * (1 - static_cast<float>(std::fabs(std::fmod(h * 6.0f, 2.0f) - 1)));
                float m = v - c;
                float rr = 0.0f;
                float gg = 0.0f;
                float bb = 0.0f;
                int hi = static_cast<int>(std::floor(h * 6.0f)) % 6;
                if (hi == 0)      { rr = c; gg = x; bb = 0; }
                else if (hi == 1) { rr = x; gg = c; bb = 0; }
                else if (hi == 2) { rr = 0; gg = c; bb = x; }
                else if (hi == 3) { rr = 0; gg = x; bb = c; }
                else if (hi == 4) { rr = x; gg = 0; bb = c; }
                else              { rr = c; gg = 0; bb = x; }
                r = rr + m;
                g = gg + m;
                b = bb + m;
            }

            if (showPotential) {
                auto V = sim.V[sim.idx(i, j)];
                float pv = static_cast<float>(std::clamp(std::real(V) / Vscale, -1.0, 1.0));
                if (pv > 0) {
                    r = std::min(1.0f, r + pv * 0.3f);
                } else if (pv < 0) {
                    b = std::min(1.0f, b + (-pv) * 0.3f);
                }
            }

            const size_t k = static_cast<size_t>((j * W + i) * 4);
            outRGBA[k + 0] = static_cast<unsigned char>(std::round(r * 255.0f));
            outRGBA[k + 1] = static_cast<unsigned char>(std::round(g * 255.0f));
            outRGBA[k + 2] = static_cast<unsigned char>(std::round(b * 255.0f));
            outRGBA[k + 3] = 255;
        }
    }
}

} // namespace ui

#endif
