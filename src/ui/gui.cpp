#if BUILD_GUI

#include "gui.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl2.h>

#include <GLFW/glfw3.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <GL/gl.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <system_error>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "sim/simulation.hpp"
#include "io/scene.hpp"

namespace {

static constexpr float kPacketHandleRadiusPx = 9.0f;
static constexpr float kMomentumHandleRadiusPx = 12.0f;
static constexpr float kMomentumUVScale = 0.004f;
static constexpr float kWellHandleRadiusPx = 10.0f;
static constexpr float kDragThresholdPx = 4.0f;

struct AppState {
    sim::Simulation sim;
    sim::ViewMode view{sim::ViewMode::MagnitudePhase};
    bool showPotential{true};
    bool normalizeView{true};
    bool lockAspect{true};
    bool initialGridApplied{false};
    double viewportAspect{1.0};
    float viewportAvailWidth{1.0f};
    float viewportAvailHeight{1.0f};
    enum class LastEdited { None, Nx, Ny } lastEdited{LastEdited::None};

    // Packet placement defaults
    double packetAmplitude{1.0};
    double packetSigma{0.05};
    double packetKx{12.0};
    double packetKy{0.0};

    // Box placement defaults
    double boxHeight{2400.0};

    // Radial well placement defaults
    double wellStrength{200.0};
    double wellRadius{0.08};
    sim::RadialWell::Profile wellProfile{sim::RadialWell::Profile::SoftCoulomb};

    // Interaction state
    enum class Mode { Drag, AddPacket, AddBox, AddWell } mode{Mode::Drag};
    enum class DragAction {
        None,
        MoveBox,
        MovePacket,
        AdjustPacketMomentum,
        MoveWell,
        AddBox,
        AddPacket,
        AddWell
    };
    DragAction dragAction{DragAction::None};
    int selectedBox{-1};
    int selectedPacket{-1};
    int selectedWell{-1};
    int activeDragPacket{-1};
    int activeDragWell{-1};
    bool pendingPacketClick{false};
    bool packetDragDirty{false};
    ImVec2 dragStart{0,0};
    ImVec2 dragEnd{0,0};
    double packetDragStartCx{0.0};
    double packetDragStartCy{0.0};
    double packetDragStartKx{0.0};
    double packetDragStartKy{0.0};

    bool boxEditorOpen{false};
    bool packetEditorOpen{false};
    bool wellEditorOpen{false};
    ImVec2 boxEditorPos{0,0};
    ImVec2 packetEditorPos{0,0};
    ImVec2 wellEditorPos{0,0};

    bool showStyleEditor{false};
    float toastTimer{0.0f};
    std::string toastMessage;

    bool windowDragActive{false};
    ImVec2 windowDragMouseStart{0.0f, 0.0f};
    int windowDragStartX{0};
    int windowDragStartY{0};

    struct EigenPanelState {
        int modes{3};
        int basis{0};
        int maxIter{0};
        double tol{1e-6};
        int selected{-1};
        std::string status;
        std::vector<sim::EigenState> states;
    } eigen;

    // GL texture for field visualization
    GLuint tex{0};
    int texW{0}, texH{0};
};

static void load_default_doubleslit_scene(AppState& app);
static void load_counterpropagating_scene(AppState& app);
static void load_waveguide_scene(AppState& app);
static void load_trap_scene(AppState& app);
static void load_well_lattice_scene(AppState& app);
static void load_ring_resonator_scene(AppState& app);
static void load_barrier_gauntlet_scene(AppState& app);
static void push_toast(AppState& app, const std::string& message, float duration_seconds);
static bool save_current_view_png(const AppState& app, const std::filesystem::path& path);
static std::filesystem::path default_screenshot_path();
static void take_screenshot(AppState& app);
static inline ImVec2 operator+(ImVec2 a, ImVec2 b) { return ImVec2(a.x + b.x, a.y + b.y); }
static inline ImVec2 operator-(ImVec2 a, ImVec2 b) { return ImVec2(a.x - b.x, a.y - b.y); }
static inline ImVec2 operator*(ImVec2 a, float s) { return ImVec2(a.x * s, a.y * s); }

// Custom theme preset helper, mirroring ImGui::StyleColors* API.
// Applies a dark dashboard-like theme to the current context or the provided style.
static void StyleColorsDashboard(ImGuiStyle* dst = nullptr) {
    ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();

    style->WindowRounding = 2.0f;
    style->FrameRounding = 2.0f;
    style->GrabRounding = 2.0f;
    style->WindowBorderSize = 0.0f;
    style->FrameBorderSize = 1.0f;
    style->ScrollbarSize = 12.0f;
    style->ItemSpacing = ImVec2(8, 6);
    style->ItemInnerSpacing = ImVec2(6, 4);
    style->FramePadding = ImVec2(10, 6);

    ImVec4 bg0 = ImVec4(0.06f, 0.06f, 0.07f, 1.0f);
    ImVec4 bg1 = ImVec4(0.09f, 0.09f, 0.10f, 1.0f);
    ImVec4 bg2 = ImVec4(0.13f, 0.13f, 0.15f, 1.0f);
    ImVec4 text = ImVec4(0.95f, 0.95f, 0.96f, 1.0f);
    ImVec4 textMuted = ImVec4(0.75f, 0.75f, 0.78f, 1.0f);
    ImVec4 border = ImVec4(0.22f, 0.22f, 0.25f, 1.0f);
    ImVec4 accent = ImVec4(0.95f, 0.25f, 0.20f, 1.0f);

    ImVec4* c = style->Colors;
    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = textMuted;
    c[ImGuiCol_WindowBg] = bg0;
    c[ImGuiCol_ChildBg] = bg1;
    c[ImGuiCol_PopupBg] = bg1;
    c[ImGuiCol_Border] = border;
    c[ImGuiCol_FrameBg] = bg2;
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.18f, 0.20f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.22f, 0.25f, 1.0f);
    c[ImGuiCol_TitleBg] = bg1;
    c[ImGuiCol_TitleBgActive] = bg2;
    c[ImGuiCol_MenuBarBg] = bg1;
    c[ImGuiCol_Button] = ImVec4(0.14f, 0.14f, 0.16f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.22f, 0.22f, 0.26f, 1.0f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.26f, 0.26f, 0.30f, 1.0f);
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = ImVec4(0.85f, 0.20f, 0.18f, 1.0f);
    c[ImGuiCol_Header] = ImVec4(0.14f, 0.14f, 0.16f, 1.0f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.22f, 0.22f, 0.26f, 1.0f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.26f, 0.30f, 1.0f);
    c[ImGuiCol_Separator] = border;
    c[ImGuiCol_Tab] = bg1;
    c[ImGuiCol_TabActive] = bg2;
    c[ImGuiCol_PlotLines] = accent;
    c[ImGuiCol_NavHighlight] = accent;
}

static ImVec4 Desaturate(const ImVec4& c, float amount) {
    // amount: 0 = no change, 1 = fully gray
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(c.x, c.y, c.z, h, s, v);
    s = s * (1.0f - amount);
    ImVec4 out;
    ImGui::ColorConvertHSVtoRGB(h, s, v, out.x, out.y, out.z);
    out.w = c.w;
    return out;
}

static ImVec4 Darken(const ImVec4& c, float amount) {
    // amount: 0 = no change, 1 = black
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(c.x, c.y, c.z, h, s, v);
    v = v * (1.0f - amount);
    ImVec4 out;
    ImGui::ColorConvertHSVtoRGB(h, s, v, out.x, out.y, out.z);
    out.w = c.w;
    return out;
}

static ImVec4 Lighten(const ImVec4& c, float amount) {
    // amount: 0 = no change, 1 = full bright white
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(c.x, c.y, c.z, h, s, v);
    v = v + (1.0f - v) * amount;
    ImVec4 out;
    ImGui::ColorConvertHSVtoRGB(h, s, v, out.x, out.y, out.z);
    out.w = c.w;
    return out;
}

static void ensure_texture(AppState& app, int w, int h) {
    if (app.tex == 0) {
        glGenTextures(1, &app.tex);
    }
    if (app.texW != w || app.texH != h) {
        app.texW = w;
        app.texH = h;
        glBindTexture(GL_TEXTURE_2D, app.tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

static ImU32 make_rgba(float r, float g, float b, float a = 1.0f) {
    r = std::clamp(r, 0.0f, 1.0f);
    g = std::clamp(g, 0.0f, 1.0f);
    b = std::clamp(b, 0.0f, 1.0f);
    a = std::clamp(a, 0.0f, 1.0f);
    return IM_COL32((int)(r * 255.0f), (int)(g * 255.0f), (int)(b * 255.0f), (int)(a * 255.0f));
}

static void render_field_to_rgba(const sim::Simulation& sim, std::vector<unsigned char>& outRGBA,
                                 bool showPotential, sim::ViewMode view, bool normalizeView) {
    const int W = sim.Nx, H = sim.Ny;
    outRGBA.resize((size_t)W * H * 4);
    double maxmag = 1.0;
    if (normalizeView) {
        double m = 1e-12;
        for (auto& z : sim.psi) {
            m = std::max(m, (double)std::abs(z));
        }
        maxmag = m;
    }
    double maxVre = 0.0;
    if (showPotential) {
        for (int j = 0; j < H; ++j) {
            for (int i = 0; i < W; ++i) {
                maxVre = std::max(maxVre, std::abs((double)std::real(sim.V[sim.idx(i, j)])));
            }
        }
    }
    const double Vscale = (maxVre > 1e-12 ? 0.8 * maxVre : 20.0);
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            auto z = sim.psi[sim.idx(i, j)];
            float r = 0, g = 0, b = 0;
            if (view == sim::ViewMode::Real) {
                float v = (float)(0.5 + 0.5 * (std::real(z) / maxmag));
                r = g = b = v;
            } else if (view == sim::ViewMode::Imag) {
                float v = (float)(0.5 + 0.5 * (std::imag(z) / maxmag));
                r = g = b = v;
            } else if (view == sim::ViewMode::Magnitude) {
                float v = (float)std::min(1.0, std::abs(z) / maxmag);
                r = g = b = v;
            } else if (view == sim::ViewMode::Phase) {
                const double PI = 3.14159265358979323846;
                double phase = std::atan2(std::imag(z), std::real(z));
                float h = (float)((phase + PI) / (2.0 * PI));
                float s = 1.0f;
                float v = normalizeView ? 1.0f : (float)std::min(1.0, std::abs(z) / maxmag);
                float c = v * s;
                float x = c * (1 - (float)std::fabs(std::fmod(h * 6.0f, 2.0f) - 1));
                float m = v - c;
                float rr = 0, gg = 0, bb = 0;
                int hi = (int)std::floor(h * 6.0f) % 6;
                if (hi == 0)      { rr = c; gg = x; bb = 0; }
                else if (hi == 1) { rr = x; gg = c; bb = 0; }
                else if (hi == 2) { rr = 0; gg = c; bb = x; }
                else if (hi == 3) { rr = 0; gg = x; bb = c; }
                else if (hi == 4) { rr = x; gg = 0; bb = c; }
                else              { rr = c; gg = 0; bb = x; }
                r = rr + m; g = gg + m; b = bb + m;
            } else {
                const double PI = 3.14159265358979323846;
                double mag = std::abs(z) / maxmag;
                double phase = std::atan2(std::imag(z), std::real(z));
                float h = (float)((phase + PI) / (2.0 * PI));
                float s = 1.0f;
                float v = (float)std::min(1.0, mag);
                float c = v * s;
                float x = c * (1 - (float)std::fabs(std::fmod(h * 6.0f, 2.0f) - 1));
                float m = v - c;
                float rr = 0, gg = 0, bb = 0;
                int hi = (int)std::floor(h * 6.0f) % 6;
                if (hi == 0)      { rr = c; gg = x; bb = 0; }
                else if (hi == 1) { rr = x; gg = c; bb = 0; }
                else if (hi == 2) { rr = 0; gg = c; bb = x; }
                else if (hi == 3) { rr = 0; gg = x; bb = c; }
                else if (hi == 4) { rr = x; gg = 0; bb = c; }
                else              { rr = c; gg = 0; bb = x; }
                r = rr + m; g = gg + m; b = bb + m;
            }

            float a = 1.0f;
            if (showPotential) {
                auto V = sim.V[sim.idx(i, j)];
                float pv = (float)std::clamp(std::real(V) / Vscale, -1.0, 1.0);
                if (pv > 0) { r = std::min(1.0f, r + pv * 0.3f); }
                else if (pv < 0) { b = std::min(1.0f, b + (-pv) * 0.3f); }
            }

            size_t k = (size_t)((j * W + i) * 4);
            outRGBA[k + 0] = (unsigned char)std::round(r * 255.0f);
            outRGBA[k + 1] = (unsigned char)std::round(g * 255.0f);
            outRGBA[k + 2] = (unsigned char)std::round(b * 255.0f);
            outRGBA[k + 3] = (unsigned char)std::round(a * 255.0f);
        }
    }
}

static void load_default_twowall_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.0, 0.52, 1.0, 2400.0});
    app.sim.pfield.build(app.sim.V);

    sim::Packet p1{0.25, 0.75, 0.05, 1.0, 10.0, -1.0};
    app.sim.packets.push_back(p1);
    sim::Packet p2{0.25, 0.25, 0.05, 1.0, 42.0, 4.0};
    app.sim.packets.push_back(p2);
    app.sim.reset();
}

static void load_default_doubleslit_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.0, 0.52, 0.4, 2400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.6, 0.52, 1.0, 2400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.45, 0.52, 0.55, 2400.0});
    app.sim.pfield.build(app.sim.V);

    sim::Packet p1{0.25, 0.5, 0.05, 1.0, 24.0, 0.0};
    app.sim.packets.push_back(p1);
    app.sim.reset();
}

static void load_default_doubleslit2_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.sim.dt = 0.000010;
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.0, 0.52, 0.4, 100000.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.6, 0.52, 1.0, 100000.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.45, 0.52, 0.55, 100000.0});
    app.sim.pfield.build(app.sim.V);

    sim::Packet p1{0.25, 0.5, 0.05, 1.0, 192.0, 0.0};
    app.sim.packets.push_back(p1);
    app.sim.reset();
}

static void load_counterpropagating_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    app.sim.pfield.build(app.sim.V);

    sim::Packet left{0.28, 0.5, 0.045, 0.8, 22.0, 0.0};
    sim::Packet right{0.72, 0.5, 0.045, 0.8, -22.0, 0.0};
    sim::Packet offset{0.5, 0.68, 0.035, 0.6, -6.0, -10.0};
    app.sim.packets.push_back(left);
    app.sim.packets.push_back(right);
    app.sim.packets.push_back(offset);
    app.sim.reset();
}

static void load_waveguide_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    // Build a narrow S-shaped waveguide using axis-aligned barriers.
    app.sim.pfield.boxes.push_back(sim::Box{0.0, 0.0, 1.0, 0.08, 2200.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.0, 0.92, 1.0, 1.0, 2200.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.36, 0.0, 0.44, 0.38, 2200.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.56, 0.62, 0.64, 1.0, 2200.0});
    app.sim.pfield.build(app.sim.V);

    sim::Packet guide{0.12, 0.5, 0.05, 1.0, 28.0, 0.0};
    app.sim.packets.push_back(guide);
    app.sim.reset();
}

static void load_trap_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    // Create a square trap with a central obstacle to seed orbiting dynamics.
    app.sim.pfield.boxes.push_back(sim::Box{0.1, 0.1, 0.9, 0.12, 3400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.1, 0.88, 0.9, 0.9, 3400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.1, 0.1, 0.12, 0.9, 3400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.88, 0.1, 0.9, 0.9, 3400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.43, 0.43, 0.57, 0.57, 2800.0});
    app.sim.pfield.wells.push_back(sim::RadialWell{0.5, 0.5, -320.0, 0.08, sim::RadialWell::Profile::SoftCoulomb});
    app.sim.pfield.build(app.sim.V);

    sim::Packet ring1{0.3, 0.5, 0.04, 0.7, 12.0, 6.0};
    sim::Packet ring2{0.7, 0.5, 0.04, 0.7, -12.0, -6.0};
    sim::Packet ring3{0.5, 0.3, 0.035, 0.6, 0.0, 14.0};
    app.sim.packets.push_back(ring1);
    app.sim.packets.push_back(ring2);
    app.sim.packets.push_back(ring3);
    app.sim.reset();
}

static void load_central_well_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.sim.dt = 0.000025;
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    sim::RadialWell well;
    well.cx = 0.5; well.cy = 0.5;
    well.strength = -260.0;
    well.radius = 0.075;
    well.profile = sim::RadialWell::Profile::Gaussian;
    app.sim.pfield.wells.push_back(well);
    app.sim.pfield.build(app.sim.V);

    sim::Packet orbit1{0.35, 0.5, 0.035, 0.85, 0.0, 14.0};
    sim::Packet orbit2{0.65, 0.5, 0.035, 0.85, 0.0, -14.0};
    app.sim.packets.push_back(orbit1);
    app.sim.packets.push_back(orbit2);
    app.sim.reset();
}

static void load_central_well_2_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.sim.dt = 0.000025;
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    sim::RadialWell well;
    well.cx = 0.5; well.cy = 0.5;
    well.strength = -500.0;
    well.radius = 0.075;
    well.profile = sim::RadialWell::Profile::InverseSquare;
    app.sim.pfield.wells.push_back(well);
    app.sim.pfield.build(app.sim.V);

    sim::Packet orbit1{0.175, 0.5, 0.035, 0.85, 65.0, 25.0};
    app.sim.packets.push_back(orbit1);
    app.sim.reset();
}

static void load_central_well_3_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.sim.dt = 0.000025;
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    sim::RadialWell well;
    well.cx = 0.5; well.cy = 0.5;
    well.strength = -4000.0;
    well.radius = 0.18;
    well.profile = sim::RadialWell::Profile::HarmonicOscillator;
    app.sim.pfield.wells.push_back(well);
    app.sim.pfield.build(app.sim.V);

    sim::Packet orbit1{0.425, 0.5, 0.035, 0.85, 15.0, 0.0};
    app.sim.packets.push_back(orbit1);
    app.sim.reset();
}

static void load_well_lattice_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.sim.dt = 0.00002;
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

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
            bool attractive = ((i + j) % 2) == 0;
            w.strength = attractive ? -320.0 : 320.0;
            w.profile = attractive ? sim::RadialWell::Profile::SoftCoulomb : sim::RadialWell::Profile::Gaussian;
            app.sim.pfield.wells.push_back(w);
        }
    }
    app.sim.pfield.build(app.sim.V);

    sim::Packet beamA{0.08, 0.25, 0.03, 0.85, 60.0, 2.0};
    sim::Packet beamB{0.08, 0.75, 0.03, 0.85, 55.0, -2.0};
    app.sim.packets.push_back(beamA);
    app.sim.packets.push_back(beamB);
    app.sim.reset();
}

static void load_ring_resonator_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.sim.dt = 0.00002;
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

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
        app.sim.pfield.wells.push_back(wall);
    }
    sim::RadialWell core;
    core.cx = 0.5;
    core.cy = 0.5;
    core.radius = 0.07;
    core.strength = -450.0;
    core.profile = sim::RadialWell::Profile::HarmonicOscillator;
    app.sim.pfield.wells.push_back(core);

    app.sim.pfield.build(app.sim.V);

    sim::Packet runner1{0.35, 0.5, 0.035, 0.8, 0.0, 24.0};
    sim::Packet runner2{0.65, 0.5, 0.035, 0.8, 0.0, -24.0};
    sim::Packet runner3{0.5, 0.65, 0.03, 0.6, -18.0, 0.0};
    app.sim.packets.push_back(runner1);
    app.sim.packets.push_back(runner2);
    app.sim.packets.push_back(runner3);
    app.sim.reset();
}

static void load_barrier_gauntlet_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.pfield.wells.clear();
    app.sim.packets.clear();
    app.sim.dt = 0.00002;
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    app.sim.pfield.boxes.push_back(sim::Box{0.12, 0.1, 0.88, 0.18, 3400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.12, 0.82, 0.88, 0.9, 3400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.12, 0.28, 0.32, 0.72, 3400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.68, 0.28, 0.88, 0.72, 3400.0});
    app.sim.pfield.boxes.push_back(sim::Box{0.44, 0.44, 0.56, 0.56, 4200.0});

    for (int i = 0; i < 3; ++i) {
        sim::RadialWell sink;
        sink.cx = 0.35 + 0.15 * i;
        sink.cy = (i % 2 == 0) ? 0.3 : 0.7;
        sink.radius = 0.06;
        sink.strength = -380.0;
        sink.profile = sim::RadialWell::Profile::InverseSquare;
        app.sim.pfield.wells.push_back(sink);
    }
    sim::RadialWell exit;
    exit.cx = 0.85;
    exit.cy = 0.5;
    exit.radius = 0.07;
    exit.strength = -520.0;
    exit.profile = sim::RadialWell::Profile::SoftCoulomb;
    app.sim.pfield.wells.push_back(exit);

    app.sim.pfield.build(app.sim.V);

    sim::Packet beam{0.18, 0.5, 0.035, 0.9, 48.0, 0.0};
    sim::Packet probe{0.22, 0.35, 0.025, 0.7, 60.0, 12.0};
    app.sim.packets.push_back(beam);
    app.sim.packets.push_back(probe);
    app.sim.reset();
}

static std::filesystem::path default_screenshot_path() {
#ifdef PROJECT_SOURCE_DIR
    std::filesystem::path base(PROJECT_SOURCE_DIR);
#else
    std::filesystem::path base = std::filesystem::current_path();
#endif
    std::filesystem::path dir = base / "screenshots";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream name;
    name << "screenshot_" << std::put_time(&tm, "%Y%m%d_%H%M%S");

    std::filesystem::path candidate = dir / (name.str() + ".png");
    int counter = 1;
    while (std::filesystem::exists(candidate)) {
        candidate = dir / (name.str() + "_" + std::to_string(counter++) + ".png");
    }
    return candidate;
}

static bool save_current_view_png(const AppState& app, const std::filesystem::path& path) {
    if (app.sim.Nx <= 0 || app.sim.Ny <= 0)
        return false;
    std::vector<unsigned char> rgba;
    render_field_to_rgba(app.sim, rgba, app.showPotential, app.view, app.normalizeView);
    int stride = app.sim.Nx * 4;
    return stbi_write_png(path.string().c_str(), app.sim.Nx, app.sim.Ny, 4, rgba.data(), stride) != 0;
}

static ImVec2 fit_size_keep_aspect(ImVec2 content, ImVec2 avail) {
    float scale = std::min(avail.x / content.x, avail.y / content.y);
    return ImVec2(content.x * scale, content.y * scale);
}

static ImVec2 screen_to_uv(ImVec2 p, ImVec2 img_tl, ImVec2 img_br) {
    return ImVec2((p.x - img_tl.x) / (img_br.x - img_tl.x), (p.y - img_tl.y) / (img_br.y - img_tl.y));
}

static ImVec2 uv_to_screen(ImVec2 uv, ImVec2 img_tl, ImVec2 img_br) {
    return ImVec2(img_tl.x + uv.x * (img_br.x - img_tl.x), img_tl.y + uv.y * (img_br.y - img_tl.y));
}

static void push_toast(AppState& app, const std::string& message, float duration_seconds) {
    app.toastMessage = message;
    app.toastTimer = duration_seconds;
}

static void take_screenshot(AppState& app) {
    std::filesystem::path target = default_screenshot_path();
    if (save_current_view_png(app, target)) {
        push_toast(app, std::string("Saved screenshot to ") + target.string(), 3.0f);
    } else {
        push_toast(app, "Failed to save screenshot", 3.0f);
    }
}

static void draw_object_editors(AppState& app);
static void draw_tools_panel(AppState& app);

static const char* tool_mode_name(AppState::Mode mode) {
    switch (mode) {
        case AppState::Mode::Drag: return "Drag";
        case AppState::Mode::AddPacket: return "Add Packet";
        case AppState::Mode::AddBox: return "Add Box";
        case AppState::Mode::AddWell: return "Add Well";
        default: return "Unknown";
    }
}

static void draw_settings(AppState& app) {
    // For the 4 main controls: Slightly larger padding and a visible border
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.0f);

    if (ImGui::Button(app.sim.running ? "Pause [Space]" : "Start [Space]")) {
        app.sim.running = !app.sim.running;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) app.sim.step();
    //Line
    if (ImGui::Button("Reset [R]")) app.sim.reset();
    ImGui::SameLine();
    if (ImGui::Button("Renormalize")) {
        double m = app.sim.mass();
        if (m > 1e-12) {
            double s = 1.0 / std::sqrt(m);
            for (auto& z : app.sim.psi) z *= s;
        }
    }

    ImGui::PopStyleVar(2);
    {
        double dt_min = 1e-5, dt_max = 5e-3;
        ImGui::SliderScalar("dt", ImGuiDataType_Double, &app.sim.dt, &dt_min, &dt_max, "%.6f", ImGuiSliderFlags_Logarithmic);
    }
    const int originalNx = app.sim.Nx;
    const int originalNy = app.sim.Ny;
    int nx = originalNx;
   
    const auto clampGrid = [](int v) { return std::clamp(v, 16, 1024); };

    bool nxValueChanged = ImGui::InputInt("Nx", &nx);
    bool nxActive = ImGui::IsItemActive();
    if (nxActive) app.lastEdited = AppState::LastEdited::Nx;
    nx = clampGrid(nx);

    int ny = originalNy;
    bool nyValueChanged = ImGui::InputInt("Ny", &ny);
    bool nyActive = ImGui::IsItemActive();
    if (nyActive) app.lastEdited = AppState::LastEdited::Ny;
    ny = clampGrid(ny);

    bool lockToggled = ImGui::Checkbox("Lock aspect", &app.lockAspect);

    double aspect = (app.viewportAspect > 1e-6)
                        ? app.viewportAspect
                        : ((app.sim.Ny > 0)
                               ? static_cast<double>(app.sim.Nx) / static_cast<double>(app.sim.Ny)
                               : 1.0);

    const double availW = static_cast<double>(std::max(app.viewportAvailWidth, 1.0f));
    const double availH = static_cast<double>(std::max(app.viewportAvailHeight, 1.0f));

    auto compute_best_ny = [&](int targetNx) {
        if (targetNx <= 0 || aspect <= 1e-9) return clampGrid(std::max(originalNy, 16));
        long double ideal = static_cast<long double>(targetNx) / aspect;
        int floorNy = clampGrid(static_cast<int>(std::floor(ideal)));
        int ceilNy = clampGrid(static_cast<int>(std::ceil(ideal)));
        if (floorNy == ceilNy) return floorNy;
        double sx = availW / std::max(targetNx, 1);
        double syFloor = availH / std::max(floorNy, 1);
        double syCeil = availH / std::max(ceilNy, 1);
        double diffFloor = std::fabs(sx - syFloor);
        double diffCeil = std::fabs(sx - syCeil);
        const double eps = 1e-9;
        if (diffFloor < diffCeil - eps) return floorNy;
        if (diffCeil < diffFloor - eps) return ceilNy;
        double limitFloor = std::min(sx, syFloor);
        double limitCeil = std::min(sx, syCeil);
        return (limitFloor >= limitCeil) ? floorNy : ceilNy;
    };

    auto compute_best_nx = [&](int targetNy) {
        if (targetNy <= 0 || aspect <= 1e-9) return clampGrid(std::max(originalNx, 16));
        long double ideal = static_cast<long double>(targetNy) * aspect;
        int floorNx = clampGrid(static_cast<int>(std::floor(ideal)));
        int ceilNx = clampGrid(static_cast<int>(std::ceil(ideal)));
        if (floorNx == ceilNx) return floorNx;
        double sy = availH / std::max(targetNy, 1);
        double sxFloor = availW / std::max(floorNx, 1);
        double sxCeil = availW / std::max(ceilNx, 1);
        double diffFloor = std::fabs(sxFloor - sy);
        double diffCeil = std::fabs(sxCeil - sy);
        const double eps = 1e-9;
        if (diffFloor < diffCeil - eps) return floorNx;
        if (diffCeil < diffFloor - eps) return ceilNx;
        double limitFloor = std::min(sxFloor, sy);
        double limitCeil = std::min(sxCeil, sy);
        return (limitFloor >= limitCeil) ? floorNx : ceilNx;
    };

    if (app.lockAspect && aspect > 1e-6) {
        if (lockToggled && app.lockAspect) {
            if (app.lastEdited == AppState::LastEdited::Ny || (nyValueChanged && !nxValueChanged)) {
                nx = compute_best_nx(ny);
            } else {
                ny = compute_best_ny(nx);
            }
        } else {
            if (nxValueChanged && !nyValueChanged) {
                ny = compute_best_ny(nx);
            } else if (nyValueChanged && !nxValueChanged) {
                nx = compute_best_nx(ny);
            } else if (nxValueChanged && nyValueChanged) {
                if (app.lastEdited == AppState::LastEdited::Ny) {
                    nx = compute_best_nx(ny);
                } else {
                    ny = compute_best_ny(nx);
                }
            }
        }
    }

    nx = clampGrid(nx);
    ny = clampGrid(ny);

    if (nx != app.sim.Nx || ny != app.sim.Ny) {
        app.sim.resize(nx, ny);
        app.selectedBox = -1;
        app.selectedPacket = -1;
        app.selectedWell = -1;
        app.boxEditorOpen = false;
        app.packetEditorOpen = false;
        app.wellEditorOpen = false;
    }
    double mass = app.sim.mass();
    double left = 0.0, right = 0.0;
    app.sim.mass_split(left, right);
    ImGui::Text("Mass: %.6f", mass);
    ImGui::Text("Left: %.6f  Right: %.6f", left, right);
    ImGui::Separator();

    ImGui::Text("View");
    ImGui::Checkbox("Normalize view", &app.normalizeView);
    ImGui::Checkbox("Potential overlay", &app.showPotential);
    int vm = static_cast<int>(app.view);
    const char* modes[] = {"Mag+Phase","Real","Imag","Magnitude","Phase"};
    if (ImGui::Combo("Mode", &vm, modes, IM_ARRAYSIZE(modes))) {
        app.view = static_cast<sim::ViewMode>(vm);
    }
    ImGui::Separator();

    ImGui::Text("Tools");
    ImGui::Text("Active: %s", tool_mode_name(app.mode));
    ImGui::TextDisabled("Use the toolbar on the right to change.");

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Placement Defaults")) {
        ImGui::TextUnformatted("Gaussian packet");
        double vmin, vmax;
        vmin = 0.1;  vmax = 5.0;  ImGui::SliderScalar("Amplitude", ImGuiDataType_Double, &app.packetAmplitude, &vmin, &vmax, "%.3f");
        vmin = 0.01; vmax = 0.2;  ImGui::SliderScalar("Sigma", ImGuiDataType_Double, &app.packetSigma, &vmin, &vmax, "%.3f");
        vmin = -80.0; vmax = 80.0; ImGui::SliderScalar("k_x", ImGuiDataType_Double, &app.packetKx, &vmin, &vmax, "%.1f");
        vmin = -80.0; vmax = 80.0; ImGui::SliderScalar("k_y", ImGuiDataType_Double, &app.packetKy, &vmin, &vmax, "%.1f");

        ImGui::Separator();
        ImGui::TextUnformatted("New box");
        vmin = -4000.0; vmax = 4000.0;
        ImGui::SliderScalar("Height", ImGuiDataType_Double, &app.boxHeight, &vmin, &vmax, "%.1f");

        ImGui::Separator();
        ImGui::TextUnformatted("New radial well");
        vmin = -4000.0; vmax = 4000.0;
        ImGui::SliderScalar("Strength", ImGuiDataType_Double, &app.wellStrength, &vmin, &vmax, "%.1f");
        vmin = 0.01; vmax = 0.5;
        ImGui::SliderScalar("Radius", ImGuiDataType_Double, &app.wellRadius, &vmin, &vmax, "%.3f");
        const char* profiles[] = {"Gaussian", "Soft Coulomb", "Inverse Square", "Harmonic Oscillator"};
        int profileIdx = static_cast<int>(app.wellProfile);
        if (ImGui::Combo("Profile", &profileIdx, profiles, IM_ARRAYSIZE(profiles))) {
            const int profileCount = static_cast<int>(IM_ARRAYSIZE(profiles));
            profileIdx = std::clamp(profileIdx, 0, profileCount - 1);
            app.wellProfile = static_cast<sim::RadialWell::Profile>(profileIdx);
        }
    }

    if (ImGui::CollapsingHeader("Potential Field")) {
        double vmin = 0.0, vmax = 5.0;
        bool changed = ImGui::SliderScalar("CAP strength", ImGuiDataType_Double, &app.sim.pfield.cap_strength, &vmin, &vmax, "%.2f");
        vmin = 0.02; vmax = 0.25;
        changed |= ImGui::SliderScalar("CAP ratio", ImGuiDataType_Double, &app.sim.pfield.cap_ratio, &vmin, &vmax, "%.3f");
        if (changed) {
            app.sim.pfield.build(app.sim.V);
        }
        ImGui::Text("%d box(es)", static_cast<int>(app.sim.pfield.boxes.size()));
        ImGui::Text("%d well(s)", static_cast<int>(app.sim.pfield.wells.size()));
        if (ImGui::Button("Clear boxes")) {
            app.sim.pfield.boxes.clear();
            app.sim.reset();
            app.selectedBox = -1;
            app.boxEditorOpen = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Rebuild V & Reset")) {
            app.sim.reset();
        }
        if (ImGui::Button("Clear wells")) {
            app.sim.pfield.wells.clear();
            app.sim.reset();
            app.selectedWell = -1;
            app.wellEditorOpen = false;
        }
    }

    if (ImGui::CollapsingHeader("Simulation Content")) {
        ImGui::Text("%d packet(s)", static_cast<int>(app.sim.packets.size()));
        if (ImGui::Button("Clear packets")) {
            app.sim.packets.clear();
            app.sim.reset();
            app.selectedPacket = -1;
            app.packetEditorOpen = false;
        }
    }

    if (ImGui::CollapsingHeader("Scene IO")) {
        static char path[256] = "examples/scene.json";
        ImGui::InputText("Path", path, sizeof(path));
        if (ImGui::Button("Save JSON")) {
            io::Scene scene;
            io::from_simulation(app.sim, scene);
            io::save_scene(path, scene);
        }
        ImGui::SameLine();
        if (ImGui::Button("Load JSON")) {
            io::Scene scene;
            if (io::load_scene(path, scene)) {
                io::to_simulation(scene, app.sim);
                app.selectedBox = -1;
                app.selectedPacket = -1;
                app.boxEditorOpen = false;
                app.packetEditorOpen = false;
            }
        }
    }

}

static void draw_tools_panel(AppState& app) {
    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::TextUnformatted("Tools");
    ImGui::Separator();
    ImGui::Spacing();

    const ImVec4 base = style.Colors[ImGuiCol_ChildBg];
    const ImVec4 hover = Lighten(base, 0.2f);
    const ImVec4 activeFill = style.Colors[ImGuiCol_Header];
    const ImVec4 border = style.Colors[ImGuiCol_Border];
    const ImVec4 borderActive = style.Colors[ImGuiCol_PlotLines];
    const ImVec4 text = style.Colors[ImGuiCol_Text];
    const ImVec4 subtext = style.Colors[ImGuiCol_TextDisabled];

    struct ToolEntry {
        AppState::Mode mode;
        const char* title;
        const char* subtitle;
        const char* tooltip;
    };

    static const ToolEntry tools[] = {
        {AppState::Mode::Drag,      "DRAG",   "Move / Edit",    "Drag boxes, packets, wells, and adjust momentum"},
        {AppState::Mode::AddPacket, "PACKET", "Insert",         "Click-drag in the field to place a packet"},
        {AppState::Mode::AddBox,    "BOX",    "Barrier",        "Click-drag to create a potential box"},
        {AppState::Mode::AddWell,   "WELL",   "Radial",         "Click to place a radial potential well"},
    };

    const int columns = 2;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(4.0f, 4.0f));
    if (ImGui::BeginTable("tool_grid", columns, ImGuiTableFlags_SizingStretchSame)) {
        for (int idx = 0; idx < IM_ARRAYSIZE(tools); ++idx) {
            if (idx % columns == 0)
                ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(idx % columns);
            ImGui::PushID(idx);

            ImVec2 tileSize(std::max(0.0f, ImGui::GetContentRegionAvail().x), 92.0f);
            ImVec2 tileMin = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("tool_tile", tileSize);
            bool hovered = ImGui::IsItemHovered();
            bool clicked = ImGui::IsItemClicked();
            bool active = (app.mode == tools[idx].mode);

            ImVec4 fill = active ? activeFill : (hovered ? hover : base);
            ImVec4 outline = active ? borderActive : border;
            float outlineThickness = active ? 2.0f : 1.0f;
            ImVec2 tileMax = tileMin + tileSize;

            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(tileMin, tileMax, ImGui::GetColorU32(fill), 6.0f);
            dl->AddRect(tileMin, tileMax, ImGui::GetColorU32(outline), 6.0f, 0, outlineThickness);

            ImGui::SetCursorScreenPos(tileMin + ImVec2(14.0f, 18.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, text);
            ImGui::TextUnformatted(tools[idx].title);
            ImGui::PopStyleColor();

            ImGui::SetCursorScreenPos(tileMin + ImVec2(14.0f, tileSize.y - 28.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, subtext);
            ImGui::TextUnformatted(tools[idx].subtitle);
            ImGui::PopStyleColor();

            ImGui::SetCursorScreenPos(ImVec2(tileMin.x, tileMax.y));

            if (clicked && app.mode != tools[idx].mode) {
                app.mode = tools[idx].mode;
            }
            if (hovered && tools[idx].tooltip)
                ImGui::SetTooltip("%s", tools[idx].tooltip);

            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar(2);

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Eigenstates"), ImGuiTreeNodeFlags_DefaultOpen) {
        ImGui::BeginDisabled();
        ImGui::TextWrapped("Solves lowest modes of H = -(1/2)∇² + Re(V)");
        ImGui::EndDisabled();
        const int maxBasisAllowed = std::max(1, 2*std::max(app.sim.Nx, app.sim.Ny));
        if (app.eigen.basis <= 0) app.eigen.basis = maxBasisAllowed;
        if (app.eigen.maxIter <= 0) app.eigen.maxIter = maxBasisAllowed;
        int modes = app.eigen.modes;
        if (ImGui::InputInt("Modes", &modes)) {
            app.eigen.modes = std::clamp(modes, 1, std::min(32, maxBasisAllowed));
        }
        int basis = app.eigen.basis;
        if (ImGui::InputInt("Krylov size", &basis)) {
            app.eigen.basis = std::clamp(basis, app.eigen.modes, maxBasisAllowed);
        }
        int maxIter = app.eigen.maxIter;
        if (ImGui::InputInt("Max iters", &maxIter)) {
            app.eigen.maxIter = std::clamp(maxIter, app.eigen.basis, maxBasisAllowed);
        }
        double tol = app.eigen.tol;
        if (ImGui::InputDouble("Tolerance", &tol, 0.0, 0.0, "%.2e")) {
            app.eigen.tol = std::max(1e-12, tol);
        }
        if (ImGui::Button("Solve eigenmodes")) {
            app.eigen.status = "Solving...";
            ImGui::SetItemDefaultFocus();
            app.eigen.states = app.sim.compute_eigenstates(app.eigen.modes, app.eigen.basis, app.eigen.maxIter, app.eigen.tol);
            app.eigen.selected = app.eigen.states.empty() ? -1 : 0;
            app.eigen.status = app.eigen.states.empty() ? "No modes found" : "Solved";
        }
        if (!app.eigen.status.empty()) {
            ImGui::TextDisabled("%s", app.eigen.status.c_str());
        }
        if (!app.eigen.states.empty()) {
            ImGui::Separator();
            ImGui::Text("Modes:");
            for (int i = 0; i < static_cast<int>(app.eigen.states.size()); ++i) {
                ImGui::PushID(i);
                bool selected = (app.eigen.selected == i);
                double e = app.eigen.states[i].energy;
                if (ImGui::Selectable("##eig", selected, ImGuiSelectableFlags_AllowDoubleClick)) {
                    app.eigen.selected = i;
                    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        app.sim.apply_eigenstate(app.eigen.states[i]);
                    }
                }
                ImGui::SameLine();
                ImGui::Text("E = %.6f", e);
                if (ImGui::Button("Load")) {
                    app.sim.apply_eigenstate(app.eigen.states[i]);
                    app.eigen.selected = i;
                }
                ImGui::PopID();
            }
        }
    }
}

static void draw_view_content(AppState& app) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 target = fit_size_keep_aspect(ImVec2((float)app.sim.Lx, (float)app.sim.Ly), avail);
    ImVec2 cur = ImGui::GetCursorScreenPos();

    std::vector<unsigned char> rgba;
    render_field_to_rgba(app.sim, rgba, app.showPotential, app.view, app.normalizeView);
    ensure_texture(app, app.sim.Nx, app.sim.Ny);
    glBindTexture(GL_TEXTURE_2D, app.tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.sim.Nx, app.sim.Ny, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    ImGui::Image((void*)(intptr_t)app.tex, target);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 tl = cur;
    ImVec2 br = cur + target;

    for (size_t bi = 0; bi < app.sim.pfield.boxes.size(); ++bi) {
        const auto& b = app.sim.pfield.boxes[bi];
        ImVec2 p0 = uv_to_screen(ImVec2((float)b.x0, (float)b.y0), tl, br);
        ImVec2 p1 = uv_to_screen(ImVec2((float)b.x1, (float)b.y1), tl, br);
        ImVec2 top_left(std::min(p0.x, p1.x), std::min(p0.y, p1.y));
        ImVec2 bottom_right(std::max(p0.x, p1.x), std::max(p0.y, p1.y));
        bool selected = (static_cast<int>(bi) == app.selectedBox);
        ImU32 col = selected ? make_rgba(1.0f, 0.9f, 0.1f, 0.95f)
                             : make_rgba(1.0f, 0.4f, 0.1f, 0.7f);
        float thickness = selected ? 3.0f : 1.5f;
        dl->AddRect(top_left, bottom_right, col, 0.0f, 0, thickness);
    }

    struct WellVisual {
        int idx;
        ImVec2 centerUV;
        ImVec2 centerScreen;
        float radiusPx;
        const sim::RadialWell* well;
    };
    std::vector<WellVisual> wellVis;
    wellVis.reserve(app.sim.pfield.wells.size());
    for (int wi = 0; wi < (int)app.sim.pfield.wells.size(); ++wi) {
        const auto& well = app.sim.pfield.wells[wi];
        WellVisual wv;
        wv.idx = wi;
        wv.well = &well;
        wv.centerUV = ImVec2((float)well.cx, (float)well.cy);
        wv.centerScreen = uv_to_screen(wv.centerUV, tl, br);
        ImVec2 sampleUV = ImVec2((float)(well.cx + well.radius), (float)well.cy);
        ImVec2 sampleScreen = uv_to_screen(sampleUV, tl, br);
        float dx = sampleScreen.x - wv.centerScreen.x;
        float dy = sampleScreen.y - wv.centerScreen.y;
        wv.radiusPx = std::sqrt(dx * dx + dy * dy);
        wellVis.push_back(wv);
    }

    struct PacketVisual {
        int idx;
        ImVec2 centerUV;
        ImVec2 centerScreen;
        ImVec2 tipUV;
        ImVec2 tipScreen;
    };
    std::vector<PacketVisual> packetVis;
    packetVis.reserve(app.sim.packets.size());
    for (int pi = 0; pi < (int)app.sim.packets.size(); ++pi) {
        const auto& pkt = app.sim.packets[pi];
        PacketVisual pv;
        pv.idx = pi;
        pv.centerUV = ImVec2((float)pkt.cx, (float)pkt.cy);
        pv.centerScreen = uv_to_screen(pv.centerUV, tl, br);
        ImVec2 momentumDeltaUV = ImVec2((float)pkt.kx, (float)pkt.ky) * kMomentumUVScale;
        pv.tipUV = pv.centerUV + momentumDeltaUV;
        pv.tipScreen = uv_to_screen(pv.tipUV, tl, br);
        packetVis.push_back(pv);
    }

    ImGuiIO& io = ImGui::GetIO();
    bool hovered = ImGui::IsItemHovered();
    int hoveredMomentumIdx = -1;
    int hoveredPacketIdx = -1;
    int hoveredWellIdx = -1;
    if (hovered) {
        float bestMomentum = kMomentumHandleRadiusPx * kMomentumHandleRadiusPx;
        float bestCenter = kPacketHandleRadiusPx * kPacketHandleRadiusPx;
        for (const auto& pv : packetVis) {
            float dx = pv.tipScreen.x - io.MousePos.x;
            float dy = pv.tipScreen.y - io.MousePos.y;
            float dist2 = dx * dx + dy * dy;
            if (dist2 <= bestMomentum) {
                bestMomentum = dist2;
                hoveredMomentumIdx = pv.idx;
            }
            dx = pv.centerScreen.x - io.MousePos.x;
            dy = pv.centerScreen.y - io.MousePos.y;
            dist2 = dx * dx + dy * dy;
            if (dist2 <= bestCenter) {
                bestCenter = dist2;
                hoveredPacketIdx = pv.idx;
            }
        }
        float bestWell = kWellHandleRadiusPx * kWellHandleRadiusPx;
        for (const auto& wv : wellVis) {
            float dx = wv.centerScreen.x - io.MousePos.x;
            float dy = wv.centerScreen.y - io.MousePos.y;
            float dist2 = dx * dx + dy * dy;
            if (dist2 <= bestWell) {
                bestWell = dist2;
                hoveredWellIdx = wv.idx;
            }
        }
    }

    for (const auto& wv : wellVis) {
        bool selected = (app.selectedWell == wv.idx);
        bool centerHover = (hoveredWellIdx == wv.idx && app.mode == AppState::Mode::Drag);
        const auto& well = *wv.well;
        bool attractive = well.strength < 0.0;
        ImU32 ringColor = attractive ? make_rgba(0.2f, 0.7f, 1.0f, selected ? 0.95f : 0.75f)
                                     : make_rgba(0.95f, 0.4f, 0.25f, selected ? 0.95f : 0.75f);
        ImU32 fillColor = attractive ? make_rgba(0.2f, 0.6f, 1.0f, selected ? 0.18f : 0.12f)
                                     : make_rgba(0.95f, 0.35f, 0.25f, selected ? 0.18f : 0.12f);
        float ringThickness = selected ? 3.0f : 2.0f;
        float circleRadius = std::max(4.0f, wv.radiusPx);
        dl->AddCircleFilled(wv.centerScreen, circleRadius, fillColor);
        dl->AddCircle(wv.centerScreen, circleRadius, ringColor, 0, ringThickness);

        float handleRadius = selected ? kWellHandleRadiusPx + 2.0f : kWellHandleRadiusPx;
        if (centerHover) handleRadius += 2.0f;
        ImU32 handleColor = attractive ? make_rgba(0.3f, 0.8f, 1.0f, 1.0f)
                                       : make_rgba(1.0f, 0.5f, 0.3f, 1.0f);
        dl->AddCircleFilled(wv.centerScreen, handleRadius * 0.45f, handleColor);
        dl->AddCircle(wv.centerScreen, handleRadius, handleColor, 0, 2.0f);
    }

    for (const auto& pv : packetVis) {
        bool selected = (app.selectedPacket == pv.idx);
        bool centerHover = (hoveredPacketIdx == pv.idx && app.mode == AppState::Mode::Drag);
        bool momentumHover = (hoveredMomentumIdx == pv.idx && app.mode == AppState::Mode::Drag);

        float radius = selected ? 8.0f : 6.0f;
        if (centerHover) radius += 1.5f;
        ImU32 outline = selected ? make_rgba(0.2f, 0.9f, 1.0f, 0.95f) : make_rgba(0.2f, 0.6f, 1.0f, 0.8f);
        float thickness = selected ? 3.0f : 2.0f;
        dl->AddCircle(pv.centerScreen, radius, outline, 0, thickness);
        dl->AddCircleFilled(pv.centerScreen, radius * 0.4f, make_rgba(0.2f, 0.6f, 1.0f, 0.8f));

        ImVec2 tip = pv.tipScreen;
        ImVec2 dir = tip - pv.centerScreen;
        float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        ImU32 arrowColor = selected ? make_rgba(0.95f, 0.35f, 0.25f, 0.95f) : make_rgba(0.25f, 0.75f, 1.0f, 0.9f);
        if (momentumHover) arrowColor = make_rgba(1.0f, 0.6f, 0.3f, 0.95f);
        float lineThickness = selected ? 2.6f : 1.9f;
        if (len > 1e-3f) {
            dl->AddLine(pv.centerScreen, tip, arrowColor, lineThickness);
            ImVec2 dirNorm = dir * (1.0f / len);
            ImVec2 ortho(-dirNorm.y, dirNorm.x);
            float headLen = 10.0f;
            float headWidth = 6.0f;
            ImVec2 head = tip;
            ImVec2 p2 = head - dirNorm * headLen + ortho * headWidth;
            ImVec2 p3 = head - dirNorm * headLen - ortho * headWidth;
            dl->AddTriangleFilled(head, p2, p3, arrowColor);
        }
        float handleRadius = momentumHover ? kMomentumHandleRadiusPx + 2.0f : kMomentumHandleRadiusPx;
        dl->AddCircleFilled(tip, handleRadius * 0.45f, arrowColor);
        dl->AddCircle(tip, handleRadius * 0.65f, arrowColor, 0, 1.5f);
    }

    ImVec2 mouseUV = screen_to_uv(io.MousePos, tl, br);

    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        app.dragStart = io.MousePos;
        app.dragEnd = io.MousePos;
        app.dragAction = AppState::DragAction::None;
        app.activeDragPacket = -1;
        app.activeDragWell = -1;
        app.pendingPacketClick = false;
        app.packetDragDirty = false;

        if (app.mode == AppState::Mode::Drag) {
            if (hoveredMomentumIdx >= 0) {
                app.selectedPacket = hoveredMomentumIdx;
                app.selectedBox = -1;
                app.boxEditorOpen = false;
                app.selectedWell = -1;
                app.wellEditorOpen = false;
                app.packetEditorOpen = false;
                app.activeDragPacket = hoveredMomentumIdx;
                app.dragAction = AppState::DragAction::AdjustPacketMomentum;
                const auto& pkt = app.sim.packets[hoveredMomentumIdx];
                app.packetDragStartKx = pkt.kx;
                app.packetDragStartKy = pkt.ky;
            } else if (hoveredPacketIdx >= 0) {
                app.selectedPacket = hoveredPacketIdx;
                app.selectedBox = -1;
                app.boxEditorOpen = false;
                app.selectedWell = -1;
                app.wellEditorOpen = false;
                app.activeDragPacket = hoveredPacketIdx;
                app.dragAction = AppState::DragAction::MovePacket;
                const auto& pkt = app.sim.packets[hoveredPacketIdx];
                app.packetDragStartCx = pkt.cx;
                app.packetDragStartCy = pkt.cy;
                app.packetEditorOpen = false;
                app.pendingPacketClick = true;
            } else if (hoveredWellIdx >= 0) {
                app.selectedWell = hoveredWellIdx;
                app.wellEditorOpen = true;
                app.wellEditorPos = io.MousePos + ImVec2(16, 16);
                app.selectedBox = -1;
                app.boxEditorOpen = false;
                app.selectedPacket = -1;
                app.packetEditorOpen = false;
                app.activeDragWell = hoveredWellIdx;
                app.dragAction = AppState::DragAction::MoveWell;
            } else {
                int boxHit = -1;
                for (int bi = static_cast<int>(app.sim.pfield.boxes.size()) - 1; bi >= 0; --bi) {
                    const auto& b = app.sim.pfield.boxes[bi];
                    double minx = std::min(b.x0, b.x1);
                    double maxx = std::max(b.x0, b.x1);
                    double miny = std::min(b.y0, b.y1);
                    double maxy = std::max(b.y0, b.y1);
                    if (mouseUV.x >= minx && mouseUV.x <= maxx && mouseUV.y >= miny && mouseUV.y <= maxy) {
                        boxHit = bi;
                        break;
                    }
                }
                if (boxHit >= 0) {
                    app.selectedBox = boxHit;
                    app.boxEditorOpen = true;
                    app.boxEditorPos = io.MousePos + ImVec2(16, 16);
                    app.selectedPacket = -1;
                    app.packetEditorOpen = false;
                    app.selectedWell = -1;
                    app.wellEditorOpen = false;
                    app.dragAction = AppState::DragAction::MoveBox;
                } else {
                    app.selectedBox = -1;
                    app.boxEditorOpen = false;
                    app.selectedPacket = -1;
                    app.packetEditorOpen = false;
                    app.selectedWell = -1;
                    app.wellEditorOpen = false;
                }
            }
        } else if (app.mode == AppState::Mode::AddBox) {
            app.dragAction = AppState::DragAction::AddBox;
        } else if (app.mode == AppState::Mode::AddPacket) {
            app.dragAction = AppState::DragAction::AddPacket;
        } else if (app.mode == AppState::Mode::AddWell) {
            app.dragAction = AppState::DragAction::AddWell;
        }
    }

    if (app.dragAction != AppState::DragAction::None && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        app.dragEnd = io.MousePos;

        switch (app.dragAction) {
        case AppState::DragAction::MoveBox: {
            if (app.selectedBox >= 0 && app.selectedBox < static_cast<int>(app.sim.pfield.boxes.size())) {
                ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br);
                ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);
                ImVec2 d = uv1 - uv0;
                auto& b = app.sim.pfield.boxes[app.selectedBox];
                b.x0 += d.x; b.x1 += d.x;
                b.y0 += d.y; b.y1 += d.y;
                app.dragStart = app.dragEnd;
                app.sim.pfield.build(app.sim.V);
            }
            break;
        }
        case AppState::DragAction::MovePacket: {
            if (app.activeDragPacket >= 0 && app.activeDragPacket < static_cast<int>(app.sim.packets.size())) {
                float dx = io.MousePos.x - app.dragStart.x;
                float dy = io.MousePos.y - app.dragStart.y;
                if (app.pendingPacketClick) {
                    if ((dx * dx + dy * dy) >= kDragThresholdPx * kDragThresholdPx) {
                        app.pendingPacketClick = false;
                    }
                }
                if (!app.pendingPacketClick) {
                    auto& pkt = app.sim.packets[app.activeDragPacket];
                    ImVec2 uv = screen_to_uv(io.MousePos, tl, br);
                    pkt.cx = std::clamp((double)uv.x, 0.0, 1.0);
                    pkt.cy = std::clamp((double)uv.y, 0.0, 1.0);
                    app.packetDragDirty = true;
                }
            }
            break;
        }
        case AppState::DragAction::AdjustPacketMomentum: {
            if (app.activeDragPacket >= 0 && app.activeDragPacket < static_cast<int>(app.sim.packets.size())) {
                auto& pkt = app.sim.packets[app.activeDragPacket];
                ImVec2 centerUV((float)pkt.cx, (float)pkt.cy);
                ImVec2 currentUV = screen_to_uv(io.MousePos, tl, br);
                ImVec2 deltaUV = currentUV - centerUV;
                pkt.kx = deltaUV.x / kMomentumUVScale;
                pkt.ky = deltaUV.y / kMomentumUVScale;
                app.packetDragDirty = true;
            }
            break;
        }
        case AppState::DragAction::MoveWell: {
            if (app.activeDragWell >= 0 && app.activeDragWell < static_cast<int>(app.sim.pfield.wells.size())) {
                auto& well = app.sim.pfield.wells[app.activeDragWell];
                ImVec2 uv = screen_to_uv(io.MousePos, tl, br);
                well.cx = std::clamp((double)uv.x, 0.0, 1.0);
                well.cy = std::clamp((double)uv.y, 0.0, 1.0);
                app.sim.pfield.build(app.sim.V);
            }
            break;
        }
        default:
            break;
        }
    }

    if (app.dragAction != AppState::DragAction::None && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br);
        ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);

        switch (app.dragAction) {
        case AppState::DragAction::AddBox: {
            sim::Box b;
            b.x0 = std::clamp((double)uv0.x, 0.0, 1.0);
            b.y0 = std::clamp((double)uv0.y, 0.0, 1.0);
            b.x1 = std::clamp((double)uv1.x, 0.0, 1.0);
            b.y1 = std::clamp((double)uv1.y, 0.0, 1.0);
            b.height = app.boxHeight;
            int newIndex = static_cast<int>(app.sim.pfield.boxes.size());
            app.sim.addBox(b);
            app.selectedBox = newIndex;
            app.boxEditorOpen = true;
            app.boxEditorPos = io.MousePos + ImVec2(16, 16);
            app.selectedPacket = -1;
            app.packetEditorOpen = false;
            break;
        }
        case AppState::DragAction::AddPacket: {
            ImVec2 uv_center = screen_to_uv(app.dragStart, tl, br);
            ImVec2 uv_release = screen_to_uv(app.dragEnd, tl, br);
            sim::Packet p;
            p.cx = std::clamp((double)uv_center.x, 0.0, 1.0);
            p.cy = std::clamp((double)uv_center.y, 0.0, 1.0);
            p.sigma = app.packetSigma;
            p.amplitude = app.packetAmplitude;
            ImVec2 d = uv_release - uv_center;
            if (std::fabs(d.x) + std::fabs(d.y) > 1e-6f) {
                float norm = std::sqrt(d.x * d.x + d.y * d.y);
                p.kx = (d.x / norm) * app.packetKx;
                p.ky = (d.y / norm) * app.packetKy;
            } else {
                p.kx = app.packetKx;
                p.ky = app.packetKy;
            }
            app.sim.packets.push_back(p);
            app.sim.injectGaussian(p);
            app.selectedPacket = static_cast<int>(app.sim.packets.size()) - 1;
            app.packetEditorOpen = true;
            app.packetEditorPos = io.MousePos + ImVec2(16, 16);
            app.selectedBox = -1;
            app.boxEditorOpen = false;
            break;
        }
        case AppState::DragAction::MovePacket:
            if (app.pendingPacketClick && app.activeDragPacket >= 0) {
                app.selectedPacket = app.activeDragPacket;
                app.packetEditorOpen = true;
                app.packetEditorPos = io.MousePos + ImVec2(16, 16);
            } else if (app.packetDragDirty) {
                app.sim.reset();
            }
            break;
        case AppState::DragAction::AdjustPacketMomentum:
            if (app.packetDragDirty) {
                app.sim.reset();
            }
            break;
        case AppState::DragAction::AddWell: {
            ImVec2 uv_center = screen_to_uv(app.dragEnd, tl, br);
            sim::RadialWell w;
            w.cx = std::clamp((double)uv_center.x, 0.0, 1.0);
            w.cy = std::clamp((double)uv_center.y, 0.0, 1.0);
            w.strength = app.wellStrength;
            w.radius = std::clamp(app.wellRadius, 0.01, 0.5);
            w.profile = app.wellProfile;
            app.sim.addWell(w);
            app.selectedWell = static_cast<int>(app.sim.pfield.wells.size()) - 1;
            app.wellEditorOpen = true;
            app.wellEditorPos = io.MousePos + ImVec2(16, 16);
            app.selectedBox = -1;
            app.boxEditorOpen = false;
            app.selectedPacket = -1;
            app.packetEditorOpen = false;
            break;
        }
        case AppState::DragAction::MoveWell:
            // keep selection; editor already open
            break;
        default:
            break;
        }

        app.dragAction = AppState::DragAction::None;
        app.activeDragPacket = -1;
        app.activeDragWell = -1;
        app.pendingPacketClick = false;
        app.packetDragDirty = false;
    }

    if ((app.dragAction == AppState::DragAction::AddBox || app.dragAction == AppState::DragAction::AddPacket) && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImU32 col = make_rgba(1, 1, 1, 0.7f);
        dl->AddRect(app.dragStart, app.dragEnd, col, 0.0f, 0, 2.0f);
    }

    if (hovered) {
        ImVec2 uv = screen_to_uv(ImGui::GetIO().MousePos, tl, br);
        int i = (int)std::round(uv.x * (app.sim.Nx - 1));
        int j = (int)std::round(uv.y * (app.sim.Ny - 1));
        i = std::clamp(i, 0, app.sim.Nx - 1);
        j = std::clamp(j, 0, app.sim.Ny - 1);
        auto z = app.sim.psi[app.sim.idx(i,j)];
        auto V = app.sim.V[app.sim.idx(i,j)];
        ImGui::SetTooltip("(i=%d,j=%d) psi=(%.3g,%.3g) |psi|=%.3g phase=%.3g rad V=(%.3g,%.3g)",
                          i,j, std::real(z), std::imag(z), std::abs(z), std::atan2(std::imag(z), std::real(z)),
                          std::real(V), std::imag(V));
    }

    if (!ImGui::GetIO().WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) app.sim.running = !app.sim.running;
        if (ImGui::IsKeyPressed(ImGuiKey_R)) app.sim.reset();
        if (ImGui::IsKeyPressed(ImGuiKey_Delete)) {
            if (app.selectedBox >= 0) {
                app.sim.pfield.boxes.erase(app.sim.pfield.boxes.begin() + app.selectedBox);
                app.selectedBox = -1;
                app.boxEditorOpen = false;
                app.sim.reset();
            } else if (app.selectedWell >= 0) {
                if (app.selectedWell < static_cast<int>(app.sim.pfield.wells.size())) {
                    app.sim.pfield.wells.erase(app.sim.pfield.wells.begin() + app.selectedWell);
                    app.selectedWell = -1;
                    app.wellEditorOpen = false;
                    app.sim.reset();
                }
            }
        }
    }

    draw_object_editors(app);
}

static void draw_object_editors(AppState& app) {
    const ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse;

    if (app.boxEditorOpen) {
        if (app.selectedBox < 0 || app.selectedBox >= static_cast<int>(app.sim.pfield.boxes.size())) {
            app.boxEditorOpen = false;
            app.selectedBox = -1;
        } else {
            ImGui::SetNextWindowPos(app.boxEditorPos, ImGuiCond_Appearing);
            bool open = app.boxEditorOpen;
            if (ImGui::Begin("Box Properties", &open, flags)) {
                auto& b = app.sim.pfield.boxes[app.selectedBox];
                app.boxEditorPos = ImGui::GetWindowPos();
                ImGui::Text("Box #%d", app.selectedBox);
                ImGui::Separator();
                bool rebuild = false;
                double rect[4] = {b.x0, b.y0, b.x1, b.y1};
                if (ImGui::DragScalarN("Bounds [x0,y0,x1,y1]", ImGuiDataType_Double, rect, 4, 0.002f, nullptr, nullptr, "%.3f")) {
                    b.x0 = std::clamp(rect[0], 0.0, 1.0);
                    b.y0 = std::clamp(rect[1], 0.0, 1.0);
                    b.x1 = std::clamp(rect[2], 0.0, 1.0);
                    b.y1 = std::clamp(rect[3], 0.0, 1.0);
                    rebuild = true;
                }
                double hmin = -4000.0, hmax = 4000.0;
                if (ImGui::SliderScalar("Height", ImGuiDataType_Double, &b.height, &hmin, &hmax, "%.2f")) {
                    rebuild = true;
                }
                if (rebuild) {
                    app.sim.pfield.build(app.sim.V);
                }
                if (ImGui::Button("Delete box")) {
                    app.sim.pfield.boxes.erase(app.sim.pfield.boxes.begin() + app.selectedBox);
                    app.selectedBox = -1;
                    app.boxEditorOpen = false;
                    app.sim.reset();
                    open = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Close##boxEditor")) {
                    open = false;
                }
            }
            ImGui::End();
            app.boxEditorOpen = open && app.selectedBox >= 0 && app.selectedBox < static_cast<int>(app.sim.pfield.boxes.size());
            if (!app.boxEditorOpen) {
                app.selectedBox = -1;
            }
        }
    }

    if (app.packetEditorOpen) {
        if (app.selectedPacket < 0 || app.selectedPacket >= static_cast<int>(app.sim.packets.size())) {
            app.packetEditorOpen = false;
            app.selectedPacket = -1;
        } else {
            ImGui::SetNextWindowPos(app.packetEditorPos, ImGuiCond_Appearing);
            bool open = app.packetEditorOpen;
            if (ImGui::Begin("Packet Properties", &open, flags)) {
                auto& p = app.sim.packets[app.selectedPacket];
                app.packetEditorPos = ImGui::GetWindowPos();
                ImGui::Text("Packet #%d", app.selectedPacket);
                ImGui::Separator();
                bool changed = false;
                double cxMin = 0.0, cxMax = 1.0;
                if (ImGui::SliderScalar("Center X", ImGuiDataType_Double, &p.cx, &cxMin, &cxMax, "%.3f")) changed = true;
                double cyMin = 0.0, cyMax = 1.0;
                if (ImGui::SliderScalar("Center Y", ImGuiDataType_Double, &p.cy, &cyMin, &cyMax, "%.3f")) changed = true;
                double sigmaMin = 0.01, sigmaMax = 0.3;
                if (ImGui::SliderScalar("Sigma", ImGuiDataType_Double, &p.sigma, &sigmaMin, &sigmaMax, "%.3f")) changed = true;
                double ampMin = 0.05, ampMax = 5.0;
                if (ImGui::SliderScalar("Amplitude", ImGuiDataType_Double, &p.amplitude, &ampMin, &ampMax, "%.3f")) changed = true;
                double kMin = -80.0, kMax = 80.0;
                if (ImGui::SliderScalar("k_x", ImGuiDataType_Double, &p.kx, &kMin, &kMax, "%.1f")) changed = true;
                if (ImGui::SliderScalar("k_y", ImGuiDataType_Double, &p.ky, &kMin, &kMax, "%.1f")) changed = true;
                if (changed) {
                    app.sim.reset();
                }
                if (ImGui::Button("Re-inject")) {
                    app.sim.reset();
                }
                ImGui::SameLine();
                if (ImGui::Button("Delete packet")) {
                    app.sim.packets.erase(app.sim.packets.begin() + app.selectedPacket);
                    app.selectedPacket = -1;
                    app.packetEditorOpen = false;
                    app.sim.reset();
                    open = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Close##packetEditor")) {
                    open = false;
                }
            }
            ImGui::End();
            app.packetEditorOpen = open && app.selectedPacket >= 0 && app.selectedPacket < static_cast<int>(app.sim.packets.size());
            if (!app.packetEditorOpen) {
                app.selectedPacket = -1;
            }
        }
    }

    if (app.wellEditorOpen) {
        if (app.selectedWell < 0 || app.selectedWell >= static_cast<int>(app.sim.pfield.wells.size())) {
            app.wellEditorOpen = false;
            app.selectedWell = -1;
        } else {
            ImGui::SetNextWindowPos(app.wellEditorPos, ImGuiCond_Appearing);
            bool open = app.wellEditorOpen;
            if (ImGui::Begin("Well Properties", &open, flags)) {
                auto& w = app.sim.pfield.wells[app.selectedWell];
                app.wellEditorPos = ImGui::GetWindowPos();
                ImGui::Text("Well #%d", app.selectedWell);
                ImGui::Separator();
                bool rebuild = false;
                const double cxMin = 0.0, cxMax = 1.0;
                if (ImGui::SliderScalar("Center X", ImGuiDataType_Double, &w.cx, &cxMin, &cxMax, "%.3f")) {
                    w.cx = std::clamp(w.cx, 0.0, 1.0);
                    rebuild = true;
                }
                const double cyMin = 0.0, cyMax = 1.0;
                if (ImGui::SliderScalar("Center Y", ImGuiDataType_Double, &w.cy, &cyMin, &cyMax, "%.3f")) {
                    w.cy = std::clamp(w.cy, 0.0, 1.0);
                    rebuild = true;
                }
                const double sMin = -4000.0, sMax = 4000.0;
                if (ImGui::SliderScalar("Strength", ImGuiDataType_Double, &w.strength, &sMin, &sMax, "%.1f")) {
                    rebuild = true;
                }
                const double rMin = 0.01, rMax = 0.5;
                if (ImGui::SliderScalar("Radius", ImGuiDataType_Double, &w.radius, &rMin, &rMax, "%.3f")) {
                    w.radius = std::clamp(w.radius, rMin, rMax);
                    rebuild = true;
                }
                const char* profileNames[] = {"Gaussian", "Soft Coulomb", "Inverse Square", "Harmonic Oscillator"};
                int profileIdx = static_cast<int>(w.profile);
                if (ImGui::Combo("Profile", &profileIdx, profileNames, IM_ARRAYSIZE(profileNames))) {
                    const int profileCount = static_cast<int>(IM_ARRAYSIZE(profileNames));
                    profileIdx = std::clamp(profileIdx, 0, profileCount - 1);
                    w.profile = static_cast<sim::RadialWell::Profile>(profileIdx);
                    rebuild = true;
                }
                if (rebuild) {
                    app.sim.pfield.build(app.sim.V);
                }
                if (ImGui::Button("Delete well")) {
                    app.sim.pfield.wells.erase(app.sim.pfield.wells.begin() + app.selectedWell);
                    app.selectedWell = -1;
                    app.wellEditorOpen = false;
                    app.sim.reset();
                    open = false;
                }
                ImGui::SameLine();
                if (ImGui::Button("Close##wellEditor")) {
                    open = false;
                }
            }
            ImGui::End();
            app.wellEditorOpen = open && app.selectedWell >= 0 && app.selectedWell < static_cast<int>(app.sim.pfield.wells.size());
            if (!app.wellEditorOpen) {
                app.selectedWell = -1;
            }
        }
    }
}

static void draw_style_editor(AppState& app) {
    if (!app.showStyleEditor)
        return;
    ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Style Editor", &app.showStyleEditor)) {
        ImGuiStyle& style = ImGui::GetStyle();
        ImGui::SliderFloat("Window Rounding", &style.WindowRounding, 0.0f, 20.0f, "%.1f");
        ImGui::SliderFloat("Frame Rounding", &style.FrameRounding, 0.0f, 20.0f, "%.1f");
        ImGui::SliderFloat("Grab Rounding", &style.GrabRounding, 0.0f, 20.0f, "%.1f");
        ImGui::SliderFloat("Window Border Size", &style.WindowBorderSize, 0.0f, 4.0f, "%.1f");
        ImGui::SliderFloat("Frame Border Size", &style.FrameBorderSize, 0.0f, 4.0f, "%.1f");
        ImGui::Checkbox("Anti-aliased lines", &style.AntiAliasedLines);
        ImGui::Checkbox("Anti-aliased fill", &style.AntiAliasedFill);
        ImGui::Separator();
        ImGui::ColorEdit3("Window BG", &style.Colors[ImGuiCol_WindowBg].x);
        ImGui::ColorEdit3("Header", &style.Colors[ImGuiCol_Header].x);
        ImGui::ColorEdit3("Button", &style.Colors[ImGuiCol_Button].x);
        ImGui::ColorEdit3("Accent", &style.Colors[ImGuiCol_PlotLines].x);
    }
    ImGui::End();
}

static void draw_toast_overlay(AppState& app) {
    if (app.toastTimer <= 0.0f || app.toastMessage.empty())
        return;
    ImGuiIO& io = ImGui::GetIO();
    app.toastTimer -= io.DeltaTime;
    if (app.toastTimer <= 0.0f) {
        app.toastTimer = 0.0f;
        app.toastMessage.clear();
        return;
    }
    ImGui::SetNextWindowBgAlpha(0.85f);
    ImGui::SetNextWindowPos(ImVec2(16.0f, io.DisplaySize.y - 80.0f), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav;
    if (ImGui::Begin("##toast_overlay", nullptr, flags)) {
        ImGui::TextUnformatted(app.toastMessage.c_str());
    }
    ImGui::End();
}

static void draw_top_bar(AppState& app, GLFWwindow* window, float& out_height, ImVec2& out_min, ImVec2& out_max) {
    constexpr float kTopPadding = 4.0f;
    constexpr float kSidePadding = 10.0f;

    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(0.0f, kTopPadding), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 0.0f));

    out_height = 0.0f;
    out_min = ImVec2(0.0f, 0.0f);
    out_max = ImVec2(0.0f, 0.0f);

    if (!ImGui::BeginMainMenuBar())
        return;

    ImGuiStyle& style = ImGui::GetStyle();
    float frameHeight = ImGui::GetWindowSize().y;
    out_height = frameHeight + kTopPadding;
    ImVec2 barPos = ImGui::GetWindowPos();
    ImVec2 barSize = ImGui::GetWindowSize();

    // Left padding to keep menus from hugging the edge
    ImGui::Dummy(ImVec2(kSidePadding, 0.0f));
    ImGui::SameLine(0.0f, 0.0f);

    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Screenshot", "Ctrl+S")) {
            take_screenshot(app);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Exit")) {
            glfwSetWindowShouldClose(window, 1);
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Examples")) {
        if (ImGui::MenuItem("Double Slit")) {
            load_default_doubleslit_scene(app);
            push_toast(app, "Loaded double slit scene", 2.5f);
        }
        if (ImGui::MenuItem("Double Slit (High Energy)")) {
            load_default_doubleslit2_scene(app);
            push_toast(app, "Loaded double slit scene", 2.5f);
        }
        if (ImGui::MenuItem("Two Packets")) {
            load_default_twowall_scene(app);
            push_toast(app, "Loaded two packet scene", 2.5f);
        }
        if (ImGui::MenuItem("Counter-propagating")) {
            load_counterpropagating_scene(app);
            push_toast(app, "Loaded counter-propagating packets", 2.5f);
        }
        if (ImGui::MenuItem("Waveguide Bend")) {
            load_waveguide_scene(app);
            push_toast(app, "Loaded waveguide scene", 2.5f);
        }
        if (ImGui::MenuItem("Trapped Swirl")) {
            load_trap_scene(app);
            push_toast(app, "Loaded trapped swirl", 2.5f);
        }
        if (ImGui::MenuItem("Central Well 1")) {
            load_central_well_scene(app);
            push_toast(app, "Loaded central radial well", 2.5f);
        }
        if (ImGui::MenuItem("Central Well 2")) {
            load_central_well_2_scene(app);
            push_toast(app, "Loaded central radial well (2)", 2.5f);
        }
        if (ImGui::MenuItem("Quantum Harmonic Oscillator")) {
            load_central_well_3_scene(app);
            push_toast(app, "Quantum Harmonic Oscillator", 2.5f);
        }
        if (ImGui::MenuItem("Well Lattice Fly-through")) {
            load_well_lattice_scene(app);
            push_toast(app, "Loaded lattice traversal", 2.5f);
        }
        if (ImGui::MenuItem("Ring Resonator")) {
            load_ring_resonator_scene(app);
            push_toast(app, "Loaded ring resonator", 2.5f);
        }
        if (ImGui::MenuItem("Barrier Gauntlet")) {
            load_barrier_gauntlet_scene(app);
            push_toast(app, "Loaded barrier gauntlet", 2.5f);
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Dark Theme")) {
            ImGui::StyleColorsDark();
        }
        if (ImGui::MenuItem("Light Theme")) {
            ImGui::StyleColorsLight();
        }
        if (ImGui::MenuItem("Classic Theme")) {
            ImGui::StyleColorsClassic();
        }
        // Custom preset directly callable like the built-in ones
        if (ImGui::MenuItem("Dashboard Theme")) {
            StyleColorsDashboard();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Style Editor...", nullptr, app.showStyleEditor)) {
            app.showStyleEditor = true;
        }
        ImGui::EndMenu();
    }

    bool maximized = glfwGetWindowAttrib(window, GLFW_MAXIMIZED);
    const char* maximizeLabel = maximized ? "Restore" : "Maximize";

    float spacing = style.ItemSpacing.x;
    float minimizeWidth = ImGui::CalcTextSize("-").x + style.FramePadding.x * 2.0f;
    float maximizeWidth = ImGui::CalcTextSize(maximizeLabel).x + style.FramePadding.x * 2.0f;
    float closeWidth = ImGui::CalcTextSize("X").x + style.FramePadding.x * 2.0f;

    float buttonHeight = ImGui::GetFrameHeight();
    float buttonY = (frameHeight - buttonHeight) * 0.5f;

    float totalButtons = minimizeWidth + maximizeWidth + closeWidth + spacing * 2.0f;
    float cursorX = ImGui::GetCursorPosX();
    float rawButtonStart = ImGui::GetWindowContentRegionMax().x - kSidePadding - totalButtons;
    float buttonStart = std::max(cursorX, rawButtonStart);
    float dragWidth = rawButtonStart - cursorX;

    if (dragWidth > 0.0f) {
        ImGui::SetCursorPosY(buttonY);
        ImGui::InvisibleButton("##drag_zone", ImVec2(dragWidth, buttonHeight));
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && !maximized) {
            app.windowDragActive = true;
            app.windowDragMouseStart = ImGui::GetIO().MousePos;
            glfwGetWindowPos(window, &app.windowDragStartX, &app.windowDragStartY);
        }
        ImGui::SameLine(0.0f, spacing);
    } else {
        ImGui::SameLine(buttonStart, spacing);
    }

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(style.FramePadding.x, style.FramePadding.y + 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2.0f);

    ImGui::SetCursorPosY(buttonY);
    if (ImGui::Button("-", ImVec2(minimizeWidth, buttonHeight))) {
        glfwIconifyWindow(window);
        app.windowDragActive = false;
    }
    ImGui::SameLine(0.0f, spacing);
    ImGui::SetCursorPosY(buttonY);
    if (ImGui::Button(maximizeLabel, ImVec2(maximizeWidth, buttonHeight))) {
        if (maximized) {
            glfwRestoreWindow(window);
        } else {
            glfwMaximizeWindow(window);
        }
        app.windowDragActive = false;
    }
    ImGui::SameLine(0.0f, spacing);
    //X button colors
    ImGuiStyle& s = ImGui::GetStyle();
    ImVec4 accent = s.Colors[ImGuiCol_PlotLines]; // primary/accent

    ImGui::PushStyleColor(ImGuiCol_Button, accent);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Desaturate(accent, 0.4f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, Darken(accent, 0.25f));
    ImGui::SetCursorPosY(buttonY);
    if (ImGui::Button("X", ImVec2(closeWidth, buttonHeight))) {
        glfwSetWindowShouldClose(window, 1);
        app.windowDragActive = false;
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(3);

    ImGui::EndMainMenuBar();

    out_min = barPos;
    out_max = barPos + barSize;

    if (app.windowDragActive) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImVec2 delta = io.MousePos - app.windowDragMouseStart;
            int newX = app.windowDragStartX + static_cast<int>(delta.x);
            int newY = app.windowDragStartY + static_cast<int>(delta.y);
            glfwSetWindowPos(window, newX, newY);
        } else {
            app.windowDragActive = false;
        }
    }
}

} // namespace

int run_gui(GLFWwindow* window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    // Aesthetics
    ImFontConfig cfg;
    cfg.OversampleH = 6;
    cfg.OversampleV = 6;
    cfg.PixelSnapH = false;

    io.Fonts->Clear();

    std::string fontPath = "third_party/imgui/misc/fonts/Cousine-Regular.ttf";
    #ifdef PROJECT_SOURCE_DIR
    fontPath = std::string(PROJECT_SOURCE_DIR) + "/third_party/imgui/misc/fonts/Cousine-Regular.ttf";
    #endif

    ImFont* roboto = io.Fonts->AddFontFromFileTTF(fontPath.c_str(), 16.0f, &cfg);
    if (!roboto) {
        // fall back if the file isn’t found (e.g. bad path)
        roboto = io.Fonts->AddFontDefault();
    }

    io.Fonts->Build();
    io.FontGlobalScale = 1.0f;

    StyleColorsDashboard();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL2_Init();

    AppState app;
    load_default_doubleslit2_scene(app);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        float topBarHeight = 0.0f;
        ImVec2 topBarMin, topBarMax;
        draw_top_bar(app, window, topBarHeight, topBarMin, topBarMax);

        ImGuiIO& frame_io = ImGui::GetIO();
        if ((frame_io.KeyCtrl || frame_io.KeySuper) && ImGui::IsKeyPressed(ImGuiKey_S)) {
            take_screenshot(app);
        }

        float contentY = std::max(0.0f, topBarHeight);
        const float left_w = 360.0f; // fixed settings panel width
        const float tools_w = 264.0f;
        float usableHeight = std::max(0.0f, frame_io.DisplaySize.y - contentY);
        float view_w = std::max(0.0f, frame_io.DisplaySize.x - left_w - tools_w);

        if (view_w > 1.0f && usableHeight > 1.0f) {
            app.viewportAvailWidth = view_w;
            app.viewportAvailHeight = usableHeight;
            // Overestimate width slightly for aspect, as aspect != 1.24 on 1080p in testing
            app.viewportAspect = (static_cast<double>(view_w) + static_cast<double>(5.0f)) / static_cast<double>(usableHeight);
            if (!app.initialGridApplied) {
                int baseNy = app.sim.Ny;
                long double idealNx = static_cast<long double>(baseNy) * app.viewportAspect;
                int floorNx = std::clamp(static_cast<int>(std::floor(idealNx)), 16, 1024);
                int ceilNx = std::clamp(static_cast<int>(std::ceil(idealNx)), 16, 1024);
                int targetNx = floorNx;
                if (floorNx != ceilNx) {
                    double sy = static_cast<double>(usableHeight) / std::max(baseNy, 1);
                    double sxFloor = static_cast<double>(view_w) / std::max(floorNx, 1);
                    double sxCeil = static_cast<double>(view_w) / std::max(ceilNx, 1);
                    double diffFloor = std::fabs(sxFloor - sy);
                    double diffCeil = std::fabs(sxCeil - sy);
                    const double eps = 1e-9;
                    if (diffFloor < diffCeil - eps)
                        targetNx = floorNx;
                    else if (diffCeil < diffFloor - eps)
                        targetNx = ceilNx;
                    else {
                        double limitFloor = std::min(sxFloor, sy);
                        double limitCeil = std::min(sxCeil, sy);
                        targetNx = (limitFloor >= limitCeil) ? floorNx : ceilNx;
                    }
                }
                if (targetNx != app.sim.Nx) {
                    app.sim.resize(targetNx, baseNy);
                    app.selectedBox = -1;
                    app.selectedPacket = -1;
                    app.selectedWell = -1;
                    app.boxEditorOpen = false;
                    app.packetEditorOpen = false;
                    app.wellEditorOpen = false;
                }
                app.initialGridApplied = true;
            }
        }

        ImGuiWindowFlags paneFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                     ImGuiWindowFlags_NoNavFocus;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

        const ImGuiWindowFlags settingsFlags = paneFlags | ImGuiWindowFlags_NoScrollbar;
        ImGui::SetNextWindowPos(ImVec2(0.0f, contentY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(left_w, usableHeight), ImGuiCond_Always);
        ImGui::Begin("##SettingsPane", nullptr, settingsFlags);
        draw_settings(app);
        ImVec2 settingsMin = ImGui::GetWindowPos();
        ImVec2 settingsMax = settingsMin + ImGui::GetWindowSize();
        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(frame_io.DisplaySize.x - tools_w, contentY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(tools_w, usableHeight), ImGuiCond_Always);
        ImGui::Begin("##ToolsPane", nullptr, paneFlags);
        draw_tools_panel(app);
        ImVec2 toolsMin = ImGui::GetWindowPos();
        ImVec2 toolsMax = toolsMin + ImGui::GetWindowSize();
        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(left_w, contentY), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(view_w, usableHeight), ImGuiCond_Always);
        ImGui::Begin("##ViewPane", nullptr, paneFlags);
        draw_view_content(app);
        ImGui::End();

        ImGui::PopStyleVar(2);

        ImVec2 topMin = topBarMin;
        ImVec2 topMax = topBarMax;
        ImVec2 totalMin(0.0f, topMin.y);
        ImVec2 totalMax(frame_io.DisplaySize.x - 1.0f, frame_io.DisplaySize.y - 1.0f);
        const float baseThickness = 1.0f;
        const float vertexThickness = 2.6f;
        const float vertexLen = 6.0f;
        const float junctionHalf = vertexLen * 0.5f;
        ImU32 borderCol = ImGui::GetColorU32(ImGui::GetStyle().Colors[ImGuiCol_Border]);
        auto snap = [](ImVec2 p) { return ImVec2(std::floor(p.x) + 0.5f, std::floor(p.y) + 0.5f); };
        ImDrawList* fg = ImGui::GetForegroundDrawList();
        auto thick_segment = [&](ImVec2 a, ImVec2 b) {
            fg->AddLine(snap(a), snap(b), borderCol, vertexThickness);
        };

        fg->AddRect(snap(totalMin), snap(totalMax), borderCol, 0.0f, 0, baseThickness);

        float separatorTop = topMax.y;
        float bottomY = totalMax.y;
        float leftSplitX = settingsMax.x;
        float rightSplitX = toolsMin.x;

        fg->AddLine(snap(ImVec2(leftSplitX, separatorTop)), snap(ImVec2(leftSplitX, bottomY)), borderCol, baseThickness);
        fg->AddLine(snap(ImVec2(rightSplitX, separatorTop)), snap(ImVec2(rightSplitX, bottomY)), borderCol, baseThickness);
        fg->AddLine(snap(ImVec2(totalMin.x, separatorTop)), snap(ImVec2(totalMax.x, separatorTop)), borderCol, baseThickness);

        ImVec2 topLeft(totalMin.x, topMin.y);
        ImVec2 topRight(totalMax.x, topMin.y);
        ImVec2 bottomLeft(totalMin.x, bottomY);
        ImVec2 bottomRight(totalMax.x, bottomY);

        thick_segment(topLeft, topLeft + ImVec2(vertexLen, 0.0f));
        thick_segment(topLeft, topLeft + ImVec2(0.0f, vertexLen));
        thick_segment(topRight, topRight - ImVec2(vertexLen, 0.0f));
        thick_segment(topRight, topRight + ImVec2(0.0f, vertexLen));
        thick_segment(bottomLeft, bottomLeft + ImVec2(vertexLen, 0.0f));
        thick_segment(bottomLeft, bottomLeft - ImVec2(0.0f, vertexLen));
        thick_segment(bottomRight, bottomRight - ImVec2(vertexLen, 0.0f));
        thick_segment(bottomRight, bottomRight - ImVec2(0.0f, vertexLen));

        ImVec2 leftSplitTop(leftSplitX, separatorTop);
        ImVec2 leftSplitBottom(leftSplitX, bottomY);
        ImVec2 rightSplitTop(rightSplitX, separatorTop);
        ImVec2 rightSplitBottom(rightSplitX, bottomY);

        thick_segment(leftSplitTop, leftSplitTop + ImVec2(0.0f, vertexLen));
        thick_segment(leftSplitBottom - ImVec2(0.0f, vertexLen), leftSplitBottom);
        thick_segment(rightSplitTop, rightSplitTop + ImVec2(0.0f, vertexLen));
        thick_segment(rightSplitBottom - ImVec2(0.0f, vertexLen), rightSplitBottom);

        thick_segment(leftSplitTop - ImVec2(junctionHalf, 0.0f), leftSplitTop + ImVec2(junctionHalf, 0.0f));
        thick_segment(rightSplitTop - ImVec2(junctionHalf, 0.0f), rightSplitTop + ImVec2(junctionHalf, 0.0f));
        thick_segment(leftSplitBottom - ImVec2(junctionHalf, 0.0f), leftSplitBottom + ImVec2(junctionHalf, 0.0f));
        thick_segment(rightSplitBottom - ImVec2(junctionHalf, 0.0f), rightSplitBottom + ImVec2(junctionHalf, 0.0f));

        draw_style_editor(app);
        draw_toast_overlay(app);

        if (app.sim.running) {
            app.sim.step();
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.05f, 0.05f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (app.tex) glDeleteTextures(1, &app.tex);
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    return 0;
}

#endif // BUILD_GUI
