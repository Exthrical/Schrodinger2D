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
static constexpr float kDragThresholdPx = 4.0f;

struct AppState {
    sim::Simulation sim;
    sim::ViewMode view{sim::ViewMode::MagnitudePhase};
    bool showPotential{true};
    bool normalizeView{true};

    // Packet placement defaults
    double packetAmplitude{1.0};
    double packetSigma{0.05};
    double packetKx{12.0};
    double packetKy{0.0};

    // Box placement defaults
    double boxHeight{2400.0};

    // Interaction state
    enum class Mode { Drag, AddPacket, AddBox } mode{Mode::Drag};
    enum class DragAction { None, MoveBox, MovePacket, AdjustPacketMomentum, AddBox, AddPacket };
    DragAction dragAction{DragAction::None};
    int selectedBox{-1};
    int selectedPacket{-1};
    int activeDragPacket{-1};
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
    ImVec2 boxEditorPos{0,0};
    ImVec2 packetEditorPos{0,0};

    bool showStyleEditor{false};
    float toastTimer{0.0f};
    std::string toastMessage;

    bool windowDragActive{false};
    ImVec2 windowDragMouseStart{0.0f, 0.0f};
    int windowDragStartX{0};
    int windowDragStartY{0};

    // GL texture for field visualization
    GLuint tex{0};
    int texW{0}, texH{0};
};

static void load_default_scene(AppState& app);
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
    style->WindowBorderSize = 1.0f;
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
    const double Vscale = (maxVre > 1e-12 ? 0.2 * maxVre : 20.0);
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

static void load_default_scene(AppState& app) {
    app.sim.running = false;
    app.sim.pfield.boxes.clear();
    app.sim.packets.clear();
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;

    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.0, 0.52, 1.0, 2400.0});
    app.sim.pfield.build(app.sim.V);

    sim::Packet p1{0.25, 0.75, 0.05, 1.0, 10.0, -1.0};
    app.sim.packets.push_back(p1);
    sim::Packet p2{0.25, 0.25, 0.05, 1.0, 42.0, 4.0};
    app.sim.packets.push_back(p2);
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
    int nx = app.sim.Nx;
    int ny = app.sim.Ny;
    if (ImGui::InputInt("Nx", &nx) | ImGui::InputInt("Ny", &ny)) {
        nx = std::clamp(nx, 16, 1024);
        ny = std::clamp(ny, 16, 1024);
        if (nx != app.sim.Nx || ny != app.sim.Ny) {
            app.sim.resize(nx, ny);
            app.selectedBox = -1;
            app.selectedPacket = -1;
            app.boxEditorOpen = false;
            app.packetEditorOpen = false;
        }
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
    if (ImGui::CollapsingHeader("Placement Defaults", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }

    if (ImGui::CollapsingHeader("Potential Field", ImGuiTreeNodeFlags_DefaultOpen)) {
        double vmin = 0.0, vmax = 5.0;
        bool changed = ImGui::SliderScalar("CAP strength", ImGuiDataType_Double, &app.sim.pfield.cap_strength, &vmin, &vmax, "%.2f");
        vmin = 0.02; vmax = 0.25;
        changed |= ImGui::SliderScalar("CAP ratio", ImGuiDataType_Double, &app.sim.pfield.cap_ratio, &vmin, &vmax, "%.3f");
        if (changed) {
            app.sim.pfield.build(app.sim.V);
        }
        ImGui::Text("%d box(es)", static_cast<int>(app.sim.pfield.boxes.size()));
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
    }

    if (ImGui::CollapsingHeader("Simulation Content", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%d packet(s)", static_cast<int>(app.sim.packets.size()));
        if (ImGui::Button("Clear packets")) {
            app.sim.packets.clear();
            app.sim.reset();
            app.selectedPacket = -1;
            app.packetEditorOpen = false;
        }
    }

    if (ImGui::CollapsingHeader("Scene IO", ImGuiTreeNodeFlags_DefaultOpen)) {
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
        {AppState::Mode::Drag,      "DRAG",   "Move / Edit",    "Drag boxes, packets, and adjust momentum"},
        {AppState::Mode::AddPacket, "PACKET", "Insert",         "Click-drag in the field to place a packet"},
        {AppState::Mode::AddBox,    "BOX",    "Barrier",        "Click-drag to create a potential box"},
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
}

static void draw_view_content(AppState& app) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 target = fit_size_keep_aspect(ImVec2((float)app.sim.Nx, (float)app.sim.Ny), avail);
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
        app.pendingPacketClick = false;
        app.packetDragDirty = false;

        if (app.mode == AppState::Mode::Drag) {
            if (hoveredMomentumIdx >= 0) {
                app.selectedPacket = hoveredMomentumIdx;
                app.selectedBox = -1;
                app.boxEditorOpen = false;
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
                app.activeDragPacket = hoveredPacketIdx;
                app.dragAction = AppState::DragAction::MovePacket;
                const auto& pkt = app.sim.packets[hoveredPacketIdx];
                app.packetDragStartCx = pkt.cx;
                app.packetDragStartCy = pkt.cy;
                app.packetEditorOpen = false;
                app.pendingPacketClick = true;
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
                    app.dragAction = AppState::DragAction::MoveBox;
                } else {
                    app.selectedBox = -1;
                    app.boxEditorOpen = false;
                    app.selectedPacket = -1;
                    app.packetEditorOpen = false;
                }
            }
        } else if (app.mode == AppState::Mode::AddBox) {
            app.dragAction = AppState::DragAction::AddBox;
        } else if (app.mode == AppState::Mode::AddPacket) {
            app.dragAction = AppState::DragAction::AddPacket;
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
        default:
            break;
        }

        app.dragAction = AppState::DragAction::None;
        app.activeDragPacket = -1;
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

    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) app.sim.running = !app.sim.running;
        if (ImGui::IsKeyPressed(ImGuiKey_R)) app.sim.reset();
        if (ImGui::IsKeyPressed(ImGuiKey_Delete) && app.selectedBox >= 0) {
            app.sim.pfield.boxes.erase(app.sim.pfield.boxes.begin() + app.selectedBox);
            app.selectedBox = -1;
            app.boxEditorOpen = false;
            app.sim.pfield.build(app.sim.V);
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

static void draw_top_bar(AppState& app, GLFWwindow* window, float& out_height) {
    constexpr float kTopPadding = 4.0f;
    constexpr float kSidePadding = 10.0f;

    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(0.0f, kTopPadding), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 0.0f));

    out_height = 0.0f;
    if (!ImGui::BeginMainMenuBar())
        return;

    ImGuiStyle& style = ImGui::GetStyle();
    float frameHeight = ImGui::GetWindowSize().y;
    out_height = frameHeight + kTopPadding;

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
        if (ImGui::MenuItem("Reload Default Scene")) {
            load_default_scene(app);
            push_toast(app, "Loaded default scene", 2.5f);
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
    load_default_scene(app);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        float topBarHeight = 0.0f;
        draw_top_bar(app, window, topBarHeight);

        ImGuiIO& frame_io = ImGui::GetIO();
        if ((frame_io.KeyCtrl || frame_io.KeySuper) && ImGui::IsKeyPressed(ImGuiKey_S)) {
            take_screenshot(app);
        }

        float contentY = topBarHeight;
        if (contentY < 0.0f) contentY = 0.0f;
        ImVec2 contentPos(0.0f, contentY);
        ImVec2 contentSize(frame_io.DisplaySize.x, std::max(0.0f, frame_io.DisplaySize.y - contentY));
        ImGui::SetNextWindowPos(contentPos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(contentSize, ImGuiCond_Always);
        ImGuiWindowFlags root_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                                      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        ImGui::Begin("Schrodinger2D", nullptr, root_flags);
        const float left_w = 360.0f; // fixed settings panel width
        const float tools_w = 264.0f;

        ImGui::BeginChild("SettingsPanel", ImVec2(left_w, 0), true);
        draw_settings(app);
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginGroup();
        ImVec2 rightAvail = ImGui::GetContentRegionAvail();
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float viewWidth = std::max(0.0f, rightAvail.x - tools_w - spacing);
        ImGui::BeginChild("ViewPanel", ImVec2(viewWidth, 0), true);
        draw_view_content(app);
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("ToolsPanel", ImVec2(tools_w, 0), true);
        draw_tools_panel(app);
        ImGui::EndChild();
        ImGui::EndGroup();
        ImGui::End();

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
