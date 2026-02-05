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
#include <fstream>
#include <cstring>
#include <cstdio>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "sim/simulation.hpp"
#include "io/scene.hpp"
#include "ui/field_renderer.hpp"
#include "ui/presets.hpp"

namespace {

static constexpr float kPacketHandleRadiusPx = 9.0f;
static constexpr float kMomentumHandleRadiusPx = 12.0f;
static constexpr float kMomentumUVScale = 0.004f;
static constexpr float kWellHandleRadiusPx = 10.0f;
static constexpr float kDragThresholdPx = 4.0f;
static constexpr float kBoxEdgePickPx = 6.0f;


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
    int stepsPerFrame{1};
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
        MoveSelection,
        AdjustBoxEdge,
        // Legacy single-item drag actions (kept for save compatibility; unused)
        MovePacket,
        AdjustPacketMomentum,
        MoveWell,
        AddBox,
        AddPacket,
        AddWell
    };

    DragAction dragAction{DragAction::None};

    struct SelectedItem {
        enum class Kind { Box, Packet, Well } kind{Kind::Box};
        int idx{-1};
    };
    std::vector<SelectedItem> selection;

    int selectedBox{-1};
    int selectedPacket{-1};
    int selectedWell{-1};

    SelectedItem::Kind dragPrimaryKind{SelectedItem::Kind::Box};
    int dragPrimaryIdx{-1};

    int activeDragPacket{-1};
    int activeDragWell{-1};
    bool pendingPacketClick{false};
    bool packetDragDirty{false};
    bool selectionDragDirty{false};
    ImVec2 dragStart{0,0};
    ImVec2 dragEnd{0,0};
    ImVec2 mouseDownPos{0,0};

    double packetDragStartCx{0.0};
    double packetDragStartCy{0.0};
    double packetDragStartKx{0.0};
    double packetDragStartKy{0.0};

    enum class BoxEdge { None, Left, Right, Top, Bottom } dragBoxEdge{BoxEdge::None};


    bool boxEditorOpen{false};
    bool packetEditorOpen{false};
    bool wellEditorOpen{false};
    ImVec2 boxEditorPos{0,0};
    ImVec2 packetEditorPos{0,0};
    ImVec2 wellEditorPos{0,0};

    bool showStyleEditor{false};
    struct StyleTokens {
        float cornerRounding{2.0f};
        float borderWeight{1.0f};
        float density{1.0f};
        float fontScale{1.0f};
        float accentHue{0.01f};
        float accentSat{0.8f};
        float accentVal{0.95f};
        float panelLift{0.03f};
        bool antiAliased{true};
    } styleTokens;
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

    // Scene IO
    std::filesystem::path sceneLastSaveDir;
    std::filesystem::path sceneLastLoadDir;
    char saveScenePath[512]{};
    char loadScenePath[512]{};
    bool scenePathInit{false};

    // Deferred shift-multiselect (apply on mouse release if it was a click)
    bool pendingShiftToggle{false};
    SelectedItem::Kind pendingShiftToggleKind{SelectedItem::Kind::Box};
    int pendingShiftToggleIdx{-1};



    // GL texture for field visualization
    GLuint tex{0};
    int texW{0}, texH{0};
    std::vector<unsigned char> rgbaBuffer;
    bool fieldDirty{true};
    bool potentialDirtyDrag{false};
    bool lastUnstable{false};
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

static void help_marker(const char* text) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && text != nullptr) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

template <typename T>
static bool slider_block(const char* label,
                         const char* id,
                         ImGuiDataType type,
                         T* value,
                         const T* vmin,
                         const T* vmax,
                         const char* format,
                         ImGuiSliderFlags flags = 0,
                         const char* help = nullptr) {
    ImGui::TextUnformatted(label);
    if (help != nullptr) {
        ImGui::SameLine();
        help_marker(help);
    }
    ImGui::SetNextItemWidth(-1.0f);
    return ImGui::SliderScalar(id, type, value, vmin, vmax, format, flags);
}

static ImVec4 Darken(const ImVec4& c, float amount);
static ImVec4 Lighten(const ImVec4& c, float amount);

static void apply_style_tokens(AppState& app) {
    ImGuiStyle& style = ImGui::GetStyle();
    const auto& t = app.styleTokens;

    style.WindowRounding = t.cornerRounding;
    style.FrameRounding = t.cornerRounding;
    style.GrabRounding = t.cornerRounding;
    style.WindowBorderSize = t.borderWeight;
    style.FrameBorderSize = std::max(0.0f, t.borderWeight - 0.25f);
    style.ItemSpacing = ImVec2(8.0f * t.density, 6.0f * t.density);
    style.ItemInnerSpacing = ImVec2(6.0f * t.density, 4.0f * t.density);
    style.FramePadding = ImVec2(10.0f * t.density, 6.0f * t.density);
    style.AntiAliasedLines = t.antiAliased;
    style.AntiAliasedFill = t.antiAliased;
    ImGui::GetIO().FontGlobalScale = t.fontScale;

    float ar, ag, ab;
    ImGui::ColorConvertHSVtoRGB(t.accentHue, t.accentSat, t.accentVal, ar, ag, ab);
    ImVec4 accent(ar, ag, ab, 1.0f);

    const float base = 0.06f;
    const float lift = std::clamp(t.panelLift, 0.0f, 0.3f);
    ImVec4 bg0(base, base, base + 0.01f, 1.0f);
    ImVec4 bg1(base + lift, base + lift, base + lift + 0.01f, 1.0f);
    ImVec4 bg2(base + 2.0f * lift, base + 2.0f * lift, base + 2.0f * lift + 0.01f, 1.0f);

    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg] = bg0;
    c[ImGuiCol_ChildBg] = bg1;
    c[ImGuiCol_FrameBg] = bg2;
    c[ImGuiCol_FrameBgHovered] = Lighten(bg2, 0.15f);
    c[ImGuiCol_FrameBgActive] = Lighten(bg2, 0.25f);
    c[ImGuiCol_Button] = bg2;
    c[ImGuiCol_ButtonHovered] = Lighten(bg2, 0.18f);
    c[ImGuiCol_ButtonActive] = Lighten(bg2, 0.28f);
    c[ImGuiCol_Header] = bg2;
    c[ImGuiCol_HeaderHovered] = Lighten(bg2, 0.16f);
    c[ImGuiCol_HeaderActive] = Lighten(bg2, 0.26f);
    c[ImGuiCol_Border] = ImVec4(base + t.borderWeight * 0.08f, base + t.borderWeight * 0.08f, base + t.borderWeight * 0.1f, 1.0f);
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = accent;
    c[ImGuiCol_SliderGrabActive] = Darken(accent, 0.12f);
    c[ImGuiCol_PlotLines] = accent;
}

static bool selection_contains(const AppState& app, AppState::SelectedItem::Kind kind, int idx) {
    for (const auto& it : app.selection) {
        if (it.kind == kind && it.idx == idx) return true;
    }
    return false;
}

static void selection_clear(AppState& app) {
    app.selection.clear();
    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;
}

static void selection_set_single(AppState& app, AppState::SelectedItem::Kind kind, int idx) {
    selection_clear(app);
    app.selection.push_back({kind, idx});
    if (kind == AppState::SelectedItem::Kind::Box) {
        app.selectedBox = idx;
        app.boxEditorOpen = true;
    } else if (kind == AppState::SelectedItem::Kind::Packet) {
        app.selectedPacket = idx;
    } else if (kind == AppState::SelectedItem::Kind::Well) {
        app.selectedWell = idx;
        app.wellEditorOpen = true;
    }
}

static void selection_sync_single_editors(AppState& app, ImVec2 mousePos);

static void selection_toggle(AppState& app, AppState::SelectedItem::Kind kind, int idx) {
    for (size_t i = 0; i < app.selection.size(); ++i) {
        if (app.selection[i].kind == kind && app.selection[i].idx == idx) {
            app.selection.erase(app.selection.begin() + (int)i);
            if (app.selection.size() == 1) {
                selection_sync_single_editors(app, ImGui::GetIO().MousePos);
            } else {
                app.selectedBox = -1;
                app.selectedPacket = -1;
                app.selectedWell = -1;
                app.boxEditorOpen = false;
                app.packetEditorOpen = false;
                app.wellEditorOpen = false;
            }
            return;
        }
    }
    app.selection.push_back({kind, idx});
    if (app.selection.size() == 1) {
        selection_sync_single_editors(app, ImGui::GetIO().MousePos);
    } else {
        app.selectedBox = -1;
        app.selectedPacket = -1;
        app.selectedWell = -1;
        app.boxEditorOpen = false;
        app.packetEditorOpen = false;
        app.wellEditorOpen = false;
    }
}

static void selection_add(AppState& app, AppState::SelectedItem::Kind kind, int idx) {
    if (!selection_contains(app, kind, idx)) {
        app.selection.push_back({kind, idx});
    }
    if (app.selection.size() == 1) {
        selection_sync_single_editors(app, ImGui::GetIO().MousePos);
    } else {
        app.selectedBox = -1;
        app.selectedPacket = -1;
        app.selectedWell = -1;
        app.boxEditorOpen = false;
        app.packetEditorOpen = false;
        app.wellEditorOpen = false;
    }
}

static void selection_sync_single_editors(AppState& app, ImVec2 mousePos) {
    if (app.selection.size() != 1) return;

    app.selectedBox = -1;
    app.selectedPacket = -1;
    app.selectedWell = -1;
    app.boxEditorOpen = false;
    app.packetEditorOpen = false;
    app.wellEditorOpen = false;

    const auto it = app.selection[0];
    if (it.kind == AppState::SelectedItem::Kind::Box) {
        app.selectedBox = it.idx;
        app.boxEditorOpen = true;
        app.boxEditorPos = mousePos + ImVec2(16, 16);
    } else if (it.kind == AppState::SelectedItem::Kind::Packet) {
        app.selectedPacket = it.idx;
        app.packetEditorOpen = true;
        app.packetEditorPos = mousePos + ImVec2(16, 16);
    } else if (it.kind == AppState::SelectedItem::Kind::Well) {
        app.selectedWell = it.idx;
        app.wellEditorOpen = true;
        app.wellEditorPos = mousePos + ImVec2(16, 16);
    }
}

static void box_apply_edge_drag(sim::Box& b, AppState::BoxEdge& edge, float deltaUVx, float deltaUVy) {
    double x0 = b.x0, x1 = b.x1, y0 = b.y0, y1 = b.y1;
    if (edge == AppState::BoxEdge::Left) {
        x0 += deltaUVx;
        if (x0 > x1) {
            std::swap(x0, x1);
            edge = AppState::BoxEdge::Right;
        }
    } else if (edge == AppState::BoxEdge::Right) {
        x1 += deltaUVx;
        if (x0 > x1) {
            std::swap(x0, x1);
            edge = AppState::BoxEdge::Left;
        }
    } else if (edge == AppState::BoxEdge::Top) {
        y0 += deltaUVy;
        if (y0 > y1) {
            std::swap(y0, y1);
            edge = AppState::BoxEdge::Bottom;
        }
    } else if (edge == AppState::BoxEdge::Bottom) {
        y1 += deltaUVy;
        if (y0 > y1) {
            std::swap(y0, y1);
            edge = AppState::BoxEdge::Top;
        }
    }
    b.x0 = std::clamp(x0, 0.0, 1.0);
    b.x1 = std::clamp(x1, 0.0, 1.0);
    b.y0 = std::clamp(y0, 0.0, 1.0);
    b.y1 = std::clamp(y1, 0.0, 1.0);
}


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
    ui::ensure_texture(app.tex, app.texW, app.texH, w, h);
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
    ui::render_field_to_rgba(sim, outRGBA, showPotential, view, normalizeView);
}

static void load_default_twowall_scene(AppState& app) {
    ui::presets::load_default_twowall_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_default_doubleslit_scene(AppState& app) {
    ui::presets::load_default_doubleslit_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_default_doubleslit2_scene(AppState& app) {
    ui::presets::load_default_doubleslit2_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_counterpropagating_scene(AppState& app) {
    ui::presets::load_counterpropagating_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_waveguide_scene(AppState& app) {
    ui::presets::load_waveguide_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_trap_scene(AppState& app) {
    ui::presets::load_trap_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_central_well_scene(AppState& app) {
    ui::presets::load_central_well_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_central_well_2_scene(AppState& app) {
    ui::presets::load_central_well_2_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_central_well_3_scene(AppState& app) {
    ui::presets::load_central_well_3_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_well_lattice_scene(AppState& app) {
    ui::presets::load_well_lattice_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_ring_resonator_scene(AppState& app) {
    ui::presets::load_ring_resonator_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
}

static void load_barrier_gauntlet_scene(AppState& app) {
    ui::presets::load_barrier_gauntlet_scene(app.sim);
    selection_clear(app);
    app.fieldDirty = true;
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
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << "screenshot-" << std::put_time(&tm, "%Y-%m-%d-%H%M%S") << ".png";
    return dir / ss.str();
}

static std::filesystem::path exe_dir_path() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return std::filesystem::current_path();
    return std::filesystem::path(buf).parent_path();
#else
    return std::filesystem::current_path();
#endif
}

static bool ensure_dir_writable(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return false;
    auto probe = dir / ".write_test";
    std::ofstream f(probe.string(), std::ios::out | std::ios::trunc);
    if (!f) return false;
    f << "ok";
    f.close();
    std::filesystem::remove(probe, ec);
    return true;
}

static std::filesystem::path default_scene_dir() {
#ifdef _WIN32
    std::filesystem::path exeDir = exe_dir_path();
    std::filesystem::path candidate = exeDir / "scenes";
    if (ensure_dir_writable(candidate)) return candidate;

    wchar_t localAppData[MAX_PATH];
    DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", localAppData, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::filesystem::path p(localAppData);
        p /= "Schrodinger2D";
        p /= "scenes";
        ensure_dir_writable(p);
        return p;
    }

    std::filesystem::path fallback = std::filesystem::current_path() / "scenes";
    ensure_dir_writable(fallback);
    return fallback;
#else
    std::filesystem::path p = std::filesystem::current_path() / "scenes";
    ensure_dir_writable(p);
    return p;
#endif
}

#ifdef _WIN32
static std::wstring widen_utf8(const std::string& s) {
    if (s.empty()) return std::wstring();
    int wlen = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return std::wstring();
    std::wstring w;
    w.resize((size_t)wlen - 1);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), wlen);
    return w;
}

static std::string narrow_utf8(const std::wstring& w) {
    if (w.empty()) return std::string();
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string s;
    s.resize((size_t)len - 1);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), len, nullptr, nullptr);
    return s;
}

static std::string open_json_file_dialog(const std::filesystem::path& initialDir) {
    wchar_t fileBuf[MAX_PATH] = {0};
    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"JSON Files\0*.json\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    std::wstring init = initialDir.wstring();
    ofn.lpstrInitialDir = init.empty() ? nullptr : init.c_str();
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;
    if (GetOpenFileNameW(&ofn)) {
        return narrow_utf8(fileBuf);
    }
    return std::string();
}

static std::string save_json_file_dialog(const std::filesystem::path& initialDir, const std::string& defaultName) {
    wchar_t fileBuf[MAX_PATH] = {0};
    std::wstring def = widen_utf8(defaultName);
    if (!def.empty()) {
        wcsncpy_s(fileBuf, def.c_str(), _TRUNCATE);
    }

    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"JSON Files\0*.json\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    std::wstring init = initialDir.wstring();
    ofn.lpstrInitialDir = init.empty() ? nullptr : init.c_str();
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT | OFN_EXPLORER;
    ofn.lpstrDefExt = L"json";
    if (GetSaveFileNameW(&ofn)) {
        return narrow_utf8(fileBuf);
    }
    return std::string();
}
#endif

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
    if (ImGui::Button("Step")) {
        app.sim.step();
        app.fieldDirty = true;
    }
    //Line
    if (ImGui::Button("Reset [R]")) {
        app.sim.reset();
        app.fieldDirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Renormalize")) {
        double m = app.sim.mass();
        if (m > 1e-12) {
            double s = 1.0 / std::sqrt(m);
            for (auto& z : app.sim.psi) z *= s;
            app.sim.refresh_diagnostics_baseline();
            app.fieldDirty = true;
        }
    }

    ImGui::PopStyleVar(2);
    {
        double dt_min = 1e-5, dt_max = 5e-3;
        slider_block("dt", "##dt", ImGuiDataType_Double, &app.sim.dt, &dt_min, &dt_max, "%.6f", ImGuiSliderFlags_Logarithmic,
                     "Time step. Larger values run faster but reduce accuracy.");
        int spfMin = 1;
        int spfMax = 32;
        slider_block("Steps / frame", "##steps_per_frame", ImGuiDataType_S32, &app.stepsPerFrame, &spfMin, &spfMax, "%d", 0,
                     "How many simulation steps run each frame while playing.");
    }
    const int originalNx = app.sim.Nx;
    const int originalNy = app.sim.Ny;
    int nx = originalNx;
   
    const auto clampGrid = [](int v) { return std::clamp(v, 16, 1024); };

    bool nxValueChanged = ImGui::InputInt("Nx", &nx);
    ImGui::SameLine();
    help_marker("Grid width. Higher values improve detail and increase CPU cost.");
    bool nxActive = ImGui::IsItemActive();
    if (nxActive) app.lastEdited = AppState::LastEdited::Nx;
    nx = clampGrid(nx);

    int ny = originalNy;
    bool nyValueChanged = ImGui::InputInt("Ny", &ny);
    ImGui::SameLine();
    help_marker("Grid height. Keep aspect lock enabled for square cells.");
    bool nyActive = ImGui::IsItemActive();
    if (nyActive) app.lastEdited = AppState::LastEdited::Ny;
    ny = clampGrid(ny);

    bool lockToggled = ImGui::Checkbox("Lock aspect", &app.lockAspect);
    ImGui::SameLine();
    help_marker("Keep Nx/Ny aligned to the viewport aspect ratio.");

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
        selection_clear(app);
        app.fieldDirty = true;
    }

    const auto& diag = app.sim.diagnostics;
    ImGui::Text("Mass: %.6f", diag.current_mass);
    ImGui::Text("Left: %.6f  Right: %.6f", diag.left_mass, diag.right_mass);
    ImGui::Text("Interior mass: %.6f  Drift: %.3g", diag.current_interior_mass, diag.rel_interior_mass_drift);
    if (diag.unstable) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "Instability detected: %s", diag.reason.c_str());
    } else {
        ImGui::TextColored(ImVec4(0.35f, 0.9f, 0.5f, 1.0f), "Stability: OK");
    }
    ImGui::Separator();

    ImGui::Text("View");
    if (ImGui::Checkbox("Normalize view", &app.normalizeView)) app.fieldDirty = true;
    ImGui::SameLine();
    help_marker("Scale color mapping to current |psi| max.");
    if (ImGui::Checkbox("Potential overlay", &app.showPotential)) app.fieldDirty = true;
    ImGui::SameLine();
    help_marker("Overlay positive/negative potential tint.");
    int vm = static_cast<int>(app.view);
    const char* modes[] = {"Mag+Phase","Real","Imag","Magnitude","Phase"};
    if (ImGui::Combo("Mode", &vm, modes, IM_ARRAYSIZE(modes))) {
        app.view = static_cast<sim::ViewMode>(vm);
        app.fieldDirty = true;
    }
    ImGui::Separator();

    ImGui::Text("Tools");
    ImGui::Text("Active: %s", tool_mode_name(app.mode));
    ImGui::TextDisabled("Use the toolbar on the right to change.");

    ImGui::Separator();
    if (ImGui::CollapsingHeader("Placement Defaults")) {
        ImGui::TextUnformatted("Gaussian packet");
        double vmin, vmax;
        vmin = 0.1;  vmax = 5.0;  slider_block("Amplitude", "##packet_amplitude", ImGuiDataType_Double, &app.packetAmplitude, &vmin, &vmax, "%.3f");
        vmin = 0.01; vmax = 0.2;  slider_block("Sigma", "##packet_sigma", ImGuiDataType_Double, &app.packetSigma, &vmin, &vmax, "%.3f");
        vmin = -80.0; vmax = 80.0; slider_block("k_x", "##packet_kx", ImGuiDataType_Double, &app.packetKx, &vmin, &vmax, "%.1f");
        vmin = -80.0; vmax = 80.0; slider_block("k_y", "##packet_ky", ImGuiDataType_Double, &app.packetKy, &vmin, &vmax, "%.1f");

        ImGui::Separator();
        ImGui::TextUnformatted("New box");
        vmin = -4000.0; vmax = 4000.0;
        slider_block("Height", "##box_height", ImGuiDataType_Double, &app.boxHeight, &vmin, &vmax, "%.1f");

        ImGui::Separator();
        ImGui::TextUnformatted("New radial well");
        vmin = -4000.0; vmax = 4000.0;
        slider_block("Strength", "##well_strength", ImGuiDataType_Double, &app.wellStrength, &vmin, &vmax, "%.1f");
        vmin = 0.01; vmax = 0.5;
        slider_block("Radius", "##well_radius", ImGuiDataType_Double, &app.wellRadius, &vmin, &vmax, "%.3f");
        const char* profiles[] = {"Gaussian", "Soft Coulomb", "Inverse Square", "Harmonic Oscillator"};
        int profileIdx = static_cast<int>(app.wellProfile);
        if (ImGui::Combo("Profile", &profileIdx, profiles, IM_ARRAYSIZE(profiles))) {
            const int profileCount = static_cast<int>(IM_ARRAYSIZE(profiles));
            profileIdx = std::clamp(profileIdx, 0, profileCount - 1);
            app.wellProfile = static_cast<sim::RadialWell::Profile>(profileIdx);
        }
    }

    if (ImGui::CollapsingHeader("Potential Field", ImGuiTreeNodeFlags_DefaultOpen)) {
        double vmin = 0.0, vmax = 5.0;
        bool changed = slider_block("CAP strength", "##cap_strength", ImGuiDataType_Double, &app.sim.pfield.cap_strength, &vmin, &vmax, "%.2f",
                                   0, "Absorption gain near boundaries.");
        vmin = 0.02; vmax = 0.25;
        changed |= slider_block("CAP ratio", "##cap_ratio", ImGuiDataType_Double, &app.sim.pfield.cap_ratio, &vmin, &vmax, "%.3f",
                                0, "Fraction of each edge used as CAP sponge.");
        if (changed) {
            app.sim.pfield.build(app.sim.V);
            app.sim.refresh_diagnostics_baseline();
            app.fieldDirty = true;
        }
        ImGui::Text("%d box(es)", static_cast<int>(app.sim.pfield.boxes.size()));
        ImGui::Text("%d well(s)", static_cast<int>(app.sim.pfield.wells.size()));
        if (ImGui::Button("Clear boxes")) {
            app.sim.pfield.boxes.clear();
            app.sim.reset();
            selection_clear(app);
            app.fieldDirty = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Rebuild V & Reset")) {
            app.sim.reset();
            app.fieldDirty = true;
        }
        if (ImGui::Button("Clear wells")) {
            app.sim.pfield.wells.clear();
            app.sim.reset();
            selection_clear(app);
            app.fieldDirty = true;
        }

    }

    if (ImGui::CollapsingHeader("Stability Guard", ImGuiTreeNodeFlags_DefaultOpen)) {
        double driftMin = 1e-4;
        double driftMax = 0.25;
        slider_block("Mass drift tolerance", "##mass_drift_tol", ImGuiDataType_Double,
                     &app.sim.stability.rel_mass_drift_tol, &driftMin, &driftMax, "%.4f", ImGuiSliderFlags_Logarithmic,
                     "Relative total-mass drift allowed before warning.");
        slider_block("Interior drift tolerance", "##interior_drift_tol", ImGuiDataType_Double,
                     &app.sim.stability.rel_interior_mass_drift_tol, &driftMin, &driftMax, "%.4f", ImGuiSliderFlags_Logarithmic,
                     "Relative mass drift in the non-CAP interior.");
        int warmMin = 0;
        int warmMax = 100;
        slider_block("Warmup steps", "##stability_warmup", ImGuiDataType_S32,
                     &app.sim.stability.warmup_steps, &warmMin, &warmMax, "%d",
                     0, "Initial steps ignored by instability checks.");
        ImGui::Checkbox("Auto-pause on instability", &app.sim.stability.auto_pause_on_instability);
        ImGui::SameLine();
        help_marker("Pause playback when instability is detected.");
        if (ImGui::Button("Re-baseline diagnostics")) {
            app.sim.refresh_diagnostics_baseline();
            app.lastUnstable = false;
        }
    }

    if (ImGui::CollapsingHeader("Simulation Content", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%d packet(s)", static_cast<int>(app.sim.packets.size()));
        if (ImGui::Button("Clear packets")) {
            app.sim.packets.clear();
            app.sim.reset();
            selection_clear(app);
            app.fieldDirty = true;
        }

    }

    if (ImGui::CollapsingHeader("Scene IO")) {
        if (app.sceneLastSaveDir.empty()) app.sceneLastSaveDir = default_scene_dir();
        if (app.sceneLastLoadDir.empty()) app.sceneLastLoadDir = default_scene_dir();

        if (!app.scenePathInit) {
            const std::string saveDefault = (app.sceneLastSaveDir / "scene.json").string();
            const std::string loadDefault = (app.sceneLastLoadDir / "scene.json").string();
            std::snprintf(app.saveScenePath, sizeof(app.saveScenePath), "%s", saveDefault.c_str());
            std::snprintf(app.loadScenePath, sizeof(app.loadScenePath), "%s", loadDefault.c_str());
            app.scenePathInit = true;
        }

        ImGui::TextDisabled("Save folder: %s", app.sceneLastSaveDir.string().c_str());
        ImGui::TextDisabled("Load folder: %s", app.sceneLastLoadDir.string().c_str());

        ImGui::TextUnformatted("Save path");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputText("##save_scene_path", app.saveScenePath, IM_ARRAYSIZE(app.saveScenePath));
        ImGui::TextUnformatted("Load path");
        ImGui::SetNextItemWidth(-1.0f);
        ImGui::InputText("##load_scene_path", app.loadScenePath, IM_ARRAYSIZE(app.loadScenePath));

        if (ImGui::Button("Save")) {
            const std::filesystem::path p(app.saveScenePath);
            io::Scene scene;
            io::from_simulation(app.sim, scene);
            if (io::save_scene(p.string(), scene)) {
                app.sceneLastSaveDir = p.parent_path();
                push_toast(app, std::string("Saved scene to ") + p.string(), 2.5f);
            } else {
                push_toast(app, std::string("Failed to save scene to ") + p.string(), 3.0f);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            const std::filesystem::path p(app.loadScenePath);
            io::Scene scene;
            if (io::load_scene(p.string(), scene)) {
                io::to_simulation(scene, app.sim);
                app.sceneLastLoadDir = p.parent_path();
                selection_clear(app);
                app.fieldDirty = true;
                push_toast(app, std::string("Loaded scene from ") + p.string(), 2.5f);
            } else {
                push_toast(app, std::string("Failed to load scene: ") + p.string(), 3.0f);
            }
        }

        ImGui::Spacing();
        ImGui::TextDisabled("Optional native dialogs:");

        if (ImGui::Button("Save...")) {
#ifdef _WIN32
            std::string chosen = save_json_file_dialog(app.sceneLastSaveDir, "scene.json");
            if (!chosen.empty()) {
                std::filesystem::path p(chosen);
                std::snprintf(app.saveScenePath, sizeof(app.saveScenePath), "%s", p.string().c_str());
                app.sceneLastSaveDir = p.parent_path();
                io::Scene scene;
                io::from_simulation(app.sim, scene);
                if (io::save_scene(p.string(), scene)) {
                    push_toast(app, std::string("Saved scene to ") + p.string(), 2.5f);
                } else {
                    push_toast(app, std::string("Failed to save scene to ") + p.string(), 3.0f);
                }
            }
#else
            push_toast(app, "File dialog not available on this platform", 3.0f);
#endif
        }
        ImGui::SameLine();
        if (ImGui::Button("Load...")) {
#ifdef _WIN32
            std::string chosen = open_json_file_dialog(app.sceneLastLoadDir);
            if (!chosen.empty()) {
                std::filesystem::path p(chosen);
                std::snprintf(app.loadScenePath, sizeof(app.loadScenePath), "%s", p.string().c_str());
                app.sceneLastLoadDir = p.parent_path();
                io::Scene scene;
                if (io::load_scene(p.string(), scene)) {
                    io::to_simulation(scene, app.sim);
                    selection_clear(app);
                    app.fieldDirty = true;
                    push_toast(app, std::string("Loaded scene from ") + p.string(), 2.5f);
                } else {
                    push_toast(app, std::string("Failed to load scene: ") + p.string(), 3.0f);
                }
            }
#else
            push_toast(app, "Native file dialog unavailable; use path fields above", 3.0f);
#endif
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
    if (ImGui::CollapsingHeader("Eigenstates", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::BeginDisabled();
        ImGui::TextWrapped("Solves lowest modes of H = -(1/2)∇² + Re(V)");
        ImGui::EndDisabled();
        const int maxBasisAllowed = std::max(1, app.sim.Nx*app.sim.Ny); // full limit
        const int defaultBasis = std::max(1, 2*std::max(app.sim.Nx, app.sim.Ny)); // reasonable default
        if (app.eigen.basis <= 0) app.eigen.basis = defaultBasis;
        if (app.eigen.maxIter <= 0) app.eigen.maxIter = 1000;
        int modes = app.eigen.modes;
        if (ImGui::InputInt("Modes", &modes)) {
            app.eigen.modes = std::clamp(modes, 1, std::min(32, defaultBasis));
        }
        int basis = app.eigen.basis;
        if (ImGui::InputInt("Krylov size", &basis)) {
            app.eigen.basis = std::clamp(basis, app.eigen.modes, maxBasisAllowed); // full limit
        }
        int maxIter = app.eigen.maxIter;
        if (ImGui::InputInt("Max iters", &maxIter)) {
            app.eigen.maxIter = std::clamp(maxIter, app.eigen.basis, 4000*modes); // Arbitrary upper limit
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
                        app.fieldDirty = true;
                    }
                }
                ImGui::SameLine();
                ImGui::Text("E%d = %.6f", i, e);
                if (ImGui::Button("Load")) {
                    app.sim.apply_eigenstate(app.eigen.states[i]);
                    app.eigen.selected = i;
                    app.fieldDirty = true;
                    push_toast(app, "Eigenstate loaded", 2.5f);
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

    const bool texSizeChanged = (app.texW != app.sim.Nx || app.texH != app.sim.Ny);
    const bool needUpload = texSizeChanged || app.fieldDirty || app.rgbaBuffer.empty();
    if (needUpload) {
        render_field_to_rgba(app.sim, app.rgbaBuffer, app.showPotential, app.view, app.normalizeView);
        app.fieldDirty = false;
    }
    ensure_texture(app, app.sim.Nx, app.sim.Ny);
    if (needUpload) {
        glBindTexture(GL_TEXTURE_2D, app.tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, app.sim.Nx, app.sim.Ny, GL_RGBA, GL_UNSIGNED_BYTE, app.rgbaBuffer.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    ImGui::Image((void*)(intptr_t)app.tex, target);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 tl = cur;
    ImVec2 br = cur + target;

    ImGuiIO& io = ImGui::GetIO();
    bool hovered = ImGui::IsItemHovered();

    // Hover edge detection for visuals (independent of click handling)
    int hoveredBoxEdgeIdx = -1;
    AppState::BoxEdge hoveredBoxEdge = AppState::BoxEdge::None;
    float hoveredBoxEdgeBest = kBoxEdgePickPx;
    if (hovered && app.mode == AppState::Mode::Drag) {
        for (int bi = (int)app.sim.pfield.boxes.size() - 1; bi >= 0; --bi) {
            const auto& b = app.sim.pfield.boxes[bi];
            ImVec2 p0 = uv_to_screen(ImVec2((float)b.x0, (float)b.y0), tl, br);
            ImVec2 p1 = uv_to_screen(ImVec2((float)b.x1, (float)b.y1), tl, br);
            ImVec2 top_left(std::min(p0.x, p1.x), std::min(p0.y, p1.y));
            ImVec2 bottom_right(std::max(p0.x, p1.x), std::max(p0.y, p1.y));

            const bool inside = (io.MousePos.x >= top_left.x && io.MousePos.x <= bottom_right.x &&
                                 io.MousePos.y >= top_left.y && io.MousePos.y <= bottom_right.y);
            if (!inside) continue;

            float dxL = std::fabs(io.MousePos.x - top_left.x);
            float dxR = std::fabs(io.MousePos.x - bottom_right.x);
            float dyT = std::fabs(io.MousePos.y - top_left.y);
            float dyB = std::fabs(io.MousePos.y - bottom_right.y);

            float best = hoveredBoxEdgeBest;
            AppState::BoxEdge bestEdge = AppState::BoxEdge::None;
            if (dxL <= best) { best = dxL; bestEdge = AppState::BoxEdge::Left; }
            if (dxR <= best) { best = dxR; bestEdge = AppState::BoxEdge::Right; }
            if (dyT <= best) { best = dyT; bestEdge = AppState::BoxEdge::Top; }
            if (dyB <= best) { best = dyB; bestEdge = AppState::BoxEdge::Bottom; }

            if (bestEdge != AppState::BoxEdge::None) {
                hoveredBoxEdgeBest = best;
                hoveredBoxEdge = bestEdge;
                hoveredBoxEdgeIdx = bi;
                break;
            }
        }
    }

    for (size_t bi = 0; bi < app.sim.pfield.boxes.size(); ++bi) {
        const auto& b = app.sim.pfield.boxes[bi];
        ImVec2 p0 = uv_to_screen(ImVec2((float)b.x0, (float)b.y0), tl, br);
        ImVec2 p1 = uv_to_screen(ImVec2((float)b.x1, (float)b.y1), tl, br);
        ImVec2 top_left(std::min(p0.x, p1.x), std::min(p0.y, p1.y));
        ImVec2 bottom_right(std::max(p0.x, p1.x), std::max(p0.y, p1.y));

        bool selected = selection_contains(app, AppState::SelectedItem::Kind::Box, (int)bi);
        ImU32 col = selected ? make_rgba(1.0f, 0.9f, 0.1f, 0.95f)
                             : make_rgba(1.0f, 0.4f, 0.1f, 0.7f);
        float thickness = selected ? 3.0f : 1.5f;
        dl->AddRect(top_left, bottom_right, col, 0.0f, 0, thickness);

        if ((int)bi == hoveredBoxEdgeIdx && hoveredBoxEdge != AppState::BoxEdge::None) {
            ImU32 edgeCol = make_rgba(1.0f, 0.8f, 0.2f, 0.95f);
            float edgeThick = 3.5f;
            if (hoveredBoxEdge == AppState::BoxEdge::Left) {
                dl->AddLine(ImVec2(top_left.x, top_left.y), ImVec2(top_left.x, bottom_right.y), edgeCol, edgeThick);
            } else if (hoveredBoxEdge == AppState::BoxEdge::Right) {
                dl->AddLine(ImVec2(bottom_right.x, top_left.y), ImVec2(bottom_right.x, bottom_right.y), edgeCol, edgeThick);
            } else if (hoveredBoxEdge == AppState::BoxEdge::Top) {
                dl->AddLine(ImVec2(top_left.x, top_left.y), ImVec2(bottom_right.x, top_left.y), edgeCol, edgeThick);
            } else if (hoveredBoxEdge == AppState::BoxEdge::Bottom) {
                dl->AddLine(ImVec2(top_left.x, bottom_right.y), ImVec2(bottom_right.x, bottom_right.y), edgeCol, edgeThick);
            }
        }
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
            bool selected = selection_contains(app, AppState::SelectedItem::Kind::Well, wv.idx);

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
        bool selected = selection_contains(app, AppState::SelectedItem::Kind::Packet, pv.idx);

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
        app.mouseDownPos = io.MousePos;

        app.dragAction = AppState::DragAction::None;
        app.activeDragPacket = -1;
        app.activeDragWell = -1;
        app.pendingPacketClick = false;
        app.packetDragDirty = false;
        app.selectionDragDirty = false;
        app.dragBoxEdge = AppState::BoxEdge::None;
        app.dragPrimaryIdx = -1;
        app.pendingShiftToggle = false;
        app.pendingShiftToggleIdx = -1;
        app.potentialDirtyDrag = false;

        const bool shift = io.KeyShift;


        // Edge hit test (topmost box wins)
        int edgeBoxIdx = -1;
        AppState::BoxEdge edgeHit = AppState::BoxEdge::None;
        float edgeBest = kBoxEdgePickPx;
        for (int bi = (int)app.sim.pfield.boxes.size() - 1; bi >= 0; --bi) {
            const auto& b = app.sim.pfield.boxes[bi];
            ImVec2 p0 = uv_to_screen(ImVec2((float)b.x0, (float)b.y0), tl, br);
            ImVec2 p1 = uv_to_screen(ImVec2((float)b.x1, (float)b.y1), tl, br);
            ImVec2 top_left(std::min(p0.x, p1.x), std::min(p0.y, p1.y));
            ImVec2 bottom_right(std::max(p0.x, p1.x), std::max(p0.y, p1.y));

            const bool inside = (io.MousePos.x >= top_left.x && io.MousePos.x <= bottom_right.x &&
                                 io.MousePos.y >= top_left.y && io.MousePos.y <= bottom_right.y);
            if (!inside) continue;

            float dxL = std::fabs(io.MousePos.x - top_left.x);
            float dxR = std::fabs(io.MousePos.x - bottom_right.x);
            float dyT = std::fabs(io.MousePos.y - top_left.y);
            float dyB = std::fabs(io.MousePos.y - bottom_right.y);

            float best = edgeBest;
            AppState::BoxEdge bestEdge = AppState::BoxEdge::None;
            if (dxL <= best) { best = dxL; bestEdge = AppState::BoxEdge::Left; }
            if (dxR <= best) { best = dxR; bestEdge = AppState::BoxEdge::Right; }
            if (dyT <= best) { best = dyT; bestEdge = AppState::BoxEdge::Top; }
            if (dyB <= best) { best = dyB; bestEdge = AppState::BoxEdge::Bottom; }

            if (bestEdge != AppState::BoxEdge::None) {
                edgeBest = best;
                edgeHit = bestEdge;
                edgeBoxIdx = bi;
                break;
            }
        }

        if (edgeHit == AppState::BoxEdge::Left || edgeHit == AppState::BoxEdge::Right) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        } else if (edgeHit == AppState::BoxEdge::Top || edgeHit == AppState::BoxEdge::Bottom) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        if (app.mode == AppState::Mode::Drag) {
            if (hoveredMomentumIdx >= 0) {
                selection_set_single(app, AppState::SelectedItem::Kind::Packet, hoveredMomentumIdx);
                app.packetEditorOpen = false;
                app.activeDragPacket = hoveredMomentumIdx;
                app.dragAction = AppState::DragAction::AdjustPacketMomentum;
                const auto& pkt = app.sim.packets[hoveredMomentumIdx];
                app.packetDragStartKx = pkt.kx;
                app.packetDragStartKy = pkt.ky;
            } else if (edgeBoxIdx >= 0 && edgeHit != AppState::BoxEdge::None) {
                // fine select an edge: breaks multi-selection
                selection_set_single(app, AppState::SelectedItem::Kind::Box, edgeBoxIdx);
                app.boxEditorOpen = false;
                app.dragAction = AppState::DragAction::AdjustBoxEdge;
                app.dragPrimaryKind = AppState::SelectedItem::Kind::Box;
                app.dragPrimaryIdx = edgeBoxIdx;
                app.dragBoxEdge = edgeHit;
            } else if (hoveredPacketIdx >= 0) {
                if (shift) {
                    // Defer toggle until mouse release (unless this becomes a drag)
                    app.pendingShiftToggle = true;
                    app.pendingShiftToggleKind = AppState::SelectedItem::Kind::Packet;
                    app.pendingShiftToggleIdx = hoveredPacketIdx;

                    // Dragging only starts on an already-selected object
                    if (selection_contains(app, AppState::SelectedItem::Kind::Packet, hoveredPacketIdx)) {
                        app.dragAction = AppState::DragAction::MoveSelection;
                        app.dragPrimaryKind = AppState::SelectedItem::Kind::Packet;
                        app.dragPrimaryIdx = hoveredPacketIdx;
                    } else {
                        app.dragAction = AppState::DragAction::None;
                    }

                    app.packetEditorOpen = false;
                    app.pendingPacketClick = false;
                } else {
                    selection_set_single(app, AppState::SelectedItem::Kind::Packet, hoveredPacketIdx);

                    // packet click opens editor unless it turns into a drag
                    app.packetEditorOpen = false;
                    app.pendingPacketClick = true;

                    app.dragAction = AppState::DragAction::MoveSelection;
                    app.dragPrimaryKind = AppState::SelectedItem::Kind::Packet;
                    app.dragPrimaryIdx = hoveredPacketIdx;
                }

            } else if (hoveredWellIdx >= 0) {
                if (shift) {
                    app.pendingShiftToggle = true;
                    app.pendingShiftToggleKind = AppState::SelectedItem::Kind::Well;
                    app.pendingShiftToggleIdx = hoveredWellIdx;

                    if (selection_contains(app, AppState::SelectedItem::Kind::Well, hoveredWellIdx)) {
                        app.dragAction = AppState::DragAction::MoveSelection;
                        app.dragPrimaryKind = AppState::SelectedItem::Kind::Well;
                        app.dragPrimaryIdx = hoveredWellIdx;
                    } else {
                        app.dragAction = AppState::DragAction::None;
                    }

                    app.wellEditorOpen = false;
                } else {
                    selection_set_single(app, AppState::SelectedItem::Kind::Well, hoveredWellIdx);
                    selection_sync_single_editors(app, io.MousePos);
                    app.dragAction = AppState::DragAction::MoveSelection;
                    app.dragPrimaryKind = AppState::SelectedItem::Kind::Well;
                    app.dragPrimaryIdx = hoveredWellIdx;
                }

            } else {
                // box body hit (topmost wins)
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
                    if (shift) {
                        app.pendingShiftToggle = true;
                        app.pendingShiftToggleKind = AppState::SelectedItem::Kind::Box;
                        app.pendingShiftToggleIdx = boxHit;

                        if (selection_contains(app, AppState::SelectedItem::Kind::Box, boxHit)) {
                            app.dragAction = AppState::DragAction::MoveSelection;
                            app.dragPrimaryKind = AppState::SelectedItem::Kind::Box;
                            app.dragPrimaryIdx = boxHit;
                        } else {
                            app.dragAction = AppState::DragAction::None;
                        }

                        app.boxEditorOpen = false;
                    } else {
                        selection_set_single(app, AppState::SelectedItem::Kind::Box, boxHit);
                        selection_sync_single_editors(app, io.MousePos);
                        app.dragAction = AppState::DragAction::MoveSelection;
                        app.dragPrimaryKind = AppState::SelectedItem::Kind::Box;
                        app.dragPrimaryIdx = boxHit;
                    }
                } else {
                    if (!shift) selection_clear(app);
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

        // Cancel pending shift toggle if we started moving
        if (app.pendingShiftToggle) {
            ImVec2 dpx = app.dragEnd - app.mouseDownPos;
            float dist2 = dpx.x * dpx.x + dpx.y * dpx.y;
            if (dist2 > kDragThresholdPx * kDragThresholdPx) {
                app.pendingShiftToggle = false;
                app.pendingShiftToggleIdx = -1;
            }
        }

        if (app.dragAction == AppState::DragAction::MoveSelection) {
            ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br);
            ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);
            ImVec2 d = uv1 - uv0;

            bool rebuildPotential = false;
            bool packetMoved = false;

            for (const auto& it : app.selection) {
                if (it.kind == AppState::SelectedItem::Kind::Box) {
                    if (it.idx >= 0 && it.idx < (int)app.sim.pfield.boxes.size()) {
                        auto& b = app.sim.pfield.boxes[it.idx];
                        b.x0 += d.x; b.x1 += d.x;
                        b.y0 += d.y; b.y1 += d.y;
                        rebuildPotential = true;
                    }
                } else if (it.kind == AppState::SelectedItem::Kind::Packet) {
                    if (it.idx >= 0 && it.idx < (int)app.sim.packets.size()) {
                        auto& pkt = app.sim.packets[it.idx];
                        pkt.cx = std::clamp(pkt.cx + (double)d.x, 0.0, 1.0);
                        pkt.cy = std::clamp(pkt.cy + (double)d.y, 0.0, 1.0);
                        packetMoved = true;
                    }
                } else if (it.kind == AppState::SelectedItem::Kind::Well) {
                    if (it.idx >= 0 && it.idx < (int)app.sim.pfield.wells.size()) {
                        auto& w = app.sim.pfield.wells[it.idx];
                        w.cx = std::clamp(w.cx + (double)d.x, 0.0, 1.0);
                        w.cy = std::clamp(w.cy + (double)d.y, 0.0, 1.0);
                        rebuildPotential = true;
                    }
                }
            }

            app.dragStart = app.dragEnd;
            if (rebuildPotential) app.potentialDirtyDrag = true;
            if (packetMoved) app.selectionDragDirty = true;
            if (rebuildPotential || packetMoved) app.fieldDirty = true;
        } else if (app.dragAction == AppState::DragAction::AdjustBoxEdge) {
            if (app.dragPrimaryIdx >= 0 && app.dragPrimaryIdx < (int)app.sim.pfield.boxes.size()) {
                ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br);
                ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);
                ImVec2 d = uv1 - uv0;

                auto& b = app.sim.pfield.boxes[app.dragPrimaryIdx];
                box_apply_edge_drag(b, app.dragBoxEdge, d.x, d.y);
                app.dragStart = app.dragEnd;
                app.potentialDirtyDrag = true;
                app.fieldDirty = true;
            }
        } else if (app.dragAction == AppState::DragAction::AdjustPacketMomentum) {
            if (app.activeDragPacket >= 0 && app.activeDragPacket < static_cast<int>(app.sim.packets.size())) {
                auto& pkt = app.sim.packets[app.activeDragPacket];
                ImVec2 centerUV((float)pkt.cx, (float)pkt.cy);
                ImVec2 currentUV = screen_to_uv(io.MousePos, tl, br);
                ImVec2 deltaUV = currentUV - centerUV;
                pkt.kx = deltaUV.x / kMomentumUVScale;
                pkt.ky = deltaUV.y / kMomentumUVScale;
                app.packetDragDirty = true;
                app.fieldDirty = true;
            }
        } else if (app.dragAction == AppState::DragAction::AddBox || app.dragAction == AppState::DragAction::AddPacket) {
            // draw-only below
        } else if (app.dragAction == AppState::DragAction::AddWell) {
            // no preview
        }
    }

    if ((app.dragAction == AppState::DragAction::AddBox || app.dragAction == AppState::DragAction::AddPacket) && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImU32 col = make_rgba(1, 1, 1, 0.7f);
        dl->AddRect(app.dragStart, app.dragEnd, col, 0.0f, 0, 2.0f);
    }

    if ((app.dragAction != AppState::DragAction::None || app.pendingShiftToggle) && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br);
        ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);

        // Apply deferred shift-toggle if it was a click (not a drag)
        if (app.pendingShiftToggle && app.pendingShiftToggleIdx >= 0) {
            ImVec2 dpx = io.MousePos - app.mouseDownPos;
            float dist2 = dpx.x * dpx.x + dpx.y * dpx.y;
            if (dist2 <= kDragThresholdPx * kDragThresholdPx) {
                selection_toggle(app, app.pendingShiftToggleKind, app.pendingShiftToggleIdx);
            }
        }

        if (app.dragAction == AppState::DragAction::AddBox) {


            sim::Box b;
            b.x0 = std::clamp((double)uv0.x, 0.0, 1.0);
            b.y0 = std::clamp((double)uv0.y, 0.0, 1.0);
            b.x1 = std::clamp((double)uv1.x, 0.0, 1.0);
            b.y1 = std::clamp((double)uv1.y, 0.0, 1.0);
            b.height = app.boxHeight;
            int newIndex = static_cast<int>(app.sim.pfield.boxes.size());
            app.sim.addBox(b);
            selection_set_single(app, AppState::SelectedItem::Kind::Box, newIndex);
            app.boxEditorOpen = true;
            app.boxEditorPos = io.MousePos + ImVec2(16, 16);
            app.fieldDirty = true;
        } else if (app.dragAction == AppState::DragAction::AddPacket) {
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
            selection_set_single(app, AppState::SelectedItem::Kind::Packet, static_cast<int>(app.sim.packets.size()) - 1);
            app.packetEditorOpen = true;
            app.packetEditorPos = io.MousePos + ImVec2(16, 16);
            app.fieldDirty = true;
        } else if (app.dragAction == AppState::DragAction::AddWell) {
            ImVec2 uv_center = screen_to_uv(app.dragEnd, tl, br);
            sim::RadialWell w;
            w.cx = std::clamp((double)uv_center.x, 0.0, 1.0);
            w.cy = std::clamp((double)uv_center.y, 0.0, 1.0);
            w.strength = app.wellStrength;
            w.radius = std::clamp(app.wellRadius, 0.01, 0.5);
            w.profile = app.wellProfile;
            app.sim.addWell(w);
            selection_set_single(app, AppState::SelectedItem::Kind::Well, static_cast<int>(app.sim.pfield.wells.size()) - 1);
            app.wellEditorOpen = true;
            app.wellEditorPos = io.MousePos + ImVec2(16, 16);
            app.fieldDirty = true;
        } else if (app.dragAction == AppState::DragAction::AdjustPacketMomentum) {
            if (app.packetDragDirty) {
                app.sim.reset();
                app.fieldDirty = true;
            }
        } else if (app.dragAction == AppState::DragAction::MoveSelection) {
            if (app.pendingPacketClick && app.dragPrimaryKind == AppState::SelectedItem::Kind::Packet) {
                selection_set_single(app, AppState::SelectedItem::Kind::Packet, app.dragPrimaryIdx);
                app.packetEditorOpen = true;
                app.packetEditorPos = io.MousePos + ImVec2(16, 16);
            } else if (app.selectionDragDirty) {
                app.sim.reset();
                app.fieldDirty = true;
            }
        } else if (app.dragAction == AppState::DragAction::AdjustBoxEdge) {
            // potential already rebuilt during drag
        }

        if (app.potentialDirtyDrag) {
            app.sim.pfield.build(app.sim.V);
            app.sim.refresh_diagnostics_baseline();
            app.potentialDirtyDrag = false;
            app.fieldDirty = true;
        }

        app.dragAction = AppState::DragAction::None;
        app.activeDragPacket = -1;
        app.activeDragWell = -1;
        app.pendingPacketClick = false;
        app.packetDragDirty = false;
        app.selectionDragDirty = false;
        app.dragBoxEdge = AppState::BoxEdge::None;
        app.dragPrimaryIdx = -1;
        app.pendingShiftToggle = false;
        app.pendingShiftToggleIdx = -1;
        app.potentialDirtyDrag = false;
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
        if (ImGui::IsKeyPressed(ImGuiKey_R)) {
            app.sim.reset();
            app.fieldDirty = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Delete)) {
            // Delete single-object selection (keep existing behavior)
            if (app.selectedBox >= 0) {
                app.sim.pfield.boxes.erase(app.sim.pfield.boxes.begin() + app.selectedBox);
                selection_clear(app);
                app.sim.reset();
                app.fieldDirty = true;
            } else if (app.selectedWell >= 0) {
                if (app.selectedWell < static_cast<int>(app.sim.pfield.wells.size())) {
                    app.sim.pfield.wells.erase(app.sim.pfield.wells.begin() + app.selectedWell);
                    selection_clear(app);
                    app.sim.reset();
                    app.fieldDirty = true;
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
                    app.sim.refresh_diagnostics_baseline();
                    app.fieldDirty = true;
                }
                if (ImGui::Button("Delete box")) {
                    app.sim.pfield.boxes.erase(app.sim.pfield.boxes.begin() + app.selectedBox);
                    app.selectedBox = -1;
                    app.boxEditorOpen = false;
                    app.sim.reset();
                    app.fieldDirty = true;
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
                    app.fieldDirty = true;
                }
                if (ImGui::Button("Re-inject")) {
                    app.sim.reset();
                    app.fieldDirty = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Delete packet")) {
                    app.sim.packets.erase(app.sim.packets.begin() + app.selectedPacket);
                    app.selectedPacket = -1;
                    app.packetEditorOpen = false;
                    app.sim.reset();
                    app.fieldDirty = true;
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
                    app.sim.refresh_diagnostics_baseline();
                    app.fieldDirty = true;
                }
                if (ImGui::Button("Delete well")) {
                    app.sim.pfield.wells.erase(app.sim.pfield.wells.begin() + app.selectedWell);
                    app.selectedWell = -1;
                    app.wellEditorOpen = false;
                    app.sim.reset();
                    app.fieldDirty = true;
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
        auto& t = app.styleTokens;
        bool changed = false;
        changed |= ImGui::SliderFloat("Corner rounding", &t.cornerRounding, 0.0f, 14.0f, "%.1f");
        changed |= ImGui::SliderFloat("Border weight", &t.borderWeight, 0.0f, 3.0f, "%.2f");
        changed |= ImGui::SliderFloat("Density", &t.density, 0.75f, 1.4f, "%.2f");
        changed |= ImGui::SliderFloat("Font scale", &t.fontScale, 0.85f, 1.35f, "%.2f");
        changed |= ImGui::SliderFloat("Accent hue", &t.accentHue, 0.0f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Accent saturation", &t.accentSat, 0.2f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Accent value", &t.accentVal, 0.3f, 1.0f, "%.2f");
        changed |= ImGui::SliderFloat("Panel lift", &t.panelLift, 0.0f, 0.18f, "%.2f");
        changed |= ImGui::Checkbox("Anti-aliasing", &t.antiAliased);

        if (ImGui::Button("Reset style tokens")) {
            t = AppState::StyleTokens{};
            changed = true;
        }

        if (changed) {
            apply_style_tokens(app);
        }
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
            app.styleTokens = AppState::StyleTokens{};
            apply_style_tokens(app);
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
    apply_style_tokens(app);
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
                    selection_clear(app);
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
            app.sim.stepN(std::max(1, app.stepsPerFrame));
            app.fieldDirty = true;
        }

        if (app.sim.diagnostics.unstable && !app.lastUnstable) {
            push_toast(app, std::string("Instability: ") + app.sim.diagnostics.reason, 4.0f);
            app.lastUnstable = true;
        } else if (!app.sim.diagnostics.unstable) {
            app.lastUnstable = false;
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
