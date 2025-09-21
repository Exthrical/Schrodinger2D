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

#include "sim/simulation.hpp"
#include "io/scene.hpp"

namespace {

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
    double boxHeight{200.0};

    // Interaction state
    enum class Mode { Select, AddPacket, AddBox } mode{Mode::Select};
    int selectedBox{-1};
    int selectedPacket{-1};
    bool dragging{false};
    ImVec2 dragStart{0,0};
    ImVec2 dragEnd{0,0};

    bool boxEditorOpen{false};
    bool packetEditorOpen{false};
    ImVec2 boxEditorPos{0,0};
    ImVec2 packetEditorPos{0,0};

    // GL texture for field visualization
    GLuint tex{0};
    int texW{0}, texH{0};
};

static inline ImVec2 operator+(ImVec2 a, ImVec2 b) { return ImVec2(a.x+b.x, a.y+b.y); }
static inline ImVec2 operator-(ImVec2 a, ImVec2 b) { return ImVec2(a.x-b.x, a.y-b.y); }
static inline ImVec2 operator*(ImVec2 a, float s) { return ImVec2(a.x*s, a.y*s); }

static void ensure_texture(AppState& app, int w, int h) {
    if (app.tex == 0) {
        glGenTextures(1, &app.tex);
    }
    if (app.texW != w || app.texH != h) {
        app.texW = w; app.texH = h;
        glBindTexture(GL_TEXTURE_2D, app.tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

static ImU32 make_rgba(float r, float g, float b, float a=1.0f) {
    r = std::clamp(r, 0.0f, 1.0f);
    g = std::clamp(g, 0.0f, 1.0f);
    b = std::clamp(b, 0.0f, 1.0f);
    a = std::clamp(a, 0.0f, 1.0f);
    return IM_COL32((int)(r*255.0f), (int)(g*255.0f), (int)(b*255.0f), (int)(a*255.0f));
}

// Map psi -> RGBA color
static void render_field_to_rgba(const sim::Simulation& sim, std::vector<unsigned char>& outRGBA,
                                 bool showPotential, sim::ViewMode view, bool normalizeView) {
    const int W = sim.Nx, H = sim.Ny;
    outRGBA.resize((size_t)W*H*4);
    double maxmag = 1.0;
    if (normalizeView) {
        double m = 1e-12;
        for (auto& z : sim.psi) {
            m = std::max(m, (double)std::abs(z));
        }
        maxmag = m;
    }
    // Dynamic potential overlay scaling for visibility across ranges
    double maxVre = 0.0;
    if (showPotential) {
        for (int j = 0; j < H; ++j) {
            for (int i = 0; i < W; ++i) {
                maxVre = std::max(maxVre, std::abs((double)std::real(sim.V[sim.idx(i,j)])));
            }
        }
    }
    const double Vscale = (maxVre > 1e-12 ? 0.2 * maxVre : 20.0); // map 20% of max to full tint
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            auto z = sim.psi[sim.idx(i,j)];
            float r=0,g=0,b=0;
            if (view == sim::ViewMode::Real) {
                float v = (float)(0.5 + 0.5 * (std::real(z) / maxmag));
                r=g=b=v;
            } else if (view == sim::ViewMode::Imag) {
                float v = (float)(0.5 + 0.5 * (std::imag(z) / maxmag));
                r=g=b=v;
            } else if (view == sim::ViewMode::Magnitude) {
                float v = (float)std::min(1.0, std::abs(z) / maxmag);
                r = g = b = v;
            } else if (view == sim::ViewMode::Phase) {
                const double PI = 3.14159265358979323846;
                double phase = std::atan2(std::imag(z), std::real(z));
                float h = (float)((phase + PI) / (2.0 * PI));
                float s = 1.0f;
                float v = normalizeView ? 1.0f : (float)std::min(1.0, std::abs(z) / maxmag);
                // HSV to RGB
                float c = v * s;
                float x = c * (1 - (float)std::fabs(std::fmod(h * 6.0f, 2.0f) - 1));
                float m = v - c;
                float rr=0,gg=0,bb=0;
                int hi = (int)std::floor(h * 6.0f) % 6;
                if (hi==0)      { rr=c; gg=x; bb=0; }
                else if (hi==1) { rr=x; gg=c; bb=0; }
                else if (hi==2) { rr=0; gg=c; bb=x; }
                else if (hi==3) { rr=0; gg=x; bb=c; }
                else if (hi==4) { rr=x; gg=0; bb=c; }
                else            { rr=c; gg=0; bb=x; }
                r = rr + m; g = gg + m; b = bb + m;
            } else { // MagnitudePhase
                const double PI = 3.14159265358979323846;
                double mag = std::abs(z) / maxmag;
                double phase = std::atan2(std::imag(z), std::real(z));
                float h = (float)((phase + PI) / (2.0 * PI));
                float s = 1.0f;
                float v = (float)std::min(1.0, mag);
                float c = v * s;
                float x = c * (1 - (float)std::fabs(std::fmod(h * 6.0f, 2.0f) - 1));
                float m = v - c;
                float rr=0,gg=0,bb=0;
                int hi = (int)std::floor(h * 6.0f) % 6;
                if (hi==0)      { rr=c; gg=x; bb=0; }
                else if (hi==1) { rr=x; gg=c; bb=0; }
                else if (hi==2) { rr=0; gg=c; bb=x; }
                else if (hi==3) { rr=0; gg=x; bb=c; }
                else if (hi==4) { rr=x; gg=0; bb=c; }
                else            { rr=c; gg=0; bb=x; }
                r = rr + m; g = gg + m; b = bb + m;
            }

            // Potential overlay as faint red (barrier) / blue (well)
            float a = 1.0f;
            if (showPotential) {
                auto V = sim.V[sim.idx(i,j)];
                float pv = (float)std::clamp(std::real(V) / Vscale, -1.0, 1.0);
                if (pv > 0) { r = std::min(1.0f, r + pv * 0.3f); }
                else if (pv < 0) { b = std::min(1.0f, b + (-pv) * 0.3f); }
            }

            size_t k = (size_t)((j * W + i) * 4);
            outRGBA[k+0] = (unsigned char)std::round(r * 255.0f);
            outRGBA[k+1] = (unsigned char)std::round(g * 255.0f);
            outRGBA[k+2] = (unsigned char)std::round(b * 255.0f);
            outRGBA[k+3] = (unsigned char)std::round(a * 255.0f);
        }
    }
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

static void draw_object_editors(AppState& app);

static void draw_settings(AppState& app) {
    if (ImGui::Button(app.sim.running ? "Pause [Space]" : "Start [Space]")) {
        app.sim.running = !app.sim.running;
    }
    ImGui::SameLine();
    if (ImGui::Button("Step")) app.sim.step();
    ImGui::SameLine();
    if (ImGui::Button("Reset [R]")) app.sim.reset();
    ImGui::SameLine();
    if (ImGui::Button("Renormalize")) {
        double m = app.sim.mass();
        if (m > 1e-12) {
            double s = 1.0 / std::sqrt(m);
            for (auto& z : app.sim.psi) z *= s;
        }
    }
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
    ImGui::Text("Mass: %.6f  |  Left: %.6f  Right: %.6f", mass, left, right);
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
    int mode = static_cast<int>(app.mode);
    const char* omodes[] = {"Select/Move Boxes", "Add Packet", "Add Box"};
    if (ImGui::Combo("Tool", &mode, omodes, IM_ARRAYSIZE(omodes))) {
        app.mode = static_cast<AppState::Mode>(mode);
    }

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

    // Overlay potential boxes and packet markers so they can be picked.
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

    for (size_t pi = 0; pi < app.sim.packets.size(); ++pi) {
        const auto& pkt = app.sim.packets[pi];
        ImVec2 center = uv_to_screen(ImVec2((float)pkt.cx, (float)pkt.cy), tl, br);
        bool selected = (static_cast<int>(pi) == app.selectedPacket);
        float radius = selected ? 8.0f : 6.0f;
        ImU32 outline = selected ? make_rgba(0.2f, 0.9f, 1.0f, 0.95f) : make_rgba(0.2f, 0.6f, 1.0f, 0.8f);
        dl->AddCircle(center, radius, outline, 0, selected ? 3.0f : 2.0f);
        dl->AddCircleFilled(center, radius * 0.4f, make_rgba(0.2f, 0.6f, 1.0f, 0.8f));
    }

    ImGuiIO& io = ImGui::GetIO();
    bool hovered = ImGui::IsItemHovered();
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        app.dragStart = io.MousePos;
        app.dragEnd = io.MousePos;
        app.dragging = false;

        if (app.mode == AppState::Mode::Select) {
            const float packetPickRadius = 12.0f;
            int packetHit = -1;
            for (int pi = static_cast<int>(app.sim.packets.size()) - 1; pi >= 0; --pi) {
                ImVec2 center = uv_to_screen(ImVec2((float)app.sim.packets[pi].cx, (float)app.sim.packets[pi].cy), tl, br);
                float dx = center.x - io.MousePos.x;
                float dy = center.y - io.MousePos.y;
                if (dx * dx + dy * dy <= packetPickRadius * packetPickRadius) {
                    packetHit = pi;
                    break;
                }
            }

            if (packetHit >= 0) {
                app.selectedPacket = packetHit;
                app.packetEditorOpen = true;
                app.packetEditorPos = io.MousePos + ImVec2(16, 16);
                app.selectedBox = -1;
                app.boxEditorOpen = false;
            } else {
                ImVec2 uv = screen_to_uv(io.MousePos, tl, br);
                int boxHit = -1;
                for (int bi = static_cast<int>(app.sim.pfield.boxes.size()) - 1; bi >= 0; --bi) {
                    const auto& b = app.sim.pfield.boxes[bi];
                    double minx = std::min(b.x0, b.x1);
                    double maxx = std::max(b.x0, b.x1);
                    double miny = std::min(b.y0, b.y1);
                    double maxy = std::max(b.y0, b.y1);
                    if (uv.x >= minx && uv.x <= maxx && uv.y >= miny && uv.y <= maxy) {
                        boxHit = bi;
                        break;
                    }
                }
                if (boxHit >= 0) {
                    app.selectedBox = boxHit;
                    app.boxEditorOpen = true;
                    app.boxEditorPos = io.MousePos + ImVec2(16, 16);
                    app.dragging = true;
                    app.selectedPacket = -1;
                    app.packetEditorOpen = false;
                } else {
                    app.selectedBox = -1;
                    app.boxEditorOpen = false;
                    app.selectedPacket = -1;
                    app.packetEditorOpen = false;
                }
            }
        } else {
            app.dragging = true;
        }
    }

    if (app.dragging && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        app.dragEnd = io.MousePos;
        if (app.mode == AppState::Mode::Select && app.selectedBox >= 0) {
            ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br);
            ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);
            ImVec2 d = uv1 - uv0;
            auto& b = app.sim.pfield.boxes[app.selectedBox];
            b.x0 += d.x; b.x1 += d.x;
            b.y0 += d.y; b.y1 += d.y;
            app.dragStart = app.dragEnd;
            app.sim.pfield.build(app.sim.V);
        }
    }

    if (app.dragging && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br);
        ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);
        if (app.mode == AppState::Mode::AddBox) {
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
        } else if (app.mode == AppState::Mode::AddPacket) {
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
        }
        app.dragging = false;
    }

    if (app.dragging && (app.mode == AppState::Mode::AddBox || app.mode == AppState::Mode::AddPacket)) {
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

} // namespace

int run_gui(GLFWwindow* window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL2_Init();

    AppState app;
    // Load a quick default example: central barrier and right-moving packet
    app.sim.pfield.boxes.push_back(sim::Box{0.48, 0.0, 0.52, 1.0, 200.0});
    app.sim.pfield.build(app.sim.V);
    sim::Packet p{0.25, 0.5, 0.05, 1.0, 12.0, 0.0};
    app.sim.packets.push_back(p);
    app.sim.reset();

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Root window covering the entire viewport with fixed layout
        ImGuiIO& frame_io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(frame_io.DisplaySize, ImGuiCond_Always);
        ImGuiWindowFlags root_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                                      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        ImGui::Begin("Schrodinger2D", nullptr, root_flags);
        const float left_w = 360.0f; // fixed settings panel width
        ImGui::BeginChild("SettingsPanel", ImVec2(left_w, 0), true);
        draw_settings(app);
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("ViewPanel", ImVec2(0, 0), true);
        draw_view_content(app);
        ImGui::EndChild();
        ImGui::End();

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
