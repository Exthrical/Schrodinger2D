#if BUILD_GUI

#include "gui.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl2.h>

#include <GLFW/glfw3.h>
#if defined(_WIN32)
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

    // Packet UI
    double packetAmplitude{1.0};
    double packetSigma{0.05};
    double packetKx{12.0};
    double packetKy{0.0};

    // Box UI
    double boxHeight{200.0};

    // Interaction
    enum class Mode { Select, AddPacket, AddBox } mode{Mode::Select};
    int selectedBox{-1};
    bool dragging{false};
    ImVec2 dragStart{0,0};
    ImVec2 dragEnd{0,0};

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
    int nx = app.sim.Nx, ny = app.sim.Ny;
    if (ImGui::InputInt("Nx", &nx) | ImGui::InputInt("Ny", &ny)) {
        nx = std::clamp(nx, 16, 1024);
        ny = std::clamp(ny, 16, 1024);
        if (nx != app.sim.Nx || ny != app.sim.Ny) {
            app.sim.resize(nx, ny);
            app.selectedBox = -1;
        }
    }
    double mass = app.sim.mass();
    double left=0, right=0; app.sim.mass_split(left, right);
    ImGui::Text("Mass: %.6f  |  Left: %.6f  Right: %.6f", mass, left, right);
    ImGui::Separator();

    ImGui::Text("View");
    ImGui::Checkbox("Normalize view", &app.normalizeView);
    ImGui::Checkbox("Potential overlay", &app.showPotential);
    int vm = (int)app.view;
    const char* modes[] = {"Mag+Phase","Real","Imag","Magnitude","Phase"};
    if (ImGui::Combo("Mode", &vm, modes, IM_ARRAYSIZE(modes))) app.view = (sim::ViewMode)vm;
    ImGui::Separator();

    ImGui::Text("Objects");
    int mode = (int)app.mode;
    const char* omodes[] = {"Select/Move Boxes", "Add Packet", "Add Box"};
    ImGui::Combo("Tool", &mode, omodes, IM_ARRAYSIZE(omodes));
    app.mode = (AppState::Mode)mode;

    ImGui::Text("Packet");
    {
        double vmin, vmax;
        vmin=0.1; vmax=5.0; ImGui::SliderScalar("Amplitude", ImGuiDataType_Double, &app.packetAmplitude, &vmin, &vmax, "%.3f");
        vmin=0.01; vmax=0.2; ImGui::SliderScalar("Sigma", ImGuiDataType_Double, &app.packetSigma, &vmin, &vmax, "%.3f");
        vmin=-80.0; vmax=80.0; ImGui::SliderScalar("k_x", ImGuiDataType_Double, &app.packetKx, &vmin, &vmax, "%.1f");
        vmin=-80.0; vmax=80.0; ImGui::SliderScalar("k_y", ImGuiDataType_Double, &app.packetKy, &vmin, &vmax, "%.1f");
    }

    ImGui::Text("Boxes");
    {
        double vmin, vmax;
        vmin=-1000.0; vmax=1000.0; ImGui::SliderScalar("Height", ImGuiDataType_Double, &app.boxHeight, &vmin, &vmax, "%.2f");
        vmin=0.0; vmax=5.0; ImGui::SliderScalar("CAP strength", ImGuiDataType_Double, &app.sim.pfield.cap_strength, &vmin, &vmax, "%.2f");
        vmin=0.02; vmax=0.25; ImGui::SliderScalar("CAP ratio", ImGuiDataType_Double, &app.sim.pfield.cap_ratio, &vmin, &vmax, "%.3f");
    }
    if (ImGui::Button("Rebuild V")) app.sim.pfield.build(app.sim.V);
    if (!app.sim.pfield.boxes.empty()) {
        ImGui::Text("%d box(es)", (int)app.sim.pfield.boxes.size());
        if (app.selectedBox >= 0 && app.selectedBox < (int)app.sim.pfield.boxes.size()) {
            auto& b = app.sim.pfield.boxes[app.selectedBox];
            ImGui::Text("Selected #%d", app.selectedBox);
            ImGui::DragScalarN("Rect [x0,y0,x1,y1]", ImGuiDataType_Double, &b.x0, 4, 0.001f, nullptr, nullptr);
            ImGui::DragScalar("Height", ImGuiDataType_Double, &b.height, 0.1f);
            if (ImGui::Button("Delete selected [Del]")) {
                app.sim.pfield.boxes.erase(app.sim.pfield.boxes.begin() + app.selectedBox);
                app.selectedBox = -1;
                app.sim.pfield.build(app.sim.V);
            }
        }
        ImGui::SameLine(); if (ImGui::Button("Clear boxes")) { app.sim.pfield.boxes.clear(); app.sim.pfield.build(app.sim.V); app.selectedBox = -1; }
    }
    if (!app.sim.packets.empty()) {
        if (ImGui::Button("Clear packets")) { app.sim.packets.clear(); app.sim.reset(); }
        ImGui::SameLine(); ImGui::Text("%d packet(s)", (int)app.sim.packets.size());
    }

    ImGui::Separator();
    ImGui::Text("Scene IO");
    static char path[256] = "examples/scene.json";
    ImGui::InputText("Path", path, sizeof(path));
    if (ImGui::Button("Save JSON")) {
        io::Scene scene; io::from_simulation(app.sim, scene);
        io::save_scene(path, scene);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load JSON")) {
        io::Scene scene; if (io::load_scene(path, scene)) {
            io::to_simulation(scene, app.sim);
            app.selectedBox = -1;
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

    // Mouse interactions over image
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 tl = cur; // top-left
    ImVec2 br = cur + target; // bottom-right

    ImGuiIO& io = ImGui::GetIO();
    bool hovered = ImGui::IsItemHovered();
    if (hovered) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            app.dragging = true; app.dragStart = io.MousePos; app.dragEnd = io.MousePos;
            if (app.mode == AppState::Mode::Select) {
                // Select box under cursor
                ImVec2 uv = screen_to_uv(io.MousePos, tl, br);
                for (int bi = (int)app.sim.pfield.boxes.size()-1; bi >= 0; --bi) {
                    const auto& b = app.sim.pfield.boxes[bi];
                    if (uv.x >= (float)std::min(b.x0,b.x1) && uv.x <= (float)std::max(b.x0,b.x1) &&
                        uv.y >= (float)std::min(b.y0,b.y1) && uv.y <= (float)std::max(b.y0,b.y1)) {
                        app.selectedBox = bi; break;
                    }
                }
            }
        }
        if (app.dragging && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            app.dragEnd = io.MousePos;
            if (app.mode == AppState::Mode::Select && app.selectedBox >= 0) {
                // Drag to move selected box
                ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br);
                ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);
                ImVec2 d = uv1 - uv0;
                auto& b = app.sim.pfield.boxes[app.selectedBox];
                b.x0 += d.x; b.x1 += d.x; b.y0 += d.y; b.y1 += d.y;
                app.dragStart = app.dragEnd; // incremental
                app.sim.pfield.build(app.sim.V);
            }
        }
        if (app.dragging && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            ImVec2 uv0 = screen_to_uv(app.dragStart, tl, br); // last start
            ImVec2 uv1 = screen_to_uv(app.dragEnd, tl, br);
            if (app.mode == AppState::Mode::AddBox) {
                sim::Box b;
                b.x0 = std::clamp((double)uv0.x, 0.0, 1.0);
                b.y0 = std::clamp((double)uv0.y, 0.0, 1.0);
                b.x1 = std::clamp((double)uv1.x, 0.0, 1.0);
                b.y1 = std::clamp((double)uv1.y, 0.0, 1.0);
                b.height = app.boxHeight;
                app.sim.addBox(b);
            } else if (app.mode == AppState::Mode::AddPacket) {
                // drag defines direction; release to place
                ImVec2 uv_center = screen_to_uv(app.dragStart, tl, br);
                ImVec2 uv_release = screen_to_uv(app.dragEnd, tl, br);
                sim::Packet p;
                p.cx = std::clamp((double)uv_center.x, 0.0, 1.0);
                p.cy = std::clamp((double)uv_center.y, 0.0, 1.0);
                p.sigma = app.packetSigma;
                p.amplitude = app.packetAmplitude;
                // momentum from drag direction or UI
                ImVec2 d = uv_release - uv_center;
                if (std::fabs(d.x) + std::fabs(d.y) > 1e-6f) {
                    float norm = std::sqrt(d.x*d.x + d.y*d.y);
                    p.kx = (d.x / norm) * app.packetKx;
                    p.ky = (d.y / norm) * app.packetKy;
                } else {
                    p.kx = app.packetKx; p.ky = app.packetKy;
                }
                app.sim.packets.push_back(p);
                app.sim.injectGaussian(p);
            }
            app.dragging = false;
        }
    }

    // Visualize drag rectangle
    if (app.dragging) {
        ImU32 col = make_rgba(1,1,1,0.7f);
        dl->AddRect(app.dragStart, app.dragEnd, col, 0.0f, 0, 2.0f);
    }

    // Show crosshair sampling info
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

    // Keyboard shortcuts
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) app.sim.running = !app.sim.running;
        if (ImGui::IsKeyPressed(ImGuiKey_R)) app.sim.reset();
        if (ImGui::IsKeyPressed(ImGuiKey_Delete) && app.selectedBox >= 0) {
            app.sim.pfield.boxes.erase(app.sim.pfield.boxes.begin()+app.selectedBox);
            app.selectedBox = -1; app.sim.pfield.build(app.sim.V);
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
        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
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
