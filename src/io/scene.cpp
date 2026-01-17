#include "scene.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <regex>
#include <iostream>

namespace io {

bool save_scene(const std::string& path, const Scene& s) {
    std::ofstream f(path);
    if (!f) return false;
    f << std::setprecision(17);
    f << "{\n";
    f << "  \"Nx\": " << s.Nx << ",\n";
    f << "  \"Ny\": " << s.Ny << ",\n";
    f << "  \"dt\": " << s.dt << ",\n";
    f << "  \"cap_strength\": " << s.cap_strength << ",\n";
    f << "  \"cap_ratio\": " << s.cap_ratio << ",\n";
    f << "  \"steps\": " << s.steps << ",\n";
    f << "  \"boxes\": [\n";
    for (size_t i = 0; i < s.boxes.size(); ++i) {
        const auto& b = s.boxes[i];
        f << "    {\"x0\": "<<b.x0<<", \"y0\": "<<b.y0<<", \"x1\": "<<b.x1<<", \"y1\": "<<b.y1<<", \"height\": "<<b.height<<"}";
        if (i + 1 < s.boxes.size()) f << ",";
        f << "\n";
    }
    f << "  ],\n";
    f << "  \"wells\": [\n";
    for (size_t i = 0; i < s.wells.size(); ++i) {
        const auto& w = s.wells[i];
        f << "    {\"cx\": "<<w.cx<<", \"cy\": "<<w.cy<<", \"strength\": "<<w.strength
          << ", \"radius\": "<<w.radius<<", \"profile\": "<<w.profile<<"}";
        if (i + 1 < s.wells.size()) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    f << "  \"packets\": [\n";
    for (size_t i = 0; i < s.packets.size(); ++i) {
        const auto& p = s.packets[i];
        f << "    {\"cx\": "<<p.cx<<", \"cy\": "<<p.cy<<", \"sigma\": "<<p.sigma
          << ", \"amplitude\": "<<p.amplitude<<", \"kx\": "<<p.kx<<", \"ky\": "<<p.ky<<"}";
        if (i + 1 < s.packets.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
    return true;
}

static bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

static double extract_number(const std::string& s, const std::string& key, double def) {
    // match e.g. "dt": 0.001
    std::regex re(std::string("\\\"") + key + std::string("\\\"\\s*:\\s*([-+eE0-9\\.]+)"));
    std::smatch m;
    if (std::regex_search(s, m, re)) return std::stod(m[1].str());
    return def;
}

static int extract_int(const std::string& s, const std::string& key, int def) {
    std::regex re(std::string("\\\"") + key + std::string("\\\"\\s*:\\s*([-+0-9]+)"));
    std::smatch m;
    if (std::regex_search(s, m, re)) return std::stoi(m[1].str());
    return def;
}

bool load_scene(const std::string& path, Scene& sc) {
    std::string txt; if (!read_file(path, txt)) return false;
    sc.Nx = extract_int(txt, "Nx", sc.Nx);
    sc.Ny = extract_int(txt, "Ny", sc.Ny);
    sc.dt = extract_number(txt, "dt", sc.dt);
    sc.cap_strength = extract_number(txt, "cap_strength", sc.cap_strength);
    sc.cap_ratio = extract_number(txt, "cap_ratio", sc.cap_ratio);
    sc.steps = extract_int(txt, "steps", sc.steps);

    // Very simple array parsing for boxes/wells/packets created by our writer
    sc.boxes.clear();
    sc.wells.clear();
    sc.packets.clear();

    // Boxes
    {
        std::regex re(R"_(\{\s*\"x0\"\s*:\s*([-+eE0-9\.]+).*?\"y0\"\s*:\s*([-+eE0-9\.]+).*?\"x1\"\s*:\s*([-+eE0-9\.]+).*?\"y1\"\s*:\s*([-+eE0-9\.]+).*?\"height\"\s*:\s*([-+eE0-9\.]+)\s*\})_");
        auto it = std::sregex_iterator(txt.begin(), txt.end(), re);
        auto end = std::sregex_iterator();
        for (; it != end; ++it) {
            SceneBox b{};
            b.x0 = std::stod((*it)[1].str());
            b.y0 = std::stod((*it)[2].str());
            b.x1 = std::stod((*it)[3].str());
            b.y1 = std::stod((*it)[4].str());
            b.height = std::stod((*it)[5].str());
            sc.boxes.push_back(b);
        }
    }
    // Wells
    {
        std::regex re(R"_(\{\s*\"cx\"\s*:\s*([-+eE0-9\.]+).*?\"cy\"\s*:\s*([-+eE0-9\.]+).*?\"strength\"\s*:\s*([-+eE0-9\.]+).*?\"radius\"\s*:\s*([-+eE0-9\.]+).*?\"profile\"\s*:\s*([-+0-9]+)\s*\})_");
        auto it = std::sregex_iterator(txt.begin(), txt.end(), re);
        auto end = std::sregex_iterator();
        for (; it != end; ++it) {
            SceneWell w{};
            w.cx = std::stod((*it)[1].str());
            w.cy = std::stod((*it)[2].str());
            w.strength = std::stod((*it)[3].str());
            w.radius = std::stod((*it)[4].str());
            w.profile = std::stoi((*it)[5].str());
            sc.wells.push_back(w);
        }
    }

    // Packets
    {
        std::regex re(R"_(\{\s*\"cx\"\s*:\s*([-+eE0-9\.]+).*?\"cy\"\s*:\s*([-+eE0-9\.]+).*?\"sigma\"\s*:\s*([-+eE0-9\.]+).*?\"amplitude\"\s*:\s*([-+eE0-9\.]+).*?\"kx\"\s*:\s*([-+eE0-9\.]+).*?\"ky\"\s*:\s*([-+eE0-9\.]+)\s*\})_");
        auto it = std::sregex_iterator(txt.begin(), txt.end(), re);
        auto end = std::sregex_iterator();
        for (; it != end; ++it) {
            // Avoid re-parsing wells: require keys unique to packets
            if ((*it).str().find("\"sigma\"") == std::string::npos) continue;
            ScenePacket p{};
            p.cx = std::stod((*it)[1].str());
            p.cy = std::stod((*it)[2].str());
            p.sigma = std::stod((*it)[3].str());
            p.amplitude = std::stod((*it)[4].str());
            p.kx = std::stod((*it)[5].str());
            p.ky = std::stod((*it)[6].str());
            sc.packets.push_back(p);
        }
    }
    return true;
}

void from_simulation(const sim::Simulation& sim, Scene& s) {
    s.Nx = sim.Nx; s.Ny = sim.Ny; s.dt = sim.dt;
    s.cap_ratio = sim.pfield.cap_ratio; s.cap_strength = sim.pfield.cap_strength;
    s.boxes.clear();
    s.wells.clear();
    s.packets.clear();
    for (const auto& b : sim.pfield.boxes) s.boxes.push_back({b.x0,b.y0,b.x1,b.y1,b.height});
    for (const auto& w : sim.pfield.wells) {
        s.wells.push_back({w.cx, w.cy, w.strength, w.radius, (int)w.profile});
    }
    for (const auto& p : sim.packets) s.packets.push_back({p.cx,p.cy,p.sigma,p.amplitude,p.kx,p.ky});
}

void to_simulation(const Scene& s, sim::Simulation& sim) {
    sim.resize(s.Nx, s.Ny);
    sim.dt = s.dt;
    sim.pfield.boxes.clear();
    for (auto& b : s.boxes) sim.pfield.boxes.push_back({b.x0,b.y0,b.x1,b.y1,b.height});
    sim.pfield.wells.clear();
    for (auto& w : s.wells) {
        sim::RadialWell rw;
        rw.cx = w.cx;
        rw.cy = w.cy;
        rw.strength = w.strength;
        rw.radius = w.radius;
        rw.profile = static_cast<sim::RadialWell::Profile>(w.profile);
        sim.pfield.wells.push_back(rw);
    }
    sim.pfield.cap_ratio = s.cap_ratio;
    sim.pfield.cap_strength = s.cap_strength;
    sim.pfield.build(sim.V);
    sim.packets.clear();
    for (auto& p : s.packets) sim.packets.push_back({p.cx,p.cy,p.sigma,p.amplitude,p.kx,p.ky});
    sim.reset();
}

int run_example_cli(const std::string& scene_path) {
    Scene s;
    if (!scene_path.empty()) {
        if (!load_scene(scene_path, s)) {
            std::cerr << "Failed to load scene: " << scene_path << "\n";
            return 2;
        }
    }
    sim::Simulation sim;
    to_simulation(s, sim);

    // Run N steps
    for (int i = 0; i < s.steps; ++i) sim.step();

    // Diagnostics: norm and split mass (approx transmission/reflection)
    double M = sim.mass();
    double L=0,R=0; sim.mass_split(L,R);
    std::cout << "Diagnostics\n";
    std::cout << "Nx="<<sim.Nx<<" Ny="<<sim.Ny<<" dt="<<sim.dt<<" steps="<<s.steps<<"\n";
    std::cout << std::setprecision(8);
    std::cout << "Mass="<<M<<" Left="<<L<<" Right="<<R<<"\n";
    return 0;
}

} // namespace io

