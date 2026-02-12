#include "scene.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cctype>
#include <stdexcept>

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
    f << "  \"rel_mass_drift_tol\": " << s.rel_mass_drift_tol << ",\n";
    f << "  \"rel_cap_mass_growth_tol\": " << s.rel_cap_mass_growth_tol << ",\n";
    f << "  \"rel_interior_mass_drift_tol\": " << s.rel_interior_mass_drift_tol << ",\n";
    f << "  \"interior_mass_drift_vs_total_tol\": " << s.interior_mass_drift_vs_total_tol << ",\n";
    f << "  \"min_initial_interior_mass_fraction\": " << s.min_initial_interior_mass_fraction << ",\n";
    f << "  \"min_interior_area_fraction\": " << s.min_interior_area_fraction << ",\n";
    f << "  \"stability_warmup_steps\": " << s.stability_warmup_steps << ",\n";
    f << "  \"interior_drift_hard_fail\": " << (s.interior_drift_hard_fail ? "true" : "false") << ",\n";
    f << "  \"auto_pause_on_instability\": " << (s.auto_pause_on_instability ? "true" : "false") << ",\n";
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

struct JsonValue {
    enum class Type { Null, Number, Bool, String, Array, Object } type{Type::Null};
    double number{0.0};
    bool boolean{false};
    std::string string;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : s_(text) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (!eof()) {
            throw std::runtime_error("unexpected trailing characters");
        }
        return v;
    }

private:
    const std::string& s_;
    size_t pos_{0};

    bool eof() const { return pos_ >= s_.size(); }
    char peek() const { return eof() ? '\0' : s_[pos_]; }
    char get() { return eof() ? '\0' : s_[pos_++]; }

    void skip_ws() {
        while (!eof() && std::isspace(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }
    }

    void expect(char c) {
        if (get() != c) {
            throw std::runtime_error("unexpected token");
        }
    }

    bool consume(const char* kw) {
        size_t n = std::char_traits<char>::length(kw);
        if (s_.compare(pos_, n, kw) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::Type::String;
            v.string = parse_string();
            return v;
        }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        if (consume("true")) {
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            v.boolean = true;
            return v;
        }
        if (consume("false")) {
            JsonValue v;
            v.type = JsonValue::Type::Bool;
            v.boolean = false;
            return v;
        }
        if (consume("null")) {
            JsonValue v;
            v.type = JsonValue::Type::Null;
            return v;
        }
        throw std::runtime_error("invalid json value");
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (!eof()) {
            char c = get();
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                if (eof()) throw std::runtime_error("invalid escape");
                char e = get();
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: throw std::runtime_error("unsupported escape");
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("unterminated string");
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (peek() == '0') {
            ++pos_;
        } else {
            if (peek() < '1' || peek() > '9') throw std::runtime_error("invalid number");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (peek() == '.') {
            ++pos_;
            if (peek() < '0' || peek() > '9') throw std::runtime_error("invalid fraction");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (peek() < '0' || peek() > '9') throw std::runtime_error("invalid exponent");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }

        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.number = std::stod(s_.substr(start, pos_ - start));
        return v;
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue v;
        v.type = JsonValue::Type::Array;
        skip_ws();
        if (peek() == ']') {
            get();
            return v;
        }
        while (true) {
            v.array.push_back(parse_value());
            skip_ws();
            if (peek() == ']') {
                get();
                return v;
            }
            expect(',');
            skip_ws();
        }
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue v;
        v.type = JsonValue::Type::Object;
        skip_ws();
        if (peek() == '}') {
            get();
            return v;
        }
        while (true) {
            if (peek() != '"') throw std::runtime_error("object key expected");
            std::string key = parse_string();
            skip_ws();
            expect(':');
            JsonValue value = parse_value();
            v.object.push_back({std::move(key), std::move(value)});
            skip_ws();
            if (peek() == '}') {
                get();
                return v;
            }
            expect(',');
            skip_ws();
        }
    }
};

static const JsonValue* get_member(const JsonValue& obj, const char* key) {
    if (obj.type != JsonValue::Type::Object) return nullptr;
    for (const auto& kv : obj.object) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

static double as_number(const JsonValue* v, double def) {
    if (!v || v->type != JsonValue::Type::Number) return def;
    return v->number;
}

static int as_int(const JsonValue* v, int def) {
    if (!v || v->type != JsonValue::Type::Number) return def;
    return static_cast<int>(std::llround(v->number));
}

static bool as_bool(const JsonValue* v, bool def) {
    if (!v || v->type != JsonValue::Type::Bool) return def;
    return v->boolean;
}

bool load_scene(const std::string& path, Scene& sc) {
    std::string txt;
    if (!read_file(path, txt)) return false;

    JsonValue root;
    try {
        root = JsonParser(txt).parse();
    } catch (const std::exception&) {
        return false;
    }
    if (root.type != JsonValue::Type::Object) return false;

    sc.Nx = as_int(get_member(root, "Nx"), sc.Nx);
    sc.Ny = as_int(get_member(root, "Ny"), sc.Ny);
    sc.dt = as_number(get_member(root, "dt"), sc.dt);
    sc.cap_strength = as_number(get_member(root, "cap_strength"), sc.cap_strength);
    sc.cap_ratio = as_number(get_member(root, "cap_ratio"), sc.cap_ratio);
    sc.steps = as_int(get_member(root, "steps"), sc.steps);
    sc.rel_mass_drift_tol = as_number(get_member(root, "rel_mass_drift_tol"), sc.rel_mass_drift_tol);
    sc.rel_cap_mass_growth_tol = as_number(get_member(root, "rel_cap_mass_growth_tol"), sc.rel_cap_mass_growth_tol);
    sc.rel_interior_mass_drift_tol = as_number(get_member(root, "rel_interior_mass_drift_tol"), sc.rel_interior_mass_drift_tol);
    sc.interior_mass_drift_vs_total_tol = as_number(get_member(root, "interior_mass_drift_vs_total_tol"), sc.interior_mass_drift_vs_total_tol);
    sc.min_initial_interior_mass_fraction = as_number(get_member(root, "min_initial_interior_mass_fraction"), sc.min_initial_interior_mass_fraction);
    sc.min_interior_area_fraction = as_number(get_member(root, "min_interior_area_fraction"), sc.min_interior_area_fraction);
    sc.stability_warmup_steps = as_int(get_member(root, "stability_warmup_steps"), sc.stability_warmup_steps);
    sc.interior_drift_hard_fail = as_bool(get_member(root, "interior_drift_hard_fail"), sc.interior_drift_hard_fail);
    sc.auto_pause_on_instability = as_bool(get_member(root, "auto_pause_on_instability"), sc.auto_pause_on_instability);

    sc.boxes.clear();
    sc.wells.clear();
    sc.packets.clear();

    if (const JsonValue* boxes = get_member(root, "boxes"); boxes && boxes->type == JsonValue::Type::Array) {
        for (const auto& item : boxes->array) {
            if (item.type != JsonValue::Type::Object) continue;
            SceneBox b{};
            b.x0 = as_number(get_member(item, "x0"), 0.0);
            b.y0 = as_number(get_member(item, "y0"), 0.0);
            b.x1 = as_number(get_member(item, "x1"), 0.0);
            b.y1 = as_number(get_member(item, "y1"), 0.0);
            b.height = as_number(get_member(item, "height"), 0.0);
            sc.boxes.push_back(b);
        }
    }

    if (const JsonValue* wells = get_member(root, "wells"); wells && wells->type == JsonValue::Type::Array) {
        for (const auto& item : wells->array) {
            if (item.type != JsonValue::Type::Object) continue;
            SceneWell w{};
            w.cx = as_number(get_member(item, "cx"), 0.0);
            w.cy = as_number(get_member(item, "cy"), 0.0);
            w.strength = as_number(get_member(item, "strength"), 0.0);
            w.radius = as_number(get_member(item, "radius"), 0.0);
            w.profile = as_int(get_member(item, "profile"), 0);
            sc.wells.push_back(w);
        }
    }

    if (const JsonValue* packets = get_member(root, "packets"); packets && packets->type == JsonValue::Type::Array) {
        for (const auto& item : packets->array) {
            if (item.type != JsonValue::Type::Object) continue;
            ScenePacket p{};
            p.cx = as_number(get_member(item, "cx"), 0.0);
            p.cy = as_number(get_member(item, "cy"), 0.0);
            p.sigma = as_number(get_member(item, "sigma"), 0.0);
            p.amplitude = as_number(get_member(item, "amplitude"), 0.0);
            p.kx = as_number(get_member(item, "kx"), 0.0);
            p.ky = as_number(get_member(item, "ky"), 0.0);
            sc.packets.push_back(p);
        }
    }

    return true;
}

void from_simulation(const sim::Simulation& srcSim, Scene& s) {
    s.Nx = srcSim.Nx;
    s.Ny = srcSim.Ny;
    s.dt = srcSim.dt;
    s.cap_ratio = srcSim.pfield.cap_ratio;
    s.cap_strength = srcSim.pfield.cap_strength;
    s.rel_mass_drift_tol = srcSim.stability.rel_mass_drift_tol;
    s.rel_cap_mass_growth_tol = srcSim.stability.rel_cap_mass_growth_tol;
    s.rel_interior_mass_drift_tol = srcSim.stability.rel_interior_mass_drift_tol;
    s.interior_mass_drift_vs_total_tol = srcSim.stability.interior_mass_drift_vs_total_tol;
    s.min_initial_interior_mass_fraction = srcSim.stability.min_initial_interior_mass_fraction;
    s.min_interior_area_fraction = srcSim.stability.min_interior_area_fraction;
    s.stability_warmup_steps = srcSim.stability.warmup_steps;
    s.interior_drift_hard_fail = srcSim.stability.interior_drift_hard_fail;
    s.auto_pause_on_instability = srcSim.stability.auto_pause_on_instability;
    s.boxes.clear();
    s.wells.clear();
    s.packets.clear();
    for (const auto& b : srcSim.pfield.boxes) s.boxes.push_back({b.x0,b.y0,b.x1,b.y1,b.height});
    for (const auto& w : srcSim.pfield.wells) {
        s.wells.push_back({w.cx, w.cy, w.strength, w.radius, (int)w.profile});
    }
    for (const auto& p : srcSim.packets) s.packets.push_back({p.cx,p.cy,p.sigma,p.amplitude,p.kx,p.ky});
}

void to_simulation(const Scene& s, sim::Simulation& dstSim) {
    dstSim.resize(s.Nx, s.Ny);
    dstSim.dt = s.dt;
    dstSim.pfield.boxes.clear();
    for (const auto& b : s.boxes) dstSim.pfield.boxes.push_back({b.x0,b.y0,b.x1,b.y1,b.height});
    dstSim.pfield.wells.clear();
    for (auto& w : s.wells) {
        sim::RadialWell rw;
        rw.cx = w.cx;
        rw.cy = w.cy;
        rw.strength = w.strength;
        rw.radius = w.radius;
        rw.profile = static_cast<sim::RadialWell::Profile>(w.profile);
        dstSim.pfield.wells.push_back(rw);
    }
    dstSim.pfield.cap_ratio = s.cap_ratio;
    dstSim.pfield.cap_strength = s.cap_strength;
    dstSim.stability.rel_mass_drift_tol = s.rel_mass_drift_tol;
    dstSim.stability.rel_cap_mass_growth_tol = s.rel_cap_mass_growth_tol;
    dstSim.stability.rel_interior_mass_drift_tol = s.rel_interior_mass_drift_tol;
    dstSim.stability.interior_mass_drift_vs_total_tol = s.interior_mass_drift_vs_total_tol;
    dstSim.stability.min_initial_interior_mass_fraction = s.min_initial_interior_mass_fraction;
    dstSim.stability.min_interior_area_fraction = s.min_interior_area_fraction;
    dstSim.stability.warmup_steps = s.stability_warmup_steps;
    dstSim.stability.interior_drift_hard_fail = s.interior_drift_hard_fail;
    dstSim.stability.auto_pause_on_instability = s.auto_pause_on_instability;
    dstSim.pfield.build(dstSim.V);
    dstSim.packets.clear();
    for (const auto& p : s.packets) dstSim.packets.push_back({p.cx,p.cy,p.sigma,p.amplitude,p.kx,p.ky});
    dstSim.reset();
}

int run_example_cli(const std::string& scene_path) {
    Scene s;
    if (!scene_path.empty()) {
        if (!load_scene(scene_path, s)) {
            std::cerr << "Failed to load scene: " << scene_path << "\n";
            return 2;
        }
    }
    sim::Simulation simulation;
    to_simulation(s, simulation);

    // Run N steps
    for (int i = 0; i < s.steps; ++i) simulation.step();

    // Diagnostics: norm and split mass (approx transmission/reflection)
    double M = simulation.mass();
    double L = 0.0;
    double R = 0.0;
    simulation.mass_split(L, R);
    const auto& diag = simulation.diagnostics;
    std::cout << "Diagnostics\n";
    std::cout << "Nx=" << simulation.Nx << " Ny=" << simulation.Ny << " dt=" << simulation.dt << " steps=" << s.steps << "\n";
    std::cout << std::setprecision(8);
    std::cout << "Mass=" << M << " Left=" << L << " Right=" << R
              << " Interior=" << diag.current_interior_mass
              << " Drift=" << diag.rel_mass_drift
              << " InteriorDrift=" << diag.rel_interior_mass_drift
              << " InteriorDriftVsTotal=" << diag.rel_interior_mass_drift_vs_total << "\n";
    if (diag.warning) {
        std::cout << "Stability=WARNING reason=\"" << diag.warning_reason << "\"\n";
    }
    if (!diag.interior_guard_active) {
        std::cout << "InteriorGuard=DISABLED reason=\"" << diag.interior_guard_reason << "\"\n";
    }
    if (diag.unstable) {
        std::cout << "Stability=UNSTABLE reason=\"" << diag.reason << "\"\n";
        return 3;
    }
    if (!diag.warning) {
        std::cout << "Stability=OK\n";
    }
    return 0;
}

} // namespace io

