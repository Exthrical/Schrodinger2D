// Schrodinger2D - main entry
// - GUI mode: Dear ImGui + GLFW + OpenGL2 (if available)
// - Headless mode: runs a small smoke example via --example

#include <iostream>
#include <string>
#include <vector>

#include "io/scene.hpp"

#if BUILD_GUI
// GUI includes (GLFW + ImGui backends)
#include <GLFW/glfw3.h>
#include "ui/gui.hpp"
#endif

static void print_usage() {
    std::cout << "Schrodinger2D\n"
              << "Usage:\n"
              << "  Schrodinger2D                 # launch GUI (if available)\n"
              << "  Schrodinger2D --example [path]# run headless smoke example\n";
}

int main(int argc, char** argv) {
    std::string example_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--example") {
            if (i + 1 < argc && argv[i+1][0] != '-') {
                example_path = argv[++i];
            } else {
                example_path = "examples/smoke_example.json"; // default
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        }
    }

    if (!example_path.empty()) {
        return io::run_example_cli(example_path);
    }

#if BUILD_GUI
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW. Use --example for headless.\n";
        return 1;
    }

    // Setup window hints suitable for OpenGL2 backend
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    GLFWwindow* window = glfwCreateWindow(1280, 800, "Schrodinger2D", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window.\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // Run GUI loop (creates ImGui context inside)
    int ret = run_gui(window);

    glfwDestroyWindow(window);
    glfwTerminate();
    return ret;
#else
    std::cerr << "GUI not available. Run with --example to execute headless smoke test.\n";
    print_usage();
    return 1;
#endif
}

#if defined(_WIN32) && BUILD_GUI
#include <windows.h>
int APIENTRY WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    return main(__argc, __argv);
}
#endif
