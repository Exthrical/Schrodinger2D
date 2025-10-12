cmake_minimum_required(VERSION 3.16)
project(Schrodinger2D CXX)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_GUI "Build with GUI (GLFW + OpenGL)" ON)

# Source groups
file(GLOB SIM_SRC
    src/sim/*.cpp
    src/sim/*.hpp
)

file(GLOB UI_SRC
    src/ui/*.cpp
    src/ui/*.hpp
)

file(GLOB IO_SRC
    src/io/*.cpp
    src/io/*.hpp
)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)

set(IMGUI_SRC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl2.cpp
)

add_executable(Schrodinger2D
    src/main.cpp
    ${SIM_SRC}
    ${IO_SRC}
)

target_include_directories(Schrodinger2D PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_definitions(Schrodinger2D PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Try to find GUI deps
set(HAVE_GUI OFF)
if(ENABLE_GUI)
    find_package(OpenGL QUIET)
    # Try config mode first (vcpkg/conan), then module mode
    find_package(glfw3 CONFIG QUIET)
    if(NOT glfw3_FOUND)
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(GLFW3 QUIET glfw3)
        endif()
    endif()

    if(OpenGL_FOUND AND (glfw3_FOUND OR GLFW3_FOUND))
        set(HAVE_GUI ON)
        message(STATUS "Building with GUI (GLFW + OpenGL)")
        target_sources(Schrodinger2D PRIVATE ${UI_SRC} ${IMGUI_SRC})
        if(TARGET glfw)
            target_link_libraries(Schrodinger2D PRIVATE glfw)
        elseif(glfw3_FOUND)
            target_link_libraries(Schrodinger2D PRIVATE glfw)
        elseif(GLFW3_FOUND)
            target_link_libraries(Schrodinger2D PRIVATE ${GLFW3_LIBRARIES})
            target_include_directories(Schrodinger2D PRIVATE ${GLFW3_INCLUDE_DIRS})
        endif()
        target_link_libraries(Schrodinger2D PRIVATE OpenGL::GL)
        target_compile_definitions(Schrodinger2D PRIVATE BUILD_GUI=1)
        if(WIN32)
            set_target_properties(Schrodinger2D PROPERTIES WIN32_EXECUTABLE TRUE)
        endif()
        if(MSVC)
            target_compile_options(Schrodinger2D PRIVATE /W4)
        else()
            target_compile_options(Schrodinger2D PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    else()
        message(WARNING "GLFW/OpenGL not found. Building headless mode only.")
        target_compile_definitions(Schrodinger2D PRIVATE BUILD_GUI=0)
    endif()
else()
    message(STATUS "ENABLE_GUI=OFF: building headless only")
    target_compile_definitions(Schrodinger2D PRIVATE BUILD_GUI=0)
endif()

# Platform specifics
if(WIN32)
    # On Windows, link against Winsock or other system libs if needed (not required here)
endif()

install(TARGETS Schrodinger2D RUNTIME DESTINATION bin)

