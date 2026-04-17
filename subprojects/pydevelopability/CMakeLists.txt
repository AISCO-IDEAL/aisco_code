cmake_minimum_required(VERSION 3.16)
project(pydevelopability LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── pybind11 ──────────────────────────────────────────────────────────────
find_package(pybind11 REQUIRED)

# ── Paths to C++ dependencies (shared from concentrate_curvature) ────────
set(CC_ROOT "/cluster/work/fuge/AISCO/concentrate_curvature"
    CACHE PATH "Root of concentrate_curvature repo")
set(DEV_REPO "${CC_ROOT}/DevelopabilityOfTriangleMeshes"
    CACHE PATH "Path to DevelopabilityOfTriangleMeshes")

# ── Eigen3 (use system 3.4.0 from spack) ─────────────────────────────────
set(EIGEN3_SPACK "/cluster/software/stacks/2025-06/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/eigen-3.4.0-lh4h7c3wed6mhhlsurszucjzd22gmo4y")
find_package(Eigen3 CONFIG QUIET HINTS "${EIGEN3_SPACK}/share/eigen3/cmake")
if(NOT TARGET Eigen3::Eigen)
    # Manual fallback
    add_library(Eigen3_Eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
    target_include_directories(Eigen3_Eigen SYSTEM INTERFACE "${EIGEN3_SPACK}/include/eigen3")
endif()

# ── libigl (header-only core) ─────────────────────────────────────────────
set(LIBIGL_INCLUDE "${CC_ROOT}/libigl-modern/include"
    CACHE PATH "Path to libigl include directory")
if(NOT EXISTS "${LIBIGL_INCLUDE}/igl/is_border_vertex.h")
    message(FATAL_ERROR "libigl not found at ${LIBIGL_INCLUDE}")
endif()

# ── Developableflow core sources ──────────────────────────────────────────
set(DEVFLOW_INCLUDE "${DEV_REPO}/include")

add_library(devflow_core OBJECT
    "${DEVFLOW_INCLUDE}/developableflow/fixed_topology_core.cpp"
)
target_include_directories(devflow_core PUBLIC
    "${DEVFLOW_INCLUDE}"
    "${LIBIGL_INCLUDE}"
)
target_link_libraries(devflow_core PUBLIC Eigen3::Eigen)

# ── PyBind11 module ───────────────────────────────────────────────────────
pybind11_add_module(_devcore bindings.cpp)
target_include_directories(_devcore PRIVATE
    "${DEVFLOW_INCLUDE}"
    "${LIBIGL_INCLUDE}"
)
target_link_libraries(_devcore PRIVATE devflow_core Eigen3::Eigen)

# Install into the package directory
install(TARGETS _devcore LIBRARY DESTINATION pydevelopability)
