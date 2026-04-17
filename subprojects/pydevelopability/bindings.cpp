/*
 * PyBind11 bindings for the "Developability of Triangle Meshes" C++ library.
 *
 * Exposes:
 *   - build_fixed_topology_problem(V, F)
 *   - evaluate_energy_and_grad(problem, V, energy_type)
 *   - run_optimization(problem, V, energy_type, linesearch, step_type, num_steps)
 *   - EnergyType, Linesearch, StepType enums
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <developableflow/fixed_topology_core.h>
#include <igl/upsample.h>

#include <tuple>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

using namespace Developables;

// Thin wrappers that accept Eigen refs (pybind11 maps numpy <-> Eigen automatically)

static FixedTopologyProblem py_build_problem(
    Eigen::Ref<const Eigen::MatrixXd> V,
    Eigen::Ref<const Eigen::MatrixXi> F)
{
    if (V.cols() != 3)
        throw std::invalid_argument("V must be Nx3");
    if (F.cols() != 3)
        throw std::invalid_argument("F must be Mx3");

    OMatrixXs Vs = V;
    OMatrixXi Fs = F;
    return build_fixed_topology_problem(Vs, Fs);
}

static std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
py_evaluate_energy_and_grad(
    const FixedTopologyProblem& problem,
    Eigen::Ref<const Eigen::MatrixXd> V,
    int energyType)
{
    if (energyType < 0 || energyType >= ENERGY_TYPE_NUMS)
        throw std::invalid_argument("Invalid energy type");

    OMatrixXs Vs = V;
    OVectorXs energy;
    OMatrixXs energyGrad;
    evaluate_energy_and_grad(problem, Vs, static_cast<EnergyType>(energyType), energy, energyGrad);

    Eigen::VectorXd e_out = energy.cast<double>();
    Eigen::MatrixXd g_out = energyGrad.cast<double>();
    return {e_out, g_out};
}

struct OptimizationResult {
    std::vector<Eigen::MatrixXd> V_per_subdiv;  // V at each subdivision level (0, 1, 2, ...)
    std::vector<Eigen::MatrixXi> F_per_subdiv;  // F at each subdivision level
    Eigen::MatrixXd V;  // Final V (same as V_per_subdiv.back())
    Eigen::MatrixXi F;  // Final F (same as F_per_subdiv.back())
    std::vector<double> energy_history;
    std::vector<int> step_statuses;
    std::vector<std::pair<int, int>> subdivision_stats;  // (num_verts, num_faces) after each subdiv
};

static OptimizationResult py_run_optimization(
    const FixedTopologyProblem& problem,
    Eigen::Ref<const Eigen::MatrixXd> V,
    int energyType,
    int linesearch,
    int stepType,
    int numSteps,
    int numSubdivisions = 0)
{
    if (energyType < 0 || energyType >= ENERGY_TYPE_NUMS)
        throw std::invalid_argument("Invalid energy type");
    if (linesearch < 0 || linesearch >= LINESEARCH_NUMS)
        throw std::invalid_argument("Invalid linesearch type");
    if (stepType < 0 || stepType >= STEP_TYPE_NUMS)
        throw std::invalid_argument("Invalid step type");
    if (numSteps < 0)
        throw std::invalid_argument("numSteps must be non-negative");
    if (numSubdivisions < 0)
        throw std::invalid_argument("numSubdivisions must be non-negative");

    FixedTopologyOptimizationConfig config;
    config.energyType = static_cast<EnergyType>(energyType);
    config.linesearch = static_cast<Linesearch>(linesearch);
    config.stepType = static_cast<StepType>(stepType);

    OMatrixXs Vs = V;
    OMatrixXi Fs = problem.F;  // Get faces from problem

    OptimizationResult out;
    out.energy_history.clear();
    out.step_statuses.clear();
    out.subdivision_stats.clear();
    out.V_per_subdiv.clear();
    out.F_per_subdiv.clear();

    // Main subdivision loop
    for (int subdiv = 0; subdiv <= numSubdivisions; ++subdiv) {
        if (subdiv > 0) {
            // Subdivide mesh using Loop subdivision
            OMatrixXs VUp;
            OMatrixXi FUp;
            igl::upsample(Vs, Fs, VUp, FUp);
            Vs = VUp;
            Fs = FUp;

            std::cout << "\n--- Subdivision " << subdiv << " ---" << std::endl;
            std::cout << "Mesh subdivided: " << Vs.rows() << " vertices, " << Fs.rows() << " faces" << std::endl;

            // Rebuild problem for new topology
            auto newProblem = build_fixed_topology_problem(Vs, Fs);

            // Run optimization on subdivided mesh
            auto result = run_unrolled_steps(newProblem, Vs, numSteps, config);

            // Update Vs with optimized vertices
            Vs = result.V;

            // Append energy history and step statuses
            for (double e : result.totalEnergyHistory) {
                out.energy_history.push_back(e);
            }
            for (int s : result.stepStatuses) {
                out.step_statuses.push_back(s);
            }
        } else {
            // Initial optimization (no subdivision)
            auto result = run_unrolled_steps(problem, Vs, numSteps, config);

            // Update Vs with optimized vertices
            Vs = result.V;

            out.energy_history = result.totalEnergyHistory;
            out.step_statuses = result.stepStatuses;
        }

        // Store optimized mesh at this subdivision level
        out.V_per_subdiv.push_back(Vs.cast<double>());
        out.F_per_subdiv.push_back(Fs.cast<int>());

        // Record mesh topology stats after this subdivision
        out.subdivision_stats.push_back({static_cast<int>(Vs.rows()), static_cast<int>(Fs.rows())});
    }

    out.V = Vs.cast<double>();
    out.F = Fs.cast<int>();
    return out;
}

// Convenience: energy-only (no grad) for profiling
static double py_total_energy(
    const FixedTopologyProblem& problem,
    Eigen::Ref<const Eigen::MatrixXd> V,
    int energyType)
{
    auto [energy, grad] = py_evaluate_energy_and_grad(problem, V, energyType);
    return energy.sum();
}


PYBIND11_MODULE(_devcore, m) {
    m.doc() = "PyBind11 bindings for Developability of Triangle Meshes";

    // Enums
    py::enum_<EnergyType>(m, "EnergyType")
        .value("HINGE", ENERGY_TYPE_HINGE)
        .value("MINWIDTH", ENERGY_TYPE_MINWIDTH)
        .value("OLDHINGE", ENERGY_TYPE_OLDHINGE)
        .value("OLDMINWIDTH", ENERGY_TYPE_OLDMINWIDTH)
        .value("PAIRWISENORMALS", ENERGY_TYPE_PAIRWISENORMALS)
        .value("MAXPAIRWISENORMALS", ENERGY_TYPE_MAXPAIRWISENORMALS)
        .export_values();

    py::enum_<Linesearch>(m, "Linesearch")
        .value("NONE", LINESEARCH_NONE)
        .value("OVERTON", LINESEARCH_OVERTON)
        .value("BACKTRACK", LINESEARCH_BACKTRACK)
        .export_values();

    py::enum_<StepType>(m, "StepType")
        .value("GRADDESC", STEP_TYPE_GRADDESC)
        .value("LBFGS", STEP_TYPE_LBFGS)
        .export_values();

    // FixedTopologyProblem (opaque handle)
    py::class_<FixedTopologyProblem>(m, "FixedTopologyProblem")
        .def_readonly("vertex_count", &FixedTopologyProblem::vertexCount)
        .def("__repr__", [](const FixedTopologyProblem& p) {
            return "<FixedTopologyProblem nv=" + std::to_string(p.vertexCount) +
                   " nf=" + std::to_string(p.F.rows()) + ">";
        });

    // OptimizationResult
    py::class_<OptimizationResult>(m, "OptimizationResult")
        .def_readonly("V", &OptimizationResult::V)
        .def_readonly("F", &OptimizationResult::F)
        .def_readonly("V_per_subdiv", &OptimizationResult::V_per_subdiv)
        .def_readonly("F_per_subdiv", &OptimizationResult::F_per_subdiv)
        .def_readonly("energy_history", &OptimizationResult::energy_history)
        .def_readonly("step_statuses", &OptimizationResult::step_statuses)
        .def_readonly("subdivision_stats", &OptimizationResult::subdivision_stats);

    // Functions
    m.def("build_problem", &py_build_problem,
          py::arg("V"), py::arg("F"),
          "Build a fixed-topology problem from vertex/face arrays.");

    m.def("evaluate_energy_and_grad", &py_evaluate_energy_and_grad,
          py::arg("problem"), py::arg("V"), py::arg("energy_type"),
          "Compute per-vertex energy and gradient. Returns (energy, grad).");

    m.def("total_energy", &py_total_energy,
          py::arg("problem"), py::arg("V"), py::arg("energy_type"),
          "Compute total (summed) energy scalar.");

    m.def("run_optimization", &py_run_optimization,
          py::arg("problem"), py::arg("V"),
          py::arg("energy_type"), py::arg("linesearch"), py::arg("step_type"),
          py::arg("num_steps"), py::arg("num_subdivisions") = 0,
          "Run N optimization steps over M subdivision iterations. Returns OptimizationResult with updated V, energy history, and subdivision_stats.");
}
