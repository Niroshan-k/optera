"""
Optera PyBind11 C++ to Python Binding Module
"""

/* 
 * PyBind11 C++ Binding Interface for Optera PSO Engine
 * 
 * To compile this native CPython module using pybind11:
 * c++ -O3 -wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
 *     pso.cpp pybind11_bindings.cpp -o optera_pso_cpp$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/pso.hpp"

namespace py = pybind11;
using namespace optera::optimizers;

PYBIND11_MODULE(optera_pso_cpp, m) {
    m.doc() = "Optera C++ Particle Swarm Optimization (IPSO) Engine PyBind11 Extension Module";

    py::class_<CategoryParams>(m, "CategoryParams")
        .def(py::init<>())
        .def_readwrite("category", &CategoryParams::category)
        .def_readwrite("mean_demand", &CategoryParams::mean_demand)
        .def_readwrite("std_dev", &CategoryParams::std_dev)
        .def_readwrite("cv", &CategoryParams::cv)
        .def_readwrite("unit_margin", &CategoryParams::unit_margin)
        .def_readwrite("risk_score", &CategoryParams::risk_score);

    py::class_<PSOConfig>(m, "PSOConfig")
        .def(py::init<>())
        .def_readwrite("swarm_size", &PSOConfig::swarm_size)
        .def_readwrite("max_iterations", &PSOConfig::max_iterations)
        .def_readwrite("lambda_risk", &PSOConfig::lambda_risk)
        .def_readwrite("alpha_diversification", &PSOConfig::alpha_diversification)
        .def_readwrite("min_alloc", &PSOConfig::min_alloc)
        .def_readwrite("max_alloc", &PSOConfig::max_alloc)
        .def_readwrite("w_start", &PSOConfig::w_start)
        .def_readwrite("w_end", &PSOConfig::w_end)
        .def_readwrite("c1", &PSOConfig::c1)
        .def_readwrite("c2", &PSOConfig::c2);

    py::class_<PSOResult>(m, "PSOResult")
        .def_readwrite("best_weights", &PSOResult::best_weights)
        .def_readwrite("best_fitness", &PSOResult::best_fitness)
        .def_readwrite("expected_gross_profit", &PSOResult::expected_gross_profit)
        .def_readwrite("portfolio_demand_variance", &PSOResult::portfolio_demand_variance)
        .def_readwrite("portfolio_demand_std_dev", &PSOResult::portfolio_demand_std_dev)
        .def_readwrite("volatility_risk_penalty", &PSOResult::volatility_risk_penalty)
        .def_readwrite("diversification_penalty", &PSOResult::diversification_penalty)
        .def_readwrite("hhi_concentration", &PSOResult::hhi_concentration)
        .def_readwrite("effective_num_categories", &PSOResult::effective_num_categories)
        .def_readwrite("pes_score", &PSOResult::pes_score)
        .def_readwrite("iterations_run", &PSOResult::iterations_run)
        .def_readwrite("convergence_iteration", &PSOResult::convergence_iteration);

    py::class_<OpteraPSO>(m, "OpteraPSO")
        .def(py::init<const PSOConfig&>(), py::arg("config") = PSOConfig())
        .def("optimize", &OpteraPSO::optimize, py::arg("categories"), py::arg("cov_matrix") = std::vector<std::vector<double>>())
        .def_static("analyze_heuristics", &OpteraPSO::analyze_heuristics, py::arg("dimension"));
}
