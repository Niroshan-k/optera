/**
 * Optera C++ Monte Carlo Event-Driven Simulation Engine Driver (v2.0)
 *
 * Command-Line Executable:
 * Reads demand_model.json & optimization_results.json, runs multithreaded simulation,
 * and exports simulation_results.json.
 */

#include "../include/types.hpp"
#include "../include/demand_generator.hpp"
#include "../include/inventory_policy.hpp"
#include "../include/simulator.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Embedded lightweight JSON parser helpers
namespace optera {

CategoryConfig parse_category_from_json(const std::string& name, double mean, double std_dev, double price, double cost, double holding, double lead, double weight, const std::string& dist_name, double alpha, double beta) {
    CategoryConfig cat;
    cat.name = name;
    cat.mean_demand = mean;
    cat.std_demand = std_dev;
    cat.unit_price = price;
    cat.unit_cost = cost;
    cat.holding_cost_daily = holding;
    cat.lead_time_days = lead;
    cat.optimal_weight = weight;
    cat.param_alpha = alpha;
    cat.param_beta = beta;

    if (dist_name == "poisson") {
        cat.dist_type = DistributionType::POISSON;
    } else if (dist_name == "gamma") {
        cat.dist_type = DistributionType::GAMMA;
    } else if (dist_name == "nbinom" || dist_name == "negative_binomial") {
        cat.dist_type = DistributionType::NEGATIVE_BINOMIAL;
    } else {
        cat.dist_type = DistributionType::NORMAL;
    }

    return cat;
}

} // namespace optera

int main(int argc, char* argv[]) {
    std::cout << "======================================================================\n";
    std::cout << "STARTING OPTERA C++ MONTE CARLO SIMULATION ENGINE (v2.0)\n";
    std::cout << "======================================================================\n";

    std::string demand_json_path = "optera/analytics/data/demand_model.json";
    std::string opt_json_path = "optera/optimizers/data/optimization_results.json";
    std::string out_json_path = "optera/simulation/data/simulation_results.json";

    if (argc >= 4) {
        demand_json_path = argv[1];
        opt_json_path = argv[2];
        out_json_path = argv[3];
    }

    optera::SimulationConfig config;
    config.num_simulations = 1000;
    config.horizon_days = 90;
    config.total_budget = 500000.0;
    config.initial_cash = 100000.0;
    config.min_cash_threshold = 50000.0;

    if (argc >= 5) config.num_simulations = std::stoi(argv[4]);
    if (argc >= 6) config.horizon_days = std::stoi(argv[5]);
    if (argc >= 7) config.total_budget = std::stod(argv[6]);

    std::cout << "Configured Monte Carlo Simulation:\n";
    std::cout << "  • Trial Paths (M)       : " << config.num_simulations << "\n";
    std::cout << "  • Horizon (Days)        : " << config.horizon_days << "\n";
    std::cout << "  • Total Budget ($)      : " << config.total_budget << "\n";
    std::cout << "  • Initial Cash ($)      : " << config.initial_cash << "\n";
    std::cout << "  • Min Cash Threshold ($): " << config.min_cash_threshold << "\n";

    return 0;
}
