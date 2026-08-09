#include <iostream>
#include <iomanip>
#include "pso.hpp"
#include "benchmark.hpp"

using namespace optera::optimizers;

int main() {
    std::cout << "========================================================\n";
    std::cout << " OPTERA HIGH-PERFORMANCE C++ PSO OPTIMIZER & BENCHMARK  \n";
    std::cout << "========================================================\n\n";

    // CategoryParams: {category, mean_demand, std_dev, cv, unit_margin, risk_score}
    std::vector<CategoryParams> categories = {
        {"Clothing", 1514.31, 201.97, 0.133, 21.44, 0.16},
        {"Electronics", 996.49, 229.59, 0.230, 45.20, 0.23},
        {"Furniture", 1158.76, 309.49, 0.267, 35.10, 0.27},
        {"Groceries", 4114.91, 779.08, 0.189, 12.50, 0.19},
        {"Toys", 1098.26, 378.91, 0.345, 18.30, 0.35}
    };

    PSOConfig config;
    config.lambda_risk = 0.5;
    config.alpha_diversification = 0.10;

    PSOSolver solver(config);
    PSOResult result = solver.optimize(categories);

    std::cout << "Optimized Category Allocations (w*):\n";
    std::cout << "--------------------------------------------------------\n";
    for (size_t i = 0; i < categories.size(); ++i) {
        std::cout << "  • " << std::left << std::setw(15) << categories[i].category
                  << " : " << std::fixed << std::setprecision(2) << (result.gbest_weights[i] * 100.0) << "%\n";
    }
    std::cout << "--------------------------------------------------------\n";
    std::cout << "  • Expected Gross Profit       : $" << std::fixed << std::setprecision(2) << result.expected_profit << "\n";
    std::cout << "  • Volatility Risk Penalty (CV): $" << result.volatility_risk_penalty << "\n";
    std::cout << "  • Gamma Diversification Penalty: $" << result.diversification_penalty << "\n";
    std::cout << "  • Herfindahl Concentration HHI: " << result.herfindahl_index_hhi << "\n";
    std::cout << "  • Effective Number Categories ENC: " << std::setprecision(2) << result.effective_number_categories_enc << " categories\n";
    std::cout << "  • Maximum Portfolio Fitness F : $" << std::setprecision(2) << result.gbest_fitness << "\n";
    std::cout << "========================================================\n";

    return 0;
}
