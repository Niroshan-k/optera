#include "pso.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace optera {
namespace optimizers {

// -----------------------------------------------------------------------------
// SupplyChainFitness Implementation
// -----------------------------------------------------------------------------
SupplyChainFitness::SupplyChainFitness(
    const std::vector<CategoryParams>& categories,
    const std::vector<std::vector<double>>& cov_matrix,
    double lambda_risk,
    double alpha_diversification,
    double min_alloc,
    double max_alloc
)
    : categories_(categories),
      cov_matrix_(cov_matrix),
      lambda_risk_(lambda_risk),
      alpha_diversification_(alpha_diversification),
      min_alloc_(min_alloc),
      max_alloc_(max_alloc) {
    gamma_ = calculate_gamma(categories_, alpha_diversification_);
}

double SupplyChainFitness::calculate_gamma(const std::vector<CategoryParams>& categories, double alpha) {
    if (categories.empty()) return 1000.0;
    double sum_profit = 0.0;
    for (const auto& cat : categories) {
        sum_profit += (cat.mean_demand * cat.unit_margin);
    }
    double avg_profit = sum_profit / static_cast<double>(categories.size());
    return alpha * avg_profit;
}

std::vector<double> SupplyChainFitness::project_to_simplex(
    const std::vector<double>& x,
    double min_alloc,
    double max_alloc
) {
    size_t n = x.size();
    if (n == 0) return {};

    double effective_max = std::max(max_alloc, 1.0 / static_cast<double>(n));
    double effective_min = std::min(min_alloc, 1.0 / static_cast<double>(n));

    std::vector<double> w(n);
    for (size_t i = 0; i < n; ++i) {
        w[i] = std::max(effective_min, std::min(effective_max, x[i]));
    }

    for (int iter = 0; iter < 10; ++iter) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) sum += w[i];
        if (sum <= 0.0) break;

        for (size_t i = 0; i < n; ++i) {
            w[i] = std::max(effective_min, std::min(effective_max, w[i] / sum));
        }
    }

    double final_sum = 0.0;
    for (size_t i = 0; i < n; ++i) final_sum += w[i];
    if (final_sum > 0.0) {
        for (size_t i = 0; i < n; ++i) w[i] /= final_sum;
    }

    return w;
}

double SupplyChainFitness::evaluate(const std::vector<double>& weights) const {
    double profit = calculate_expected_profit(weights);
    double port_var = calculate_portfolio_variance(weights);
    double risk_pen = lambda_risk_ * port_var;
    double div_pen = calculate_diversification_penalty(weights);
    return profit - risk_pen - div_pen;
}

double SupplyChainFitness::calculate_expected_profit(const std::vector<double>& weights) const {
    double profit = 0.0;
    size_t n = std::min(weights.size(), categories_.size());
    for (size_t i = 0; i < n; ++i) {
        profit += weights[i] * categories_[i].mean_demand * categories_[i].unit_margin;
    }
    return profit;
}

double SupplyChainFitness::calculate_portfolio_variance(const std::vector<double>& weights) const {
    size_t n = std::min(weights.size(), categories_.size());
    double port_var = 0.0;

    if (!cov_matrix_.empty() && cov_matrix_.size() >= n) {
        // Full quadratic form w^T \Sigma w
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                port_var += weights[i] * weights[j] * cov_matrix_[i][j];
            }
        }
    } else {
        // Fallback to diagonal variance \sum w_i^2 \sigma_i^2
        for (size_t i = 0; i < n; ++i) {
            double var_i = categories_[i].std_dev * categories_[i].std_dev;
            port_var += weights[i] * weights[i] * var_i;
        }
    }
    return port_var;
}

double SupplyChainFitness::calculate_herfindahl_index(const std::vector<double>& weights) const {
    double hhi = 0.0;
    for (double w : weights) {
        hhi += w * w;
    }
    return hhi;
}

double SupplyChainFitness::calculate_enc(const std::vector<double>& weights) const {
    double hhi = calculate_herfindahl_index(weights);
    return (hhi > 0.0) ? (1.0 / hhi) : 0.0;
}

double SupplyChainFitness::calculate_diversification_penalty(const std::vector<double>& weights) const {
    return gamma_ * calculate_herfindahl_index(weights);
}

// -----------------------------------------------------------------------------
// PSOSolver Implementation
// -----------------------------------------------------------------------------
PSOSolver::PSOSolver(PSOConfig config)
    : config_(config) {}

std::pair<int, int> PSOSolver::analyze_heuristics(size_t dimension) {
    int swarm_size;
    int max_iterations;

    if (dimension <= 5) {
        swarm_size = 50;
        max_iterations = 150;
    } else if (dimension <= 20) {
        swarm_size = static_cast<int>(dimension * 12);
        max_iterations = 250;
    } else {
        swarm_size = std::min(200, static_cast<int>(dimension * 15));
        max_iterations = 400;
    }

    return std::make_pair(swarm_size, max_iterations);
}

PSOResult PSOSolver::optimize(
    const std::vector<CategoryParams>& categories,
    const std::vector<std::vector<double>>& cov_matrix
) {
    PSOResult result;
    size_t dim = categories.size();
    if (dim == 0) return result;

    auto [auto_swarm, auto_iter] = analyze_heuristics(dim);
    int swarm_size = (config_.swarm_size > 0) ? config_.swarm_size : auto_swarm;
    int max_iterations = (config_.max_iterations > 0) ? config_.max_iterations : auto_iter;

    result.swarm_size_used = swarm_size;
    result.iterations_completed = max_iterations;

    SupplyChainFitness fitness_eval(
        categories,
        cov_matrix,
        config_.lambda_risk,
        config_.alpha_diversification,
        config_.min_allocation,
        config_.max_allocation
    );

    result.gamma_used = fitness_eval.get_gamma();

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<double> pos_dist(0.05, 0.40);
    std::uniform_real_distribution<double> vel_dist(-0.05, 0.05);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    std::vector<Particle> swarm(swarm_size);
    std::vector<double> gbest_position(dim);
    std::vector<double> gbest_weights(dim);
    double gbest_fitness = -1e30;

    // Swarm Initialization
    for (int i = 0; i < swarm_size; ++i) {
        swarm[i].position.resize(dim);
        swarm[i].velocity.resize(dim);
        swarm[i].pbest_position.resize(dim);
        swarm[i].pbest_weights.resize(dim);

        for (size_t d = 0; d < dim; ++d) {
            swarm[i].position[d] = pos_dist(rng);
            swarm[i].velocity[d] = vel_dist(rng);
        }

        swarm[i].weights = SupplyChainFitness::project_to_simplex(
            swarm[i].position,
            config_.min_allocation,
            config_.max_allocation
        );
        swarm[i].current_fitness = fitness_eval.evaluate(swarm[i].weights);

        swarm[i].pbest_position = swarm[i].position;
        swarm[i].pbest_weights = swarm[i].weights;
        swarm[i].pbest_fitness = swarm[i].current_fitness;

        if (swarm[i].current_fitness > gbest_fitness) {
            gbest_fitness = swarm[i].current_fitness;
            gbest_position = swarm[i].position;
            gbest_weights = swarm[i].weights;
        }
    }

    result.convergence_history.reserve(max_iterations);

    // Swarm Iteration Loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        double w_curr = config_.w_start - (static_cast<double>(iter) / max_iterations) * (config_.w_start - config_.w_end);

        for (int i = 0; i < swarm_size; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                double r1 = u01(rng);
                double r2 = u01(rng);

                double cognitive = config_.c1 * r1 * (swarm[i].pbest_position[d] - swarm[i].position[d]);
                double social = config_.c2 * r2 * (gbest_position[d] - swarm[i].position[d]);
                swarm[i].velocity[d] = w_curr * swarm[i].velocity[d] + cognitive + social;
                swarm[i].velocity[d] = std::max(-0.1, std::min(0.1, swarm[i].velocity[d]));

                swarm[i].position[d] += swarm[i].velocity[d];
            }

            swarm[i].weights = SupplyChainFitness::project_to_simplex(
                swarm[i].position,
                config_.min_allocation,
                config_.max_allocation
            );
            swarm[i].current_fitness = fitness_eval.evaluate(swarm[i].weights);

            if (swarm[i].current_fitness > swarm[i].pbest_fitness) {
                swarm[i].pbest_fitness = swarm[i].current_fitness;
                swarm[i].pbest_position = swarm[i].position;
                swarm[i].pbest_weights = swarm[i].weights;

                if (swarm[i].current_fitness > gbest_fitness) {
                    gbest_fitness = swarm[i].current_fitness;
                    gbest_position = swarm[i].position;
                    gbest_weights = swarm[i].weights;
                }
            }
        }

        result.convergence_history.push_back(gbest_fitness);
    }

    result.gbest_weights = gbest_weights;
    result.gbest_fitness = gbest_fitness;
    result.expected_profit = fitness_eval.calculate_expected_profit(gbest_weights);
    result.portfolio_variance = fitness_eval.calculate_portfolio_variance(gbest_weights);
    result.portfolio_std_dev = std::sqrt(result.portfolio_variance);
    result.volatility_risk_penalty = config_.lambda_risk * result.portfolio_variance;
    result.diversification_penalty = fitness_eval.calculate_diversification_penalty(gbest_weights);
    result.herfindahl_index_hhi = fitness_eval.calculate_herfindahl_index(gbest_weights);
    result.effective_number_categories_enc = fitness_eval.calculate_enc(gbest_weights);

    return result;
}

} // namespace optimizers
} // namespace optera
