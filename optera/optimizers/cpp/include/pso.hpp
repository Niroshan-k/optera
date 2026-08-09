#ifndef OPTERA_OPTIMIZERS_PSO_HPP
#define OPTERA_OPTIMIZERS_PSO_HPP

#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <utility>

namespace optera {
namespace optimizers {

/**
 * @brief Category parameters extracted from demand_model.json.
 */
struct CategoryParams {
    std::string category;
    double mean_demand; // \mu_i
    double std_dev;     // \sigma_i
    double cv;          // CV_i = \sigma_i / \mu_i
    double unit_margin; // p_i - c_i (unit_price - unit_cost)
    double risk_score;  // R_i (risk_profile.score)
};

/**
 * @brief Represents an individual particle in the swarm.
 */
struct Particle {
    std::vector<double> position; // Raw search space coordinates
    std::vector<double> weights;  // Normalized budget allocation weights (\sum w_i = 1.0)
    std::vector<double> velocity; // Particle velocity vector
    std::vector<double> pbest_position;
    std::vector<double> pbest_weights;
    double pbest_fitness;         // Personal best fitness score F(w)
    double current_fitness;       // Current fitness score F(w)
};

/**
 * @brief Configuration parameters for Markowitz-Herfindahl PSO.
 */
struct PSOConfig {
    int swarm_size = -1;                 // -1 triggers dynamic automatic heuristic calculation
    int max_iterations = -1;             // -1 triggers dynamic automatic heuristic calculation
    double w_start = 0.9;                // Initial inertia weight
    double w_end = 0.4;                  // Final inertia weight
    double c1 = 2.0;                     // Cognitive acceleration coefficient
    double c2 = 2.0;                     // Social acceleration coefficient
    double lambda_risk = 0.5;            // Risk aversion parameter \lambda
    double alpha_diversification = 0.10; // Herfindahl diversification factor \alpha \in [0.05, 0.20]
    double min_allocation = 0.05;        // Minimum category weight bound w_min (5%)
    double max_allocation = 0.40;        // Maximum category weight bound w_max (40%)
};

/**
 * @brief Optimization result container.
 */
struct PSOResult {
    std::vector<double> gbest_weights;      // Optimal category allocation vector w*
    double gbest_fitness;                   // Max fitness achieved F(w*)
    double expected_profit;                 // \sum w_i \mu_i (p_i - c_i)
    double portfolio_variance;              // w^T \Sigma w
    double portfolio_std_dev;               // \sqrt{w^T \Sigma w}
    double volatility_risk_penalty;         // \lambda * (w^T \Sigma w)
    double diversification_penalty;         // \gamma \sum w_i^2
    double herfindahl_index_hhi;            // HHI = \sum w_i^2
    double effective_number_categories_enc; // ENC = 1 / HHI
    double gamma_used;                      // \gamma = \alpha * average(expected_category_profit)
    int swarm_size_used;
    int iterations_completed;
    std::vector<double> convergence_history;
};

/**
 * @brief Evaluates the Markowitz-Herfindahl Portfolio Objective Function:
 * F(w) = \sum w_i * [\mu_i * (p_i - c_i)] - \lambda * (w^T \Sigma w) - \gamma * \sum w_i^2
 */
class SupplyChainFitness {
public:
    SupplyChainFitness(
        const std::vector<CategoryParams>& categories,
        const std::vector<std::vector<double>>& cov_matrix,
        double lambda_risk = 0.5,
        double alpha_diversification = 0.10,
        double min_alloc = 0.05,
        double max_alloc = 0.40
    );

    /**
     * @brief Normalizes a raw search vector x into a valid budget allocation vector w (\sum w_i = 1.0, w_min <= w_i <= w_max).
     */
    static std::vector<double> project_to_simplex(
        const std::vector<double>& x,
        double min_alloc = 0.05,
        double max_alloc = 0.40
    );

    /**
     * @brief Calculates \gamma = \alpha * average(expected_category_profit).
     */
    static double calculate_gamma(const std::vector<CategoryParams>& categories, double alpha);

    /**
     * @brief Evaluates fitness score F(w) for an allocation vector w.
     */
    double evaluate(const std::vector<double>& weights) const;

    /**
     * @brief Calculates gross expected profit for allocation vector w.
     */
    double calculate_expected_profit(const std::vector<double>& weights) const;

    /**
     * @brief Calculates quadratic portfolio variance w^T \Sigma w.
     */
    double calculate_portfolio_variance(const std::vector<double>& weights) const;

    /**
     * @brief Calculates Herfindahl-Hirschman Concentration Index HHI = \sum w_i^2.
     */
    double calculate_herfindahl_index(const std::vector<double>& weights) const;

    /**
     * @brief Calculates Effective Number of Categories ENC = 1 / HHI.
     */
    double calculate_enc(const std::vector<double>& weights) const;

    /**
     * @brief Calculates diversification penalty \gamma \sum w_i^2.
     */
    double calculate_diversification_penalty(const std::vector<double>& weights) const;

    double get_gamma() const { return gamma_; }

private:
    std::vector<CategoryParams> categories_;
    std::vector<std::vector<double>> cov_matrix_;
    double lambda_risk_;
    double alpha_diversification_;
    double gamma_;
    double min_alloc_;
    double max_alloc_;
};

/**
 * @brief High-performance Improved Particle Swarm Optimizer (IPSO).
 */
class PSOSolver {
public:
    PSOSolver(PSOConfig config = PSOConfig());

    /**
     * @brief Analyzes category dimension and determines optimal default swarm size & iteration count.
     */
    static std::pair<int, int> analyze_heuristics(size_t dimension);

    /**
     * @brief Executes PSO optimization over the given category demand model parameters & covariance matrix.
     */
    PSOResult optimize(
        const std::vector<CategoryParams>& categories,
        const std::vector<std::vector<double>>& cov_matrix = {}
    );

private:
    PSOConfig config_;
};

} // namespace optimizers
} // namespace optera

#endif // OPTERA_OPTIMIZERS_PSO_HPP
