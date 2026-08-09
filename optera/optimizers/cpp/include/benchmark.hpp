#ifndef OPTERA_OPTIMIZERS_BENCHMARK_HPP
#define OPTERA_OPTIMIZERS_BENCHMARK_HPP

#include <vector>
#include <string>

namespace optera {
namespace optimizers {
namespace benchmark {

/**
 * @brief Sphere function (Convex baseline).
 */
double sphere(const std::vector<double>& x);

/**
 * @brief Ackley function (Multimodal baseline).
 */
double ackley(const std::vector<double>& x);

/**
 * @brief Rastrigin function (Highly non-convex benchmark with local minima traps).
 */
double rastrigin(const std::vector<double>& x);

/**
 * @brief Supply Chain Non-Convex Operational Cost Function.
 */
double supply_chain_cost(const std::vector<double>& x);

} // namespace benchmark
} // namespace optimizers
} // namespace optera

#endif // OPTERA_OPTIMIZERS_BENCHMARK_HPP
