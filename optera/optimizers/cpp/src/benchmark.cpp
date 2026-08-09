#include "benchmark.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace optera {
namespace optimizers {
namespace benchmark {

double sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
}

double ackley(const std::vector<double>& x) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    size_t n = x.size();
    if (n == 0) return 0.0;
    for (double val : x) {
        sum1 += val * val;
        sum2 += std::cos(2.0 * M_PI * val);
    }
    return -20.0 * std::exp(-0.2 * std::sqrt(sum1 / n)) - std::exp(sum2 / n) + 20.0 + M_E;
}

double rastrigin(const std::vector<double>& x) {
    double sum = 10.0 * x.size();
    for (double val : x) {
        sum += (val * val - 10.0 * std::cos(2.0 * M_PI * val));
    }
    return sum;
}

double supply_chain_cost(const std::vector<double>& x) {
    // Non-convex supply chain operational cost surface benchmark
    if (x.size() < 2) return 0.0;
    double S = x[0];
    double Q = x[1];
    
    double base_cost = 0.05 * S + 250.0 / (Q + 1e-5) + 0.01 * (Q * Q);
    double nonconvexity = 0.20 * base_cost * std.sin(4.5 * M_PI * S / 100.0) * std.cos(4.5 * M_PI * Q / 100.0);
    return base_cost + nonconvexity;
}

} // namespace benchmark
} // namespace optimizers
} // namespace optera
