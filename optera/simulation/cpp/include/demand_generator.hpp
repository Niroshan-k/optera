/**
 * Optera Demand Generator Module - Strategy Pattern (v2.0)
 *
 * Provides thread-safe stochastic demand sampling from Normal, Poisson,
 * Gamma, and Negative Binomial distributions based on demand_model.json parameters.
 */

#ifndef OPTERA_DEMAND_GENERATOR_HPP
#define OPTERA_DEMAND_GENERATOR_HPP

#include "types.hpp"
#include <random>
#include <memory>

namespace optera {

class IDemandGenerator {
public:
    virtual ~IDemandGenerator() = default;
    virtual double sample(std::mt19937& rng) = 0;
};

class NormalDemandGenerator : public IDemandGenerator {
private:
    std::normal_distribution<double> dist_;
public:
    NormalDemandGenerator(double mean, double std_dev) : dist_(mean, std_dev) {}
    double sample(std::mt19937& rng) override {
        double val = dist_(rng);
        return std::max(0.0, val);
    }
};

class PoissonDemandGenerator : public IDemandGenerator {
private:
    std::poisson_distribution<int> dist_;
public:
    PoissonDemandGenerator(double mean) : dist_(std::max(0.001, mean)) {}
    double sample(std::mt19937& rng) override {
        return static_cast<double>(dist_(rng));
    }
};

class GammaDemandGenerator : public IDemandGenerator {
private:
    std::gamma_distribution<double> dist_;
public:
    GammaDemandGenerator(double alpha, double beta) : dist_(std::max(0.001, alpha), std::max(0.001, beta)) {}
    double sample(std::mt19937& rng) override {
        return std::max(0.0, dist_(rng));
    }
};

class NegativeBinomialDemandGenerator : public IDemandGenerator {
private:
    std::negative_binomial_distribution<int> dist_;
public:
    NegativeBinomialDemandGenerator(double k, double p) 
        : dist_(std::max(1, static_cast<int>(k)), std::min(0.999, std::max(0.001, p))) {}
    double sample(std::mt19937& rng) override {
        return static_cast<double>(dist_(rng));
    }
};

class DemandGeneratorFactory {
public:
    static std::unique_ptr<IDemandGenerator> create(const CategoryConfig& config) {
        switch (config.dist_type) {
            case DistributionType::POISSON:
                return std::make_unique<PoissonDemandGenerator>(config.mean_demand);
            case DistributionType::GAMMA:
                return std::make_unique<GammaDemandGenerator>(config.param_alpha, config.param_beta);
            case DistributionType::NEGATIVE_BINOMIAL:
                return std::make_unique<NegativeBinomialDemandGenerator>(config.param_alpha, config.param_beta);
            case DistributionType::NORMAL:
            default:
                return std::make_unique<NormalDemandGenerator>(config.mean_demand, config.std_demand);
        }
    }
};

} // namespace optera

#endif // OPTERA_DEMAND_GENERATOR_HPP
