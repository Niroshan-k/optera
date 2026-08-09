"""
Optera Particle Swarm Optimization (PSO) Module

Exports the production-grade Markowitz-Herfindahl Inertia-Weighted PSO Portfolio Optimization Engine.
"""

from optera.optimizers.pso_wrapper import OpteraPSO, PSOPortfolioOptimizer, optimize_portfolio

__all__ = ["OpteraPSO", "PSOPortfolioOptimizer", "optimize_portfolio", "optimize"]


def optimize(*args, **kwargs):
    """
    Alias function delegating to the production Optera PSO portfolio optimization engine.
    """
    return optimize_portfolio(*args, **kwargs)
