"""
Optera Optimization Evaluator Module (v1.8)

Provides rigorous evaluation of the PSO algorithm:
• Monte Carlo Stability Analysis (M independent runs, mean, std, 95% CI, stability CV)
• Markowitz Efficient Frontier Generation (Risk vs Profit tradeoff sweep)
• Procurement Efficiency Score (PES = Profit / Portfolio_Std_Dev)
• Convergence Speed Analysis (iteration at 99% max fitness)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from optera.optimizers.pso_wrapper import OpteraPSO

logger = logging.getLogger("OpteraOptimizerEvaluator")


class OptimizerEvaluator:
    """
    Evaluator engine for Particle Swarm Optimization (PSO / IPSO).
    """

    def __init__(
        self,
        demand_model_path: Union[str, Path],
        lambda_risk: float = 0.0001,
        alpha_diversification: float = 0.10,
        min_allocation: float = 0.05,
        max_allocation: float = 0.40
    ):
        self.demand_model_path = Path(demand_model_path)
        self.lambda_risk = lambda_risk
        self.alpha_diversification = alpha_diversification
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

    def run_stability_analysis(self, num_runs: int = 30) -> Dict[str, Any]:
        """
        Executes N independent PSO optimization trials to evaluate stochastic algorithm
        stability, best fitness, mean fitness, std dev, 95% confidence interval, and execution time.
        """
        import time
        logger.info("Executing PSO Algorithm Stability & Robustness Analysis (%d runs)...", num_runs)
        t_start = time.time()

        fitnesses = []
        profits = []
        variances = []
        hhis = []

        for r in range(num_runs):
            # Vary swarm size slightly [35, 65] to test algorithm stability under different initializations
            s_size = 35 + (r % 30)
            pso = OpteraPSO(
                swarm_size=s_size,
                max_iterations=120 + (r % 50),
                lambda_risk=self.lambda_risk,
                alpha_diversification=self.alpha_diversification,
                min_allocation=self.min_allocation,
                max_allocation=self.max_allocation
            )
            res = pso.optimize_from_model(self.demand_model_path)
            fitnesses.append(res["optimal_fitness"])
            profits.append(res["expected_gross_profit"])
            variances.append(res["portfolio_variance"])
            hhis.append(res["herfindahl_index_hhi"])

        total_runtime = time.time() - t_start
        fit_arr = np.array(fitnesses, dtype=float)
        best_fit = float(np.max(fit_arr))
        worst_fit = float(np.min(fit_arr))
        mean_fit = float(np.mean(fit_arr))
        std_fit = float(np.std(fit_arr, ddof=1)) if num_runs > 1 else 0.0
        stability_cv = float(std_fit / mean_fit) if mean_fit > 0 else 0.0

        # 95% Confidence Interval: mean +- 1.96 * (std / sqrt(num_runs))
        margin_error = float(1.96 * (std_fit / np.sqrt(num_runs))) if num_runs > 0 else 0.0
        ci_lower = round(mean_fit - margin_error, 2)
        ci_upper = round(mean_fit + margin_error, 2)

        return {
            "num_runs": num_runs,
            "best_fitness": round(best_fit, 2),
            "worst_fitness": round(worst_fit, 2),
            "mean_fitness": round(mean_fit, 2),
            "std_fitness": round(std_fit, 4),
            "stability_cv": round(stability_cv, 6),
            "fitness_confidence_interval": [ci_lower, ci_upper],
            "execution_time_seconds": round(total_runtime, 3),
            "all_fitnesses": [round(f, 2) for f in fitnesses],
            "all_profits": [round(p, 2) for p in profits],
            "all_variances": [round(v, 2) for v in variances]
        }

    def generate_efficient_frontier(self, num_points: int = 25) -> Dict[str, Any]:
        """
        Sweeps lambda_risk parameters to compute the Markowitz Efficient Frontier curve
        (Risk std dev vs Expected Profit).
        """
        logger.info("Generating Markowitz Efficient Frontier (%d parameter evaluation points)...", num_points)
        lambdas = np.logspace(-6, -2, num=num_points)

        frontier_points = []
        for l_val in lambdas:
            pso = OpteraPSO(
                lambda_risk=float(l_val),
                alpha_diversification=self.alpha_diversification,
                min_allocation=self.min_allocation,
                max_allocation=self.max_allocation
            )
            res = pso.optimize_from_model(self.demand_model_path)
            frontier_points.append({
                "lambda": float(l_val),
                "profit": res["expected_gross_profit"],
                "portfolio_std_dev": res["portfolio_std_dev"],
                "portfolio_variance": res["portfolio_variance"],
                "fitness": res["optimal_fitness"],
                "pes": float(res["expected_gross_profit"] / res["portfolio_std_dev"]) if res["portfolio_std_dev"] > 0 else 0.0,
                "weights": res["allocations_percentage"]
            })

        # Sort by std_dev ascending for smooth plotting
        frontier_points.sort(key=lambda p: p["portfolio_std_dev"])

        return {
            "num_points": num_points,
            "points": frontier_points,
            "risks_std": [p["portfolio_std_dev"] for p in frontier_points],
            "profits": [p["profit"] for p in frontier_points]
        }

    @staticmethod
    def calculate_convergence_speed(convergence_history: List[float], threshold: float = 0.99) -> int:
        """
        Finds the first iteration index where fitness achieves threshold * max_fitness.
        """
        if not convergence_history:
            return 0
        max_val = max(convergence_history)
        target = threshold * max_val

        for idx, val in enumerate(convergence_history):
            if val >= target:
                return idx + 1  # 1-indexed iteration count

        return len(convergence_history)

    @staticmethod
    def calculate_pes(expected_profit: float, portfolio_std_dev: float) -> float:
        """
        Calculates Procurement Efficiency Score (PES = Profit / Portfolio_Std_Dev).
        """
        if portfolio_std_dev > 0:
            return round(float(expected_profit / portfolio_std_dev), 4)
        return 0.0
