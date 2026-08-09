"""
Optera Particle Swarm Optimization (PSO / IPSO) Engine

Implements Markowitz-Herfindahl risk-adjusted supply chain portfolio optimization:
F(w) = \sum w_i * [\mu_i * (p_i - c_i)] - \lambda * (w^T \Sigma w) - \gamma * \sum w_i^2

Mathematical Formulation:
1. Expected Gross Profit P = \sum w_i * \mu_i * (p_i - c_i)
2. Portfolio Demand Variance = w^T \Sigma w (where \Sigma is the N x N Category Demand Covariance Matrix)
3. Portfolio Volatility Risk Penalty V = \lambda * (w^T \Sigma w)
4. Concentration Penalty D = \gamma * \sum w_i^2 (where \gamma = \alpha * average(expected_category_profit))
5. Herfindahl Concentration Index HHI = \sum w_i^2
6. Effective Number of Categories ENC = 1 / HHI
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("OpteraPSOWrapper")


class OpteraPSO:
    """
    High-Performance Particle Swarm Optimization (PSO / IPSO) Engine.
    """

    def __init__(
        self,
        swarm_size: Optional[int] = None,
        max_iterations: Optional[int] = None,
        lambda_risk: float = 0.5,
        alpha_diversification: float = 0.10,
        min_allocation: float = 0.05,
        max_allocation: float = 0.40,
        shortage_penalty_weight: float = 1.0,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0
    ):
        """
        Parameters
        ----------
        swarm_size : Optional[int]
            Swarm particle count. If None, auto-calculated from dimension.
        max_iterations : Optional[int]
            Max iteration count. If None, auto-calculated from dimension.
        lambda_risk : float
            Risk aversion parameter \lambda for portfolio variance penalty.
        alpha_diversification : float
            Herfindahl diversification multiplier \alpha \in [0.05, 0.20] (default: 0.10).
        min_allocation : float
            Minimum category budget allocation weight (default: 0.05 / 5%).
        max_allocation : float
            Maximum category budget allocation weight (default: 0.40 / 40%).
        """
        self.user_swarm_size = swarm_size
        self.user_max_iterations = max_iterations
        self.lambda_risk = lambda_risk
        self.alpha_diversification = alpha_diversification
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2

    @staticmethod
    def analyze_heuristics(dimension: int) -> Tuple[int, int]:
        """
        Analyzes decision space dimension N to compute optimal default
        swarm size and max iterations.
        """
        if dimension <= 5:
            swarm_size = 50
            max_iterations = 150
        elif dimension <= 20:
            swarm_size = dimension * 12
            max_iterations = 250
        else:
            swarm_size = min(200, dimension * 15)
            max_iterations = 400

        return swarm_size, max_iterations

    @staticmethod
    def calculate_gamma(means: np.ndarray, margins: np.ndarray, alpha: float) -> float:
        """
        Calculates \gamma = \alpha * average(expected_category_profit).
        """
        category_profits = means * margins
        avg_profit = float(np.mean(category_profits)) if len(category_profits) > 0 else 1000.0
        return round(alpha * avg_profit, 2)

    @staticmethod
    def project_to_simplex(x: np.ndarray, min_alloc: float = 0.05, max_alloc: float = 0.40) -> np.ndarray:
        """
        Projects raw continuous vector x onto box-constrained budget simplex:
        w_min <= w_i <= w_max, \sum w_i = 1.0.
        """
        n = len(x)
        if n == 0:
            return x

        eff_max = max(max_alloc, 1.0 / n)
        eff_min = min(min_alloc, 1.0 / n)

        w = np.clip(x, eff_min, eff_max)

        # Iterative clamping projection
        for _ in range(10):
            s = np.sum(w)
            if s <= 0:
                break
            w = np.clip(w / s, eff_min, eff_max)

        final_s = np.sum(w)
        if final_s > 0:
            w = w / final_s

        return w

    def evaluate_fitness(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        margins: np.ndarray,
        cov_matrix: np.ndarray,
        gamma: float,
        costs: Optional[np.ndarray] = None,
        reorder_points: Optional[np.ndarray] = None,
        total_budget: float = 500000.0,
        psi: float = 0.15
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Calculates:
        1. Expected Profit P = \sum w_i * \mu_i * (p_i - c_i)
        2. Portfolio Variance = w^T \Sigma w
        3. Portfolio Volatility Risk Penalty V = \lambda * (w^T \Sigma w)
        4. Herfindahl Index HHI = \sum w_i^2
        5. Effective Number of Categories ENC = 1 / HHI
        6. Diversification Penalty D = \gamma * HHI
        7. Dimensionless Proportional Shortage Penalty S = \psi * mean(P) * \sum max(0, (s_i - I_{0,i}) / s_i)
        8. Total Portfolio Fitness F(w) = P - V - D - S
        """
        expected_profit = float(np.sum(weights * means * margins))
        
        # Calculate quadratic form w^T \Sigma w
        if cov_matrix.size > 0 and cov_matrix.shape == (len(weights), len(weights)):
            port_var = float(weights.T @ cov_matrix @ weights)
        else:
            port_var = float(np.sum((weights ** 2) * (means ** 2)))

        volatility_risk_penalty = float(self.lambda_risk * port_var)
        hhi = float(np.sum(weights ** 2))
        enc = float(1.0 / hhi) if hhi > 0 else 0.0
        diversification_penalty = float(gamma * hhi)

        # Dimensionless Proportional Shortage Penalty
        shortage_penalty = 0.0
        if costs is not None and reorder_points is not None:
            shortage_ratios = []
            for i in range(len(weights)):
                budget_units = np.floor((weights[i] * total_budget) / max(1.0, costs[i]))
                s_i = max(1.0, reorder_points[i])
                ratio = max(0.0, (s_i - budget_units) / s_i)
                shortage_ratios.append(ratio)

            mean_cat_profit = float(np.mean(means * margins))
            shortage_penalty = float(psi * mean_cat_profit * np.sum(shortage_ratios))

        fitness = expected_profit - volatility_risk_penalty - diversification_penalty - shortage_penalty

        return fitness, expected_profit, port_var, volatility_risk_penalty, hhi, enc, diversification_penalty

    def optimize_from_model(
        self,
        demand_model: Union[str, Path, Dict[str, Any]],
        total_budget: float = 500000.0
    ) -> Dict[str, Any]:
        """
        Executes Markowitz-Herfindahl PSO optimization over demand_model.json.
        """
        if isinstance(demand_model, (str, Path)):
            with open(Path(demand_model), "r", encoding="utf-8") as f:
                model_dict = json.load(f)
        elif isinstance(demand_model, dict):
            model_dict = demand_model
        else:
            raise ValueError("Input must be a JSON file path or dictionary.")

        # Extract metadata (covariance matrix if available)
        metadata = model_dict.get("metadata", {})
        cov_json = metadata.get("covariance_matrix", {})

        # Extract categories (skip metadata key)
        categories = [k for k in model_dict.keys() if k != "metadata"]
        dim = len(categories)

        if dim == 0:
            raise ValueError("Empty demand model input.")

        # Extract vectors
        means = np.array([model_dict[cat]["mean"] for cat in categories], dtype=float)
        margins = np.array([
            model_dict[cat]["financials"]["unit_price_mean"] - model_dict[cat]["financials"]["unit_cost_mean"]
            for cat in categories
        ], dtype=float)
        std_devs = np.array([model_dict[cat]["std"] for cat in categories], dtype=float)

        # Build Covariance Matrix \Sigma
        cov_matrix = np.zeros((dim, dim), dtype=float)
        if cov_json:
            for i, c1 in enumerate(categories):
                for j, c2 in enumerate(categories):
                    cov_matrix[i, j] = cov_json.get(c1, {}).get(c2, 0.0)
        else:
            # Diagonal fallback if cov_json missing
            np.fill_diagonal(cov_matrix, std_devs ** 2)

        # Dynamic Heuristic & Parameter Analysis
        auto_swarm, auto_iter = self.analyze_heuristics(dim)
        swarm_size = self.user_swarm_size if self.user_swarm_size is not None else auto_swarm
        max_iterations = self.user_max_iterations if self.user_max_iterations is not None else auto_iter
        gamma = self.calculate_gamma(means, margins, self.alpha_diversification)

        logger.info(
            "Running Optera Markowitz-Herfindahl PSO Optimization over %d categories (Swarm: %d, Iterations: %d, \u03bb: %.4f, \u03b1: %.2f, \u03b3: %.2f)...",
            dim, swarm_size, max_iterations, self.lambda_risk, self.alpha_diversification, gamma
        )

        costs = np.array([model_dict[cat]["financials"]["unit_cost_mean"] for cat in categories], dtype=float)
        lead_times = np.array([max(1.0, float(model_dict[cat]["financials"].get("lead_time_mean", 7.0))) for cat in categories], dtype=float)
        safety_stock_z = 1.65
        reorder_points = np.array([
            np.ceil(means[i] * lead_times[i] + safety_stock_z * std_devs[i] * np.sqrt(lead_times[i]))
            for i in range(dim)
        ], dtype=float)

        # Swarm State Initialization
        rng = np.random.default_rng()
        positions = rng.uniform(self.min_allocation, self.max_allocation, size=(swarm_size, dim))
        velocities = rng.uniform(-0.05, 0.05, size=(swarm_size, dim))

        pbest_positions = positions.copy()
        pbest_weights = np.zeros((swarm_size, dim))
        pbest_fitnesses = np.full(swarm_size, -1e30)

        gbest_position = np.zeros(dim)
        gbest_weights = np.zeros(dim)
        gbest_fitness = -1e30

        # Initial evaluation
        for i in range(swarm_size):
            w = self.project_to_simplex(positions[i], self.min_allocation, self.max_allocation)
            fit, prof, pvar, rsk, hhi, enc, div = self.evaluate_fitness(
                w, means, margins, cov_matrix, gamma, costs=costs, reorder_points=reorder_points
            )

            pbest_weights[i] = w
            pbest_fitnesses[i] = fit

            if fit > gbest_fitness:
                gbest_fitness = fit
                gbest_position = positions[i].copy()
                gbest_weights = w.copy()

        convergence_history = []
        mean_swarm_history = []
        candidate_portfolios = []

        # Swarm Iteration Loop
        for iter_idx in range(max_iterations):
            # Dynamic adaptive inertia weight decay w(t)
            w_curr = self.w_start - (iter_idx / max_iterations) * (self.w_start - self.w_end)
            current_iter_fitnesses = []

            for i in range(swarm_size):
                r1 = rng.random(size=dim)
                r2 = rng.random(size=dim)

                # Velocity Update Equation
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = w_curr * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -0.1, 0.1)

                # Position Update
                positions[i] += velocities[i]

                # Simplex projection with box constraints
                w = self.project_to_simplex(positions[i], self.min_allocation, self.max_allocation)
                fit, prof, pvar, rsk, hhi, enc, div = self.evaluate_fitness(
                    w, means, margins, cov_matrix, gamma, costs=costs, reorder_points=reorder_points
                )

                port_std = float(np.sqrt(pvar))
                pes = float(prof / port_std) if port_std > 0 else 0.0
                candidate_portfolios.append({
                    "risk_std": round(port_std, 2),
                    "profit": round(float(prof), 2),
                    "pes": round(pes, 4),
                    "fitness": round(float(fit), 2)
                })

                current_iter_fitnesses.append(fit)

                # Update Personal Best
                if fit > pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fit
                    pbest_positions[i] = positions[i].copy()
                    pbest_weights[i] = w.copy()

                    # Update Global Best
                    if fit > gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = positions[i].copy()
                        gbest_weights = w.copy()

            convergence_history.append(gbest_fitness)
            mean_swarm_history.append(float(np.mean(current_iter_fitnesses)))

        # Final Evaluation for gbest_weights
        final_fit, final_profit, final_pvar, final_vol_risk, final_hhi, final_enc, final_div_pen = self.evaluate_fitness(
            gbest_weights, means, margins, cov_matrix, gamma
        )

        port_std = float(np.sqrt(final_pvar))
        pes = round(float(final_profit / port_std), 4) if port_std > 0 else 0.0

        # Convergence speed (iteration at 99% max fitness)
        max_fit = max(convergence_history)
        target_fit = 0.99 * max_fit
        conv_iter = len(convergence_history)
        for idx, val in enumerate(convergence_history):
            if val >= target_fit:
                conv_iter = idx + 1
                break

        allocations_pct = {
            categories[i]: round(float(gbest_weights[i] * 100.0), 2)
            for i in range(dim)
        }

        allocations_raw = {
            categories[i]: round(float(gbest_weights[i]), 4)
            for i in range(dim)
        }

        return {
            "algorithm": "Particle Swarm Optimization (Markowitz-Herfindahl IPSO)",
            "dimension": dim,
            "swarm_size_used": swarm_size,
            "max_iterations_used": max_iterations,
            "lambda_risk": self.lambda_risk,
            "alpha_diversification": self.alpha_diversification,
            "gamma_diversification": gamma,
            "min_allocation_bound": self.min_allocation,
            "max_allocation_bound": self.max_allocation,
            "optimal_fitness": round(final_fit, 2),
            "expected_gross_profit": round(final_profit, 2),
            "portfolio_expected_profit": round(final_profit, 2),
            "portfolio_variance": round(final_pvar, 4),
            "portfolio_std_dev": round(port_std, 2),
            "portfolio_pes": pes,
            "volatility_risk_penalty": round(final_vol_risk, 4),
            "diversification_penalty": round(final_div_pen, 2),
            "herfindahl_index": round(final_hhi, 4),
            "herfindahl_index_hhi": round(final_hhi, 4),
            "effective_number_categories": round(final_enc, 2),
            "effective_number_categories_enc": round(final_enc, 2),
            "diversification_ratio": round(final_enc, 2),
            "convergence_iteration": conv_iter,
            "allocations_percentage": allocations_pct,
            "allocations_weight": allocations_raw,
            "convergence_history": [round(val, 2) for val in convergence_history],
            "mean_swarm_history": [round(val, 2) for val in mean_swarm_history],
            "candidate_portfolios": candidate_portfolios
        }
