"""
Optera High-Performance Monte Carlo Event-Driven Simulation Engine (v2.0)

Simulates M stochastic day-by-day procurement & inventory trial paths over T days,
computing financial distributions, inventory utilization, lost sales, VaR/CVaR,
and exporting simulation_paths.csv.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("OpteraSimulationEngine")


class OpteraSimulator:
    """
    Monte Carlo Event-Driven Supply Chain Simulator Engine.
    """

    def __init__(
        self,
        demand_model_path: Union[str, Path],
        optimization_results_path: Union[str, Path],
        total_budget: float = 500000.0,
        num_simulations: int = 1000,
        horizon_days: int = 90,
        initial_cash: float = 100000.0,
        min_cash_threshold: float = 50000.0,
        safety_stock_z: float = 1.645,
        review_cycle_days: int = 14,
        min_order_quantity: float = 100.0,
        warehouse_capacity: Optional[float] = None
    ):
        self.demand_model_path = Path(demand_model_path)
        self.optimization_results_path = Path(optimization_results_path)
        self.total_budget = float(total_budget)
        self.num_simulations = int(num_simulations)
        self.horizon_days = int(horizon_days)
        self.initial_cash = float(initial_cash)
        self.min_cash_threshold = float(min_cash_threshold)
        self.safety_stock_z = float(safety_stock_z)
        self.review_cycle_days = int(review_cycle_days)
        self.min_order_quantity = float(min_order_quantity)
        self.warehouse_capacity = float(warehouse_capacity) if warehouse_capacity is not None else None

    def load_inputs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Loads demand model parameters and optimization weights."""
        if not self.demand_model_path.exists():
            raise FileNotFoundError(f"Demand model JSON not found at: {self.demand_model_path}")
        if not self.optimization_results_path.exists():
            raise FileNotFoundError(f"Optimization results JSON not found at: {self.optimization_results_path}")

        with open(self.demand_model_path, "r", encoding="utf-8") as f:
            demand_model = json.load(f)

        with open(self.optimization_results_path, "r", encoding="utf-8") as f:
            opt_results = json.load(f)

        return demand_model, opt_results

    def run_simulation(self) -> Dict[str, Any]:
        """
        Executes M independent stochastic day-by-day simulation paths over T days.
        """
        t0 = time.time()
        logger.info(
            "Running Optera Monte Carlo Event-Driven Simulation (%d trials over %d days)...",
            self.num_simulations, self.horizon_days
        )

        demand_model, opt_results = self.load_inputs()
        categories = [k for k in demand_model.keys() if k != "metadata"]
        alloc_weights = opt_results.get("allocations_weight", {})

        # Parse Category Specs
        cat_specs = {}
        total_allocated_budget = 0.0

        for cat in categories:
            info = demand_model[cat]
            w = alloc_weights.get(cat, 1.0 / len(categories))
            fin = info.get("financials", {})
            cost = float(fin.get("unit_cost_mean", 50.0))
            price = float(fin.get("unit_price_mean", 70.0))
            holding_annual = float(fin.get("holding_cost_mean", cost * 0.15))
            holding = holding_annual / 365.0
            lead = max(1.0, float(fin.get("lead_time_mean", 7.0)))

            mean_d = float(info["mean"])
            std_d = float(info["std"])

            # 1. Base-Stock (s, S) Parameters
            review_cycle_days = self.review_cycle_days
            moq = self.min_order_quantity
            capacity_multiplier = 2.5

            lead_demand_std = std_d * np.sqrt(lead)
            safety_stock = self.safety_stock_z * lead_demand_std
            reorder_point_s = np.ceil(mean_d * lead + safety_stock)
            order_up_to_S = reorder_point_s + np.ceil(mean_d * review_cycle_days)
            max_warehouse_capacity = np.ceil(order_up_to_S * capacity_multiplier) if self.warehouse_capacity is None else self.warehouse_capacity

            # 2. Exact PSO Initial Allocation (I_0)
            cat_budget = self.total_budget * w
            initial_units_I0 = float(np.floor(cat_budget / max(1.0, cost)))
            actual_spent = initial_units_I0 * cost
            total_allocated_budget += actual_spent

            cat_specs[cat] = {
                "mean": mean_d,
                "std": std_d,
                "dist": info.get("best_fit", {}).get("distribution", "normal"),
                "params": info.get("best_fit", {}).get("params", {}),
                "cost": cost,
                "price": price,
                "holding": holding,
                "lead": lead,
                "weight": w,
                "initial_units": initial_units_I0,
                "reorder_point_s": reorder_point_s,
                "order_up_to_S": order_up_to_S,
                "max_warehouse_capacity": max_warehouse_capacity,
                "review_cycle_days": review_cycle_days,
                "moq": moq
            }

        unused_budget = max(0.0, self.total_budget - total_allocated_budget)

        # Simulation storage
        rng = np.random.default_rng(1337)
        M = self.num_simulations
        T = self.horizon_days

        # Performance accumulators across trials
        trial_profits = np.zeros(M)
        trial_ending_cash = np.zeros(M)
        trial_service_levels = np.zeros(M)
        trial_fill_rates = np.zeros(M)
        trial_turnovers = np.zeros(M)
        trial_dois = np.zeros(M)
        trial_lost_revs = np.zeros(M)
        trial_lost_profs = np.zeros(M)
        trial_reorders = np.zeros(M)

        cat_stockout_counts = {c: 0 for c in categories}
        cat_tot_revenue = {c: 0.0 for c in categories}
        cat_tot_cost = {c: 0.0 for c in categories}
        cat_tot_holding = {c: 0.0 for c in categories}
        cat_tot_profit = {c: 0.0 for c in categories}
        cat_tot_lost_sales = {c: 0.0 for c in categories}
        cat_tot_lost_rev = {c: 0.0 for c in categories}
        cat_tot_lost_prof = {c: 0.0 for c in categories}
        cat_avg_inv = {c: 0.0 for c in categories}
        cat_max_inv = {c: 0.0 for c in categories}
        cat_min_inv = {c: 0.0 for c in categories}
        cat_ending_inv = {c: 0.0 for c in categories}

        # Daily trajectory matrices for fan charts (M x T)
        daily_profit_matrix = np.zeros((M, T))
        daily_cash_matrix = np.zeros((M, T))
        daily_inventory_matrix = np.zeros((M, T))

        # CSV export records
        csv_path_records = []

        # Run M trials
        for m in range(M):
            cash = self.initial_cash
            cum_profit = 0.0
            tot_demand = 0.0
            tot_fulfilled = 0.0
            tot_cust_orders = 0.0
            tot_satisfied_orders = 0.0

            # Trial state
            on_hand = {c: cat_specs[c]["initial_units"] for c in categories}
            pending_orders = {c: [] for c in categories}  # list of (arrival_day, qty, cost)
            trial_reorder_count = 0
            trial_inv_val_sum = 0.0
            trial_daily_demand_sum = 0.0

            for t in range(1, T + 1):
                day_rev = 0.0
                day_hold = 0.0
                day_demand_tot = 0.0
                day_sales_tot = 0.0
                day_stockout_tot = 0.0
                day_inv_tot = 0.0

                for c in categories:
                    spec = cat_specs[c]

                    # 1. Arriving Orders
                    arrived_qty = 0.0
                    rem_orders = []
                    for (arr_day, o_qty, o_cost) in pending_orders[c]:
                        if arr_day <= t:
                            arrived_qty += o_qty
                        else:
                            rem_orders.append((arr_day, o_qty, o_cost))
                    pending_orders[c] = rem_orders
                    on_hand[c] += arrived_qty

                    # 2. Sample Stochastic Demand
                    d_mean, d_std = spec["mean"], spec["std"]
                    d_dist = spec["dist"]

                    if d_dist == "poisson":
                        demand = float(rng.poisson(max(0.001, d_mean)))
                    elif d_dist == "gamma":
                        shape = max(0.001, (d_mean / max(0.001, d_std)) ** 2)
                        scale = max(0.001, (d_std ** 2) / max(0.001, d_mean))
                        demand = float(rng.gamma(shape, scale))
                    else:
                        demand = max(0.0, float(rng.normal(d_mean, d_std)))

                    demand = np.round(demand)
                    tot_demand += demand
                    if demand > 0:
                        tot_cust_orders += 1.0

                    # 3. Execute Sales & Stockout
                    sales = min(on_hand[c], demand)
                    stockout = demand - sales
                    on_hand[c] -= sales
                    tot_fulfilled += sales

                    if demand > 0 and stockout == 0.0:
                        tot_satisfied_orders += 1.0

                    if stockout > 0:
                        cat_stockout_counts[c] += 1

                    # Financials
                    rev = sales * spec["price"]
                    hold = on_hand[c] * spec["holding"]
                    cogs = sales * spec["cost"]
                    lost_r = stockout * spec["price"]
                    lost_p = stockout * (spec["price"] - spec["cost"])

                    day_rev += rev
                    day_hold += hold
                    day_demand_tot += demand
                    day_sales_tot += sales
                    day_stockout_tot += stockout
                    day_inv_tot += on_hand[c]

                    cat_tot_revenue[c] += rev
                    cat_tot_cost[c] += cogs
                    cat_tot_holding[c] += hold
                    cat_tot_profit[c] += (rev - hold - cogs)
                    cat_tot_lost_sales[c] += stockout
                    cat_tot_lost_rev[c] += lost_r
                    cat_tot_lost_prof[c] += lost_p
                    cat_avg_inv[c] += on_hand[c]

                    # 4. Base-Stock (s, S) Reorder Trigger with MOQ, Working Capital, and Warehouse Capacity Constraints
                    on_order_qty = sum(q for (_, q, _) in pending_orders[c])
                    inv_position = on_hand[c] + on_order_qty

                    if inv_position <= spec["reorder_point_s"]:
                        q_needed = max(0.0, spec["order_up_to_S"] - inv_position)
                        q_target = np.ceil(spec["mean"] * spec["review_cycle_days"])
                        q_affordable = np.floor(cash / max(1.0, spec["cost"]))
                        q_storage_space = max(0.0, spec["max_warehouse_capacity"] - inv_position)

                        q_to_order = min(q_needed, q_target, q_affordable, q_storage_space)

                        if q_to_order >= spec["moq"]:
                            c_order = q_to_order * spec["cost"]
                            cash -= c_order
                            arr_day = t + int(np.round(spec["lead"]))
                            pending_orders[c].append((arr_day, q_to_order, c_order))
                            trial_reorder_count += 1

                    # Record path row (export first 5 trials to CSV for size efficiency)
                    if m < 5:
                        csv_path_records.append({
                            "trial_id": m + 1,
                            "day": t,
                            "category": c,
                            "demand": demand,
                            "sales": sales,
                            "stockout_qty": stockout,
                            "revenue": round(rev, 2),
                            "holding_cost": round(hold, 2),
                            "ending_inventory": on_hand[c],
                            "cash_balance": round(cash, 2),
                            "cumulative_profit": round(cum_profit, 2)
                        })

                # End of day updates
                cash += (day_rev - day_hold)
                net_day_prof = day_rev - day_hold
                cum_profit += net_day_prof

                daily_profit_matrix[m, t - 1] = cum_profit
                daily_cash_matrix[m, t - 1] = cash
                daily_inventory_matrix[m, t - 1] = day_inv_tot

                trial_inv_val_sum += sum(on_hand[c] * cat_specs[c]["cost"] for c in categories)
                trial_daily_demand_sum += day_demand_tot

            # Trial end metrics
            trial_profits[m] = cum_profit
            trial_ending_cash[m] = cash
            trial_service_levels[m] = (tot_satisfied_orders / tot_cust_orders) * 100.0 if tot_cust_orders > 0 else 100.0
            trial_fill_rates[m] = (tot_fulfilled / tot_demand) * 100.0 if tot_demand > 0 else 100.0
            trial_reorders[m] = trial_reorder_count

            avg_inv_val = trial_inv_val_sum / T
            tot_cogs = sum(cat_tot_cost[c] for c in categories) / M
            trial_turnovers[m] = (tot_cogs / avg_inv_val) if avg_inv_val > 0 else 0.0
            trial_dois[m] = (avg_inv_val / (trial_daily_demand_sum / T)) if trial_daily_demand_sum > 0 else 0.0

            for c in categories:
                cat_ending_inv[c] += on_hand[c]

        # Export CSV Paths file
        csv_dir = self.demand_model_path.parent.parent.parent / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_file = csv_dir / "simulation_paths.csv"
        pd.DataFrame(csv_path_records).to_csv(csv_file, index=False)
        logger.info("Exported individual simulation path records to -> %s", csv_file)

        # Statistical Calculations
        sorted_profits = np.sort(trial_profits)
        mean_prof = float(np.mean(sorted_profits))
        median_prof = float(np.median(sorted_profits))
        std_prof = float(np.std(sorted_profits, ddof=1))

        # VaR & CVaR 5%
        var_5_idx = max(0, int(np.floor(0.05 * M)))
        var_5 = float(sorted_profits[var_5_idx])
        cvar_5 = float(np.mean(sorted_profits[:max(1, var_5_idx)]))
        p95_prof = float(sorted_profits[min(M - 1, int(np.floor(0.95 * M)))])

        prob_loss = float(np.mean(trial_profits < 0.0) * 100.0)
        cash_risk = float(np.mean(trial_ending_cash < self.min_cash_threshold) * 100.0)
        overall_service = float(np.mean(trial_service_levels))
        overall_fill = float(np.mean(trial_fill_rates))
        avg_turnover = float(np.mean(trial_turnovers))
        avg_doi = float(np.mean(trial_dois))

        # Category Financial Breakdown
        cat_financials = {}
        for c in categories:
            rev = cat_tot_revenue[c] / M
            cogs = cat_tot_cost[c] / M
        # Calculate Initial Lead-Time Coverage Ratios and Inventory Capital Utilization
        cat_coverage_ratios = {}
        total_initial_units = 0.0
        total_reorder_points = 0.0
        for c in categories:
            i0 = cat_specs[c]["initial_units"]
            reorder_p = cat_specs[c]["reorder_point_s"]
            cov = float(i0 / max(1.0, reorder_p))
            cat_coverage_ratios[c] = cov
            total_initial_units += i0
            total_reorder_points += reorder_p

        portfolio_coverage_ratio = float(total_initial_units / max(1.0, total_reorder_points))

        # Overall Performance Indicators
        mean_profit = float(np.mean(trial_profits))
        median_profit = float(np.median(trial_profits))
        std_profit = float(np.std(trial_profits))
        var_5 = float(np.percentile(trial_profits, 5))
        cvar_5 = float(np.mean(trial_profits[trial_profits <= var_5])) if np.any(trial_profits <= var_5) else var_5
        p95_profit = float(np.percentile(trial_profits, 95))

        mean_service_level = float(np.mean(trial_service_levels))
        if mean_service_level <= 1.0:
            mean_service_level *= 100.0

        mean_fill_rate = float(np.mean(trial_fill_rates))
        if mean_fill_rate <= 1.0:
            mean_fill_rate *= 100.0
        mean_turnover = float(np.mean(trial_turnovers))
        mean_doi = float(np.mean(trial_dois))
        mean_lost_rev = float(np.mean(trial_lost_revs))
        mean_reorders = float(np.mean(trial_reorders))

        # Category Financial & Inventory Aggregations
        cat_summary = {}
        total_avg_inv_capital = 0.0
        total_portfolio_lost_rev = 0.0
        total_portfolio_lost_prof = 0.0
        for c in categories:
            avg_inv = cat_avg_inv[c] / (M * T)
            avg_inv_cap = avg_inv * cat_specs[c]["cost"]
            total_avg_inv_capital += avg_inv_cap

            lost_r_cat = cat_tot_lost_rev[c] / M
            lost_p_cat = cat_tot_lost_prof[c] / M
            total_portfolio_lost_rev += lost_r_cat
            total_portfolio_lost_prof += lost_p_cat

            cat_summary[c] = {
                "total_revenue": round(cat_tot_revenue[c] / M, 2),
                "revenue": round(cat_tot_revenue[c] / M, 2),
                "total_cogs": round(cat_tot_cost[c] / M, 2),
                "cost": round(cat_tot_cost[c] / M, 2),
                "total_holding_cost": round(cat_tot_holding[c] / M, 2),
                "holding_cost": round(cat_tot_holding[c] / M, 2),
                "net_profit": round(cat_tot_profit[c] / M, 2),
                "profit": round(cat_tot_profit[c] / M, 2),
                "total_lost_sales_units": round(cat_tot_lost_sales[c] / M, 1),
                "lost_sales_units": round(cat_tot_lost_sales[c] / M, 1),
                "lost_revenue": round(lost_r_cat, 2),
                "lost_profit": round(lost_p_cat, 2),
                "stockout_probability_pct": round((cat_stockout_counts[c] / (M * T)) * 100.0, 2),
                "stockout_probability": round((cat_stockout_counts[c] / (M * T)) * 100.0, 2),
                "average_onhand_inventory": round(avg_inv, 1),
                "average_inventory": round(avg_inv, 1),
                "average_inventory_value": round(avg_inv_cap, 2),
                "initial_inventory_units": round(cat_specs[c]["initial_units"], 1),
                "reorder_point_units": round(cat_specs[c]["reorder_point_s"], 1),
                "initial_coverage_ratio_pct": round(cat_coverage_ratios[c] * 100.0, 2),
                "roi_pct": round((cat_tot_profit[c] / max(1.0, cat_tot_cost[c])) * 100.0, 2)
            }

        mean_lost_rev = float(total_portfolio_lost_rev)
        mean_lost_prof = float(total_portfolio_lost_prof)

        inventory_capital_utilization_pct = float((total_avg_inv_capital / max(1.0, self.total_budget)) * 100.0)

        duration = time.time() - t0

        results = {
            "num_simulations": M,
            "horizon_days": T,
            "total_budget": self.total_budget,
            "actual_allocated_budget": total_allocated_budget,
            "budget_allocated": total_allocated_budget,
            "unused_budget": unused_budget,
            "budget_unused": unused_budget,
            "initial_cash": self.initial_cash,
            "min_cash_threshold": self.min_cash_threshold,
            "mean_net_profit": mean_profit,
            "mean_profit": mean_profit,
            "median_net_profit": median_profit,
            "median_profit": median_profit,
            "profit_std_dev": std_profit,
            "std_profit": std_profit,
            "value_at_risk_5pct": var_5,
            "var_5pct": var_5,
            "conditional_var_5pct": cvar_5,
            "cvar_5pct": cvar_5,
            "p95_net_profit": p95_profit,
            "profit_95pct": p95_profit,
            "customer_service_level_pct": mean_service_level,
            "overall_service_level": mean_service_level,
            "demand_fill_rate_pct": mean_fill_rate,
            "fill_rate": mean_fill_rate,
            "cash_reserve_risk_pct": float(np.mean(trial_ending_cash < self.min_cash_threshold)) * 100.0,
            "cash_reserve_risk": float(np.mean(trial_ending_cash < self.min_cash_threshold)) * 100.0,
            "probability_of_loss_pct": float(np.mean(trial_profits < 0.0)) * 100.0,
            "probability_of_loss": float(np.mean(trial_profits < 0.0)) * 100.0,
            "inventory_turnover_ratio": mean_turnover,
            "inventory_turnover": mean_turnover,
            "days_of_inventory": mean_doi,
            "total_lost_revenue_impact": mean_lost_rev,
            "total_lost_revenue": mean_lost_rev,
            "total_lost_profit": mean_lost_rev * 0.3,
            "average_reorders_placed": mean_reorders,
            "average_reorder_count": mean_reorders,
            "portfolio_initial_coverage_ratio_pct": round(portfolio_coverage_ratio * 100.0, 2),
            "inventory_capital_utilization_pct": round(inventory_capital_utilization_pct, 2),
            "execution_time_seconds": round(duration, 3),
            "csv_paths_file": str(Path(__file__).resolve().parent.parent / "csv" / "simulation_paths.csv"),
            "category_summary": cat_summary,
            "category_financials": cat_summary,
            "trial_profits": [round(p, 2) for p in trial_profits.tolist()],
            "trial_ending_cash": [round(c, 2) for c in trial_ending_cash.tolist()],
            "trial_turnovers": [round(t, 2) for t in trial_turnovers.tolist()],
            "daily_profit_paths": daily_profit_matrix[:100, :].tolist(),
            "daily_cash_paths": daily_cash_matrix[:100, :].tolist(),
            "daily_inventory_paths": daily_inventory_matrix[:100, :].tolist()
        }

        return results
