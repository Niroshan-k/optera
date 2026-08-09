"""
Optera Optimization Layer Orchestrator (v1.8)

Loads category parameters from demand_model.json, runs high-performance Particle Swarm
Optimization over the Markowitz-Herfindahl portfolio objective function, conducts
Monte Carlo stability analysis, generates 6 visual evaluation plots, and compiles
both Markdown and Executive PDF reports.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root & optimizers paths are in sys.path
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optera.optimizers.pso_wrapper import OpteraPSO
from optera.optimizers.evaluator import OptimizerEvaluator
from optera.optimizers.reporter import OptimizerReporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
)
logger = logging.getLogger("OpteraOptimizerPipeline")

# Paths
ANALYTICS_DATA_DIR = PROJECT_ROOT / "optera" / "analytics" / "data"
DEMAND_MODEL_JSON = ANALYTICS_DATA_DIR / "demand_model.json"
OPTIMIZER_DATA_DIR = PROJECT_ROOT / "optera" / "optimizers" / "data"
OUTPUT_JSON = OPTIMIZER_DATA_DIR / "optimization_results.json"

OPTERA_DIR = PROJECT_ROOT / "optera"
REPORTS_DIR = OPTERA_DIR / "reports"
PDFS_DIR = OPTERA_DIR / "pdfs"
PLOTS_DIR = OPTERA_DIR / "plots"


def run_optimization(
    data_path: Optional[Union[str, Path]] = None,
    demand_model_path: Optional[Union[str, Path]] = None,
    total_budget: float = 500000.0,
    swarm_size: Optional[int] = None,
    max_iterations: Optional[int] = None,
    lambda_risk: float = 0.0001,
    alpha_diversification: float = 0.10,
    shortage_penalty_weight: float = 1.0,
    min_allocation: float = 0.05,
    max_allocation: float = 0.40,
    num_runs: int = 30,
    output_dir: Optional[Union[str, Path]] = None
) -> Path:
    r"""
    Executes the Optimization Layer with comprehensive evaluation & reporting.

    Parameters
    ----------
    data_path : Optional[Union[str, Path]]
        Path to demand_model.json file.
    demand_model_path : Optional[Union[str, Path]]
        Alternative keyword argument for data_path.
    swarm_size : Optional[int]
        User-specified swarm particle count.
    max_iterations : Optional[int]
        User-specified max iterations.
    lambda_risk : float
        Risk aversion parameter \lambda for portfolio variance penalty.
    alpha_diversification : float
        Herfindahl diversification factor \alpha \in [0.05, 0.20] (default: 0.10).
    min_allocation : float
        Minimum budget allocation per category (default: 0.05 / 5%).
    max_allocation : float
        Maximum budget allocation per category (default: 0.40 / 40%).
    num_runs : int
        Number of independent PSO runs for Monte Carlo stability analysis (default: 30).
    output_dir : Optional[Union[str, Path]]
        Target workspace directory.
    """
    logger.info("=" * 70)
    logger.info("STARTING OPTERA OPTIMIZATION LAYER (v1.8 - Markowitz-Herfindahl IPSO)")
    logger.info("=" * 70)

    workspace = Path(output_dir).resolve() if output_dir else Path.cwd()
    reports_dir = workspace / "reports"
    pdfs_dir = workspace / "pdfs"
    plots_dir = workspace / "plots"
    optimizers_data_dir = workspace / "optimizers" / "data"

    reports_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    optimizers_data_dir.mkdir(parents=True, exist_ok=True)

    src_json = Path(data_path or demand_model_path).resolve() if (data_path or demand_model_path) else workspace / "analytics" / "data" / "demand_model.json"
    if not src_json.exists():
        src_json = DEMAND_MODEL_JSON

    if not src_json.exists():
        raise RuntimeError(
            f"Prerequisite model missing: 'demand_model.json' not found at: {src_json}\n"
            f"Cannot run Optimization layer without analytics model. Please run 'optera.analytics(...)' first."
        )
    demand_model_path = src_json

    t0 = time.time()

    # 1. Primary Optimization Run
    pso = OpteraPSO(
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        lambda_risk=lambda_risk,
        alpha_diversification=alpha_diversification,
        min_allocation=min_allocation,
        max_allocation=max_allocation,
        shortage_penalty_weight=shortage_penalty_weight
    )
    results = pso.optimize_from_model(demand_model_path, total_budget=total_budget)

    # 2. Optimization Evaluator Suite (Monte Carlo Stability + Efficient Frontier)
    evaluator = OptimizerEvaluator(
        demand_model_path=demand_model_path,
        lambda_risk=lambda_risk,
        alpha_diversification=alpha_diversification,
        min_allocation=min_allocation,
        max_allocation=max_allocation
    )
    stability_res = evaluator.run_stability_analysis(num_runs=num_runs)
    frontier_res = evaluator.generate_efficient_frontier(num_points=25)

    # Attach stability & evaluation fields to results JSON
    results["stability_cv"] = stability_res["stability_cv"]
    results["fitness_confidence_interval"] = stability_res["fitness_confidence_interval"]
    results["stability_analysis"] = {
        "num_runs": stability_res["num_runs"],
        "mean_fitness": stability_res["mean_fitness"],
        "std_fitness": stability_res["std_fitness"],
        "stability_cv": stability_res["stability_cv"],
        "fitness_confidence_interval": stability_res["fitness_confidence_interval"]
    }

    duration = time.time() - t0
    results["execution_time_seconds"] = round(duration, 3)

    # Save complete optimization_results.json
    output_json = optimizers_data_dir / "optimization_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 3. Generate Visual Charts, Markdown Report & Executive PDF
    logger.info("Generating Visual Evaluation Plots, Markdown Report & Executive PDF...")
    with open(demand_model_path, "r", encoding="utf-8") as f:
        demand_model_dict = json.load(f)

    reporter = OptimizerReporter(reports_dir=reports_dir, pdfs_dir=pdfs_dir, plots_dir=plots_dir)
    plot_paths = reporter.generate_all_plots(results, stability_res, frontier_res, demand_model_dict)
    md_file = reporter.generate_markdown_report(results, stability_res, frontier_res, plot_paths)
    pdf_file = reporter.generate_pdf_report(results, stability_res, plot_paths)

    logger.info("Optimization complete in %.3fs.", duration)
    logger.info("=" * 70)
    logger.info("OPTERA OPTIMIZATION LAYER EXECUTED SUCCESSFULLY!")
    logger.info("Optimization Results JSON -> %s", OUTPUT_JSON)
    logger.info("Optimization Markdown    -> %s", md_file)
    logger.info("Optimization PDF Report  -> %s", pdf_file)
    logger.info("Evaluation Plots Saved   -> %s", PLOTS_DIR)
    logger.info("=" * 70)

    # Print Summary Table to Console
    print("\n" + "=" * 70)
    print("MARKOWITZ-HERFINDAHL PSO PORTFOLIO ALLOCATION RESULTS")
    print("=" * 70)
    print(f"  • Decision Variable Dimension : {results['dimension']} categories")
    print(f"  • Swarm Size Used             : {results['swarm_size_used']} particles")
    print(f"  • Iterations Completed        : {results['max_iterations_used']} iterations")
    print(f"  • Convergence Speed           : Iteration {results['convergence_iteration']} (99% max fitness)")
    print(f"  • Execution Runtime           : {results['execution_time_seconds']:.3f} seconds")
    print(f"  • PSO Stability Trial Count   : {stability_res['num_runs']} independent runs")
    print(f"  • Best Fitness Score          : {stability_res.get('best_fitness', results['optimal_fitness']):,.2f}")
    print(f"  • Mean Fitness Score          : {stability_res['mean_fitness']:,.2f}")
    print(f"  • Fitness Std Dev             : {stability_res['std_fitness']:.4f}")
    print(f"  • Stability CV (Fitness)      : {stability_res['stability_cv']:.6f}")
    print(f"  • 95% Fitness Confidence CI   : [{stability_res['fitness_confidence_interval'][0]:,.2f}, {stability_res['fitness_confidence_interval'][1]:,.2f}]")
    print(f"  • Risk Aversion Parameter (lambda) : {results['lambda_risk']}")
    print(f"  • Diversification Factor (alpha)   : {results['alpha_diversification']}")
    print(f"  • Derived Gamma Penalty (gamma)    : {results['gamma_diversification']}")
    print(f"  • Category Bounds [min, max]       : [{results['min_allocation_bound']*100:.0f}%, {results['max_allocation_bound']*100:.0f}%]")
    print("-" * 70)
    print("OPTIMAL DIVERSIFIED BUDGET ALLOCATION VECTOR (w*):")
    for cat, pct in results["allocations_percentage"].items():
        print(f"  • {cat:<18} : {pct:>6.2f}%")
    print("-" * 70)
    print(f"  • Expected Gross Profit       : ${results['expected_gross_profit']:,.2f}")
    print(f"  • Portfolio Demand Variance   : {results['portfolio_variance']:,.2f} (units^2)")
    print(f"  • Portfolio Demand Std Dev    : {results['portfolio_std_dev']:,.2f} units")
    print(f"  • Volatility Risk Penalty     : {results['volatility_risk_penalty']:.4f} (unitless risk penalty)")
    print(f"  • Diversification Penalty     : ${results['diversification_penalty']:,.2f}")
    print(f"  • Herfindahl Concentration HHI: {results['herfindahl_index_hhi']:.4f}")
    print(f"  • Effective Categories ENC/DR : {results['effective_number_categories_enc']:.2f} categories")
    print(f"  • Procurement Efficiency PES  : {results['portfolio_pes']:.4f} (Profit / Risk)")
    print(f"  • Maximum Portfolio Fitness F : {results['optimal_fitness']:,.2f} (unitless composite score)")
    print("=" * 70 + "\n")

    return OUTPUT_JSON


if __name__ == "__main__":
    run_optimization()
