"""
Optera Monte Carlo Simulation Layer Orchestrator (v2.0)

Loads demand_model.json and optimization_results.json, runs multithreaded day-by-day
Monte Carlo simulation, exports simulation_paths.csv, generates 7 monospace visual charts,
and compiles both Markdown and Executive PDF reports.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root paths are in sys.path
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optera.simulation.wrapper import OpteraSimulator
from optera.simulation.reporter import SimulationReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
)
logger = logging.getLogger("OpteraSimulationPipeline")

# Paths
ANALYTICS_DATA_DIR = PROJECT_ROOT / "optera" / "analytics" / "data"
DEMAND_MODEL_JSON = ANALYTICS_DATA_DIR / "demand_model.json"
OPTIMIZER_DATA_DIR = PROJECT_ROOT / "optera" / "optimizers" / "data"
OPTIMIZATION_RESULTS_JSON = OPTIMIZER_DATA_DIR / "optimization_results.json"
SIMULATION_DATA_DIR = PROJECT_ROOT / "optera" / "simulation" / "data"
OUTPUT_JSON = SIMULATION_DATA_DIR / "simulation_results.json"

OPTERA_DIR = PROJECT_ROOT / "optera"
REPORTS_DIR = OPTERA_DIR / "reports"
PDFS_DIR = OPTERA_DIR / "pdfs"
PLOTS_DIR = OPTERA_DIR / "plots"


def run_simulation(
    demand_model_path: Optional[Union[str, Path]] = None,
    optimization_results_path: Optional[Union[str, Path]] = None,
    num_simulations: int = 1000,
    horizon_days: int = 90,
    total_budget: float = 500000.0,
    initial_cash: float = 100000.0,
    min_cash_threshold: float = 50000.0,
    safety_stock_z: float = 1.645,
    review_cycle_days: int = 14,
    min_order_quantity: float = 100.0,
    warehouse_capacity: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Executes the Monte Carlo Simulation Layer.

    Parameters
    ----------
    demand_model_path : Optional[Union[str, Path]]
        Path to demand_model.json file.
    optimization_results_path : Optional[Union[str, Path]]
        Path to optimization_results.json file.
    num_simulations : int
        Number of stochastic trial paths (default: 1000).
    horizon_days : int
        Simulation horizon length in days (default: 90).
    total_budget : float
        Total procurement budget configured (default: 500000.0).
    initial_cash : float
        Initial cash reserve balance (default: 100000.0).
    min_cash_threshold : float
        Cash reserve risk threshold (default: 50000.0).
    safety_stock_z : float
        Safety stock z-score coefficient (default: 1.65).
    output_dir : Optional[Union[str, Path]]
        Target workspace directory.
    """
    logger.info("=" * 70)
    logger.info("STARTING OPTERA MONTE CARLO SIMULATION LAYER (v2.0)")
    logger.info("=" * 70)

    workspace = Path(output_dir).resolve() if output_dir else Path.cwd()
    reports_dir = workspace / "reports"
    pdfs_dir = workspace / "pdfs"
    plots_dir = workspace / "plots"
    csv_dir = workspace / "csv"
    simulation_data_dir = workspace / "simulation" / "data"

    reports_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    simulation_data_dir.mkdir(parents=True, exist_ok=True)

    src_dm = Path(demand_model_path).resolve() if demand_model_path else workspace / "analytics" / "data" / "demand_model.json"
    if not src_dm.exists():
        src_dm = DEMAND_MODEL_JSON

    src_opt = Path(optimization_results_path).resolve() if optimization_results_path else workspace / "optimizers" / "data" / "optimization_results.json"
    if not src_opt.exists():
        src_opt = OPTIMIZATION_RESULTS_JSON

    demand_model_path = src_dm
    optimization_results_path = src_opt

    if not demand_model_path.exists():
        raise RuntimeError(
            f"Prerequisite model missing: 'demand_model.json' not found at: {demand_model_path}\n"
            f"Cannot run Simulation layer without analytics model. Please run 'optera.analytics(...)' first."
        )

    if not optimization_results_path.exists():
        raise RuntimeError(
            f"Prerequisite optimization missing: 'optimization_results.json' not found at: {optimization_results_path}\n"
            f"Cannot run Simulation layer without optimization weights. Please run 'optera.optimize(...)' first."
        )

    t0 = time.time()

    simulator = OpteraSimulator(
        demand_model_path=demand_model_path,
        optimization_results_path=optimization_results_path,
        total_budget=total_budget,
        num_simulations=num_simulations,
        horizon_days=horizon_days,
        initial_cash=initial_cash,
        min_cash_threshold=min_cash_threshold,
        safety_stock_z=safety_stock_z,
        review_cycle_days=review_cycle_days,
        min_order_quantity=min_order_quantity,
        warehouse_capacity=warehouse_capacity
    )

    results = simulator.run_simulation()

    # Save results JSON
    output_json = simulation_data_dir / "simulation_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate 7 visual monospace plots, MD report & Executive PDF
    logger.info("Generating 7 Visual Monospace Charts, Markdown Report & Executive PDF...")
    reporter = SimulationReporter(reports_dir=reports_dir, pdfs_dir=pdfs_dir, plots_dir=plots_dir)
    plot_paths = reporter.generate_all_plots(results)
    md_file = reporter.generate_markdown_report(results, plot_paths)
    pdf_file = reporter.generate_pdf_report(results, plot_paths)

    duration = time.time() - t0

    logger.info("Simulation complete in %.3fs.", duration)
    logger.info("=" * 70)
    logger.info("OPTERA MONTE CARLO SIMULATION LAYER EXECUTED SUCCESSFULLY!")
    logger.info("Simulation JSON Output -> %s", OUTPUT_JSON)
    logger.info("Simulation Paths CSV  -> %s", results["csv_paths_file"])
    logger.info("Simulation Markdown   -> %s", md_file)
    logger.info("Simulation PDF Report -> %s", pdf_file)
    logger.info("Visual Plots Saved    -> %s", PLOTS_DIR)
    logger.info("=" * 70)

    # Print Executive KPI Table to Console
    print("\n" + "=" * 70)
    print("OPTERA MONTE CARLO EVENT-DRIVEN SIMULATION RESULTS")
    print("=" * 70)
    print(f"  • Simulation Trial Paths (M)  : {results['num_simulations']:,} paths")
    print(f"  • Simulation Horizon (Days)   : {results['horizon_days']} days")
    print(f"  • Total Budget Configured     : ${results['total_budget']:,.2f}")
    print(f"  • Actual Budget Allocated     : ${results['budget_allocated']:,.2f}")
    print(f"  • Unused Procurement Budget   : ${results['budget_unused']:,.2f}")
    print(f"  • Mean Net Cumulative Profit  : ${results['mean_profit']:,.2f}")
    print(f"  • Median Net Profit           : ${results['median_profit']:,.2f}")
    print(f"  • Profit Standard Dev (s)     : ${results['std_profit']:,.2f}")
    print(f"  • Value at Risk (VaR 5%)      : ${results['var_5pct']:,.2f}")
    print(f"  • Conditional VaR (CVaR 5%)   : ${results['cvar_5pct']:,.2f} (Expected Shortfall)")
    print(f"  • 95th Percentile Net Profit  : ${results['profit_95pct']:,.2f}")
    print(f"  • Customer Service Level      : {results['overall_service_level']:.2f}%")
    print(f"  • Demand Fill Rate            : {results['fill_rate']:.2f}%")
    print(f"  • Cash Reserve Risk (< $50k)  : {results['cash_reserve_risk']:.2f}%")
    print(f"  • Probability of Loss (< $0)  : {results['probability_of_loss']:.2f}%")
    print(f"  • Inventory Turnover Ratio    : {results['inventory_turnover']:.2f}")
    print(f"  • Days of Inventory (DOI)     : {results['days_of_inventory']:.2f} days")
    print(f"  • Total Lost Revenue Impact   : ${results['total_lost_revenue']:,.2f}")
    print(f"  • Average Reorder Events      : {results['average_reorder_count']:.1f} orders")
    print("-" * 70)
    print("CATEGORY FINANCIAL & INVENTORY BREAKDOWN:")
    for cat, fin in results["category_financials"].items():
        print(f"  • {cat:<12} | Profit: ${fin['profit']:>10,.2f} | ROI: {fin['roi_pct']:>6.2f}% | Stockout Risk: {fin['stockout_probability']:>5.2f}% | Avg Inv: {fin['average_inventory']:>6.1f}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_simulation()
