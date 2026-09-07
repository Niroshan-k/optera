"""
Optera Supply Chain Library Entrypoint Orchestrator (v2.0)

Coordinates and executes:
1. ETL Data Pipeline (optera.etl.main)
2. Demand Analytics Layer (optera.analytics.main)
3. Optimization Layer (optera.optimizers.main)
4. Monte Carlo Event-Driven Simulation Layer (optera.simulation.main)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
)
logger = logging.getLogger("OpteraOrchestrator")

from optera.etl.main import run_pipeline as run_etl_pipeline
from optera.analytics.main import run_analytics as run_analytics_layer
from optera.optimizers.main import run_optimization as run_optimization_layer
from optera.simulation.main import run_simulation as run_simulation_layer
from optera.master_reporter import MasterReporter


def run(
    data_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    # ETL Layer Parameters
    procurement_multiplier: float = 0.70,
    holding_rate: float = 0.15,
    lead_time_days: float = 7.0,
    # Optimization Layer Parameters
    total_budget: float = 500000.0,
    lambda_risk: float = 0.0001,
    alpha_diversification: float = 0.10,
    shortage_penalty_weight: float = 1.0,
    swarm_size: Optional[int] = None,
    iterations: Optional[int] = None,
    num_runs: int = 30,
    # Simulation Layer Parameters
    num_simulations: int = 1000,
    horizon_days: int = 90,
    initial_cash: float = 100000.0,
    safety_stock_z: float = 1.645
) -> Path:
    """
    Programmatic entry point for the Optera Quantitative Supply Chain Framework.

    Parameters
    ----------
    data_path : Optional[Union[str, Path]]
        Path to raw CSV dataset (e.g., demand_forecasting.csv).
    output_dir : Optional[Union[str, Path]]
        Target workspace directory for reports, plots, pdfs, and CSVs.
    column_mapping : Optional[Dict[str, str]]
        User-defined mapping from dataset column headers to Optera standard schema:
        e.g., {"Date": "date", "Units Sold": "quantity", "Price": "unit_price", ...}
    total_budget : float
        Total initial procurement budget allocated ($B).
    lambda_risk : float
        Risk aversion penalty parameter lambda for Markowitz variance penalty.
    alpha_diversification : float
        Herfindahl diversification factor alpha in [0.05, 0.20].
    shortage_penalty_weight : float
        Dimensionless lead-time shortage penalty weight psi.
    swarm_size : Optional[int]
        Particle Swarm Optimization (PSO) swarm size particle count.
    iterations : Optional[int]
        PSO maximum iteration count.
    num_runs : int
        Number of independent PSO runs for stability analysis (default: 30).
    num_simulations : int
        Number of Monte Carlo simulation trial paths (default: 1000).
    horizon_days : int
        Simulation horizon timeframe in days (default: 90).
    initial_cash : float
        Initial operational cash balance reserve.
    safety_stock_z : float
        Safety stock z-factor multiplier (e.g., 1.645 for 95% service level).
    """
    from optera.utils.ascii_art import print_ascii_banner
    from optera.utils.logger import setup_optera_logging

    workspace_dir = Path(output_dir).resolve() if output_dir else Path.cwd()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Print ASCII Art Banner to Console
    print_ascii_banner(version="1.0.2")

    # Initialize workspace log file handler (output_dir/optera.log)
    setup_optera_logging(output_dir=workspace_dir)

    logger.info("OPTERA SUPPLY CHAIN INTEGRATED FRAMEWORK (v1.0.2)")
    logger.info("Logging persistent execution trace -> %s", workspace_dir / "optera.log")
    logger.info("=" * 80)

    input_csv = Path(data_path).resolve() if data_path else (workspace_dir / "demand_forecasting.csv" if (workspace_dir / "demand_forecasting.csv").exists() else ROOT / "etl" / "data" / "raw" / "demand_forecasting.csv")

    # 1. Run ETL Processing Pipeline
    logger.info("[Layer 1/3] Running ETL Data Processing Pipeline...")
    etl_res = run_etl_pipeline(
        input_csv=input_csv,
        column_mapping=column_mapping,
        procurement_multiplier=procurement_multiplier,
        holding_rate=holding_rate,
        lead_time_days=lead_time_days,
        output_dir=workspace_dir
    )

    # 2. Run Demand Analytics Layer
    logger.info("[Layer 1/3] Running Demand Analytics & Distribution Fitting...")
    analytics_res = run_analytics_layer(output_dir=workspace_dir)

    # 3. Run Optimization Layer
    logger.info("[Layer 2/3] Running Markowitz-Herfindahl PSO Capital Allocation...")
    opt_res = run_optimization_layer(
        total_budget=total_budget,
        swarm_size=swarm_size,
        max_iterations=iterations,
        lambda_risk=lambda_risk,
        alpha_diversification=alpha_diversification,
        shortage_penalty_weight=shortage_penalty_weight,
        num_runs=num_runs,
        output_dir=workspace_dir
    )

    # 4. Run Monte Carlo Event-Driven Simulation Layer
    logger.info("[Layer 3/3] Running Monte Carlo Event-Driven Simulation Layer...")
    sim_res = run_simulation_layer(
        num_simulations=num_simulations,
        horizon_days=horizon_days,
        total_budget=total_budget,
        initial_cash=initial_cash,
        safety_stock_z=safety_stock_z,
        output_dir=workspace_dir
    )

    # 5. Consolidate Integrated Master Executive Report
    logger.info("Compiling Integrated Executive Master Markdown Report & PDF...")
    master_reporter = MasterReporter(workspace_dir=workspace_dir)
    master_md_file = master_reporter.compile_master_report(sim_results=sim_res)

    master_pdf_file = workspace_dir / "optera_report.pdf"

    logger.info("=" * 80)
    logger.info("OPTERA INTEGRATED PIPELINE EXECUTED SUCCESSFULLY!")
    logger.info("Master Executive Markdown Report -> %s", master_md_file)
    logger.info("Master Executive PDF Report      -> %s", master_pdf_file)
    logger.info("=" * 80)

    return master_md_file


def main():
    parser = argparse.ArgumentParser(
        description="Optera Supply Chain Library Console Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--etl-only",
        action="store_true",
        help="Run only the ETL processing pipeline"
    )
    parser.add_argument(
        "--analytics-only",
        action="store_true",
        help="Run only the demand analytical profiling layer"
    )
    parser.add_argument(
        "--optimization-only",
        action="store_true",
        help="Run only the optimization layer"
    )
    parser.add_argument(
        "--simulation-only",
        action="store_true",
        help="Run only the Monte Carlo event-driven simulation layer"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        help="Path to custom input raw CSV dataset for the ETL layer"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Target output workspace directory"
    )
    parser.add_argument(
        "--swarm-size",
        type=int,
        help="User-specified PSO swarm size particle count"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="User-specified PSO maximum iteration count"
    )
    parser.add_argument(
        "--lambda-risk",
        type=float,
        default=0.0001,
        help=r"Risk aversion penalty parameter \lambda for variance w^T \Sigma w (default: 0.0001)"
    )
    parser.add_argument(
        "--alpha-diversification",
        type=float,
        default=0.10,
        help=r"Herfindahl diversification factor \alpha \in [0.05, 0.20] (default: 0.10)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=30,
        help="Number of independent PSO runs for stability analysis (default: 30)"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulation trial paths (default: 1000)"
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=90,
        help="Simulation horizon length in days (default: 90)"
    )
    parser.add_argument(
        "--total-budget",
        type=float,
        default=500000.0,
        help="Total procurement budget configured (default: 500000.0)"
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash reserve balance (default: 100000.0)"
    )
    args = parser.parse_args()

    # Determine execution flow
    run_all = not (args.etl_only or args.analytics_only or args.optimization_only or args.simulation_only)

    if run_all:
        run(
            data_path=args.input_csv,
            output_dir=args.output_dir,
            total_budget=args.total_budget,
            num_simulations=args.num_simulations,
            horizon_days=args.horizon_days,
            initial_cash=args.initial_cash,
            swarm_size=args.swarm_size,
            iterations=args.iterations,
            lambda_risk=args.lambda_risk,
            alpha_diversification=args.alpha_diversification,
            num_runs=args.num_runs
        )
    else:
        if args.etl_only:
            input_path = Path(args.input_csv) if args.input_csv else (ROOT / "etl" / "data" / "raw" / "demand_forecasting.csv")
            run_etl_pipeline(input_path)
        if args.analytics_only:
            run_analytics_layer()
        if args.optimization_only:
            run_optimization_layer(
                swarm_size=args.swarm_size,
                max_iterations=args.iterations,
                lambda_risk=args.lambda_risk,
                alpha_diversification=args.alpha_diversification,
                num_runs=args.num_runs
            )
        if args.simulation_only:
            run_simulation_layer(
                num_simulations=args.num_simulations,
                horizon_days=args.horizon_days,
                total_budget=args.total_budget,
                initial_cash=args.initial_cash
            )


if __name__ == "__main__":
    main()
