"""
Optera Quantitative Supply Chain Framework - Testing Workspace 4

Executes the 5-layer pipeline and generates consolidated optera_report.md and optera_report.pdf
with full image links and complete PDF sections.
"""

import sys
from pathlib import Path

TESTING4_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTING4_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optera

def main():
    print("=" * 80)
    print("OPTERA TEST PIPELINE EXECUTION IN WORKSPACE 4")
    print("=" * 80)

    dataset_csv = TESTING4_DIR / "demand_forecasting.csv"
    workspace = TESTING4_DIR

    column_mapping = {
        "Date": "date",
        "Product ID": "sku_id",
        "Category": "category",
        "Units Sold": "quantity",
        "Price": "unit_price",
        "Inventory Level": "inventory_level",
        "Promotion": "promotion_flag"
    }

    # 1. ETL Layer
    print("--- [STEP 1/5] Running ETL Processing Pipeline (optera.etl) ---")
    etl_res = optera.etl(
        input_csv=dataset_csv,
        column_mapping=column_mapping,
        procurement_multiplier=0.70,
        holding_rate=0.15,
        lead_time_days=7.0,
        output_dir=workspace
    )

    processed_data_csv = workspace / "data" / "processed" / "daily_category_demand.csv"

    # 2. Demand Analytics Layer
    print("--- [STEP 2/5] Running Demand Analytics Layer (optera.analytics) ---")
    analytics_res = optera.analytics(
        data_path=processed_data_csv,
        distributions=["normal", "poisson", "gamma", "nbinom"],
        confidence_levels=[0.95, 0.99],
        goodness_of_fit_metric="aic",
        output_dir=workspace
    )

    demand_model_json = workspace / "analytics" / "data" / "demand_model.json"

    # 3. Optimization Layer
    print("--- [STEP 3/5] Running Optimization Layer (optera.optimize) ---")
    opt_res = optera.optimize(
        data_path=demand_model_json,
        total_budget=500000.0,
        swarm_size=50,
        max_iterations=150,
        lambda_risk=0.0001,
        alpha_diversification=0.10,
        shortage_penalty_weight=1.0,
        output_dir=workspace
    )

    opt_results_json = workspace / "optimizers" / "data" / "optimization_results.json"

    # 4. Simulation Layer
    print("--- [STEP 4/5] Running Monte Carlo Simulation Layer (optera.simulation) ---")
    sim_res = optera.simulation(
        demand_model_path=demand_model_json,
        optimization_results_path=opt_results_json,
        num_simulations=1000,
        horizon_days=90,
        total_budget=500000.0,
        initial_cash=100000.0,
        safety_stock_z=1.645,
        review_cycle_days=14,
        min_order_quantity=100.0,
        output_dir=workspace
    )

    # 5. Master Report Consolidation
    print("--- [STEP 5/5] Consolidating Master Executive Report (optera.run) ---")
    master_report = optera.run(
        data_path=dataset_csv,
        output_dir=workspace,
        column_mapping=column_mapping,
        total_budget=500000.0,
        num_simulations=1000,
        horizon_days=90
    )

    print("=" * 80)
    print("ALL OPTERA PIPELINE LAYERS EXECUTED SUCCESSFULLY IN TESTING4!")
    print("=" * 80)
    print(f"  • Executive Markdown Report : {workspace / 'optera_report.md'}")
    print(f"  • Executive PDF Report      : {workspace / 'optera_report.pdf'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
