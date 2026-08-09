"""
Optera ETL Pipeline Runner & Report Generator (v1.0)

Executes the complete end-to-end ETL processing pipeline:
1. Schema Mapping (Schema.py) -> mapped.csv
2. Data Quality Validation (validator.py) -> validated.csv
3. Data Cleaning (cleaner.py) -> cleaned.csv
4. Feature Engineering (feature_engineering.py) -> engineered.csv
5. Demand Aggregation (aggregator.py) -> daily_category_demand.csv
6. Execution Analytics & Visual Chart Suite (reporter.py) -> etl_report.md + plots
7. Pipeline Metadata Generation -> metadata.json
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root & optera directory are in sys.path
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import pandas as pd

from optera.etl.src.etl.Schema import Schema
from optera.etl.src.etl.validator import DataValidator
from optera.etl.src.etl import cleaner
from optera.etl.src.etl import feature_engineering
from optera.etl.src.etl import aggregator
from optera.etl.src.etl.reporter import ETLReporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
)
logger = logging.getLogger("OpteraETLPipeline")

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "demand_forecasting.csv"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = PROCESSED_DIR / "report"


def run_pipeline(
    input_csv: Optional[Union[str, Path]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    procurement_multiplier: float = 0.70,
    holding_rate: float = 0.15,
    lead_time_days: float = 7.0,
    output_dir: Optional[Union[str, Path]] = None
) -> Path:
    logger.info("=" * 70)
    logger.info("STARTING OPTERA ETL PIPELINE (v1.0)")
    logger.info("=" * 70)

    workspace = Path(output_dir).resolve() if output_dir else Path.cwd()
    processed_dir = workspace / "data" / "processed"
    reports_dir = workspace / "reports"
    pdfs_dir = workspace / "pdfs"
    plots_dir = workspace / "plots"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_csv).resolve() if input_csv else (RAW_FILE if RAW_FILE.exists() else workspace / "demand_forecasting.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input raw CSV file not found at: {input_path}\n"
            f"Please ensure a valid CSV file path is passed to optera.etl(data_path='...')"
        )

    reporter = ETLReporter(input_path, processed_dir, reports_dir=reports_dir, pdfs_dir=pdfs_dir, plots_dir=plots_dir)

    # -------------------------------------------------------------
    # Step 1: Load Raw CSV & Apply Schema Mapping
    # -------------------------------------------------------------
    t0 = time.time()
    logger.info("\n--- STEP 1: SCHEMA MAPPING ---")
    raw_df = pd.read_csv(input_csv, encoding="ISO-8859-1")
    raw_rows = len(raw_df)
    logger.info("Loaded raw CSV with %d rows and columns: %s", raw_rows, list(raw_df.columns))

    schema = Schema()

    if column_mapping:
        logger.info("Applying user-defined column mapping: %s", column_mapping)
        schema.define(column_mapping)
    else:
        logger.info("Auto-detecting column mappings from raw CSV headers...")
        col_lower_map = {str(c).strip().lower(): c for c in raw_df.columns}
        mapping = {}

        for k in ["date", "order date", "order_date", "time", "day"]:
            if k in col_lower_map:
                mapping[col_lower_map[k]] = "date"
                break
        for k in ["product id", "product_id", "sku", "sku_id", "item_id", "product"]:
            if k in col_lower_map:
                mapping[col_lower_map[k]] = "sku_id"
                break
        for k in ["category", "cat", "product_category", "department"]:
            if k in col_lower_map:
                mapping[col_lower_map[k]] = "category"
                break
        for k in ["units sold", "units_sold", "demand", "quantity", "units", "sales_units"]:
            if k in col_lower_map:
                mapping[col_lower_map[k]] = "quantity"
                break
        for k in ["price", "unit price", "unit_price", "sales", "retail_price"]:
            if k in col_lower_map:
                mapping[col_lower_map[k]] = "unit_price"
                break
        for k in ["inventory level", "inventory_level", "stock", "on_hand"]:
            if k in col_lower_map:
                mapping[col_lower_map[k]] = "inventory_level"
                break
        for k in ["promotion", "promotion_flag", "promo"]:
            if k in col_lower_map:
                mapping[col_lower_map[k]] = "promotion_flag"
                break

        schema.define(mapping)
    mapped_df = schema.apply(raw_df)
    mapped_path = processed_dir / "mapped.csv"
    mapped_df.to_csv(mapped_path, index=False)
    d1 = time.time() - t0
    reporter.log_step(1, "Schema Mapping", raw_rows, len(mapped_df), mapped_path, d1, "Column standardization & schema check")
    logger.info("Step 1 complete in %.3fs. Saved -> %s", d1, mapped_path)

    # -------------------------------------------------------------
    # Step 2: Data Quality Validation
    # -------------------------------------------------------------
    t0 = time.time()
    logger.info("\n--- STEP 2: DATA VALIDATION ---")
    validator = DataValidator(mapped_df)
    validated_df = validator.validate()
    validated_path = processed_dir / "validated.csv"
    validator.save(validated_path)
    d2 = time.time() - t0
    reporter.log_step(2, "Data Validation", len(mapped_df), len(validated_df), validated_path, d2, "Quality checks & type verification")
    logger.info("Step 2 complete in %.3fs. Saved -> %s", d2, validated_path)

    # -------------------------------------------------------------
    # Step 3: Data Cleaning
    # -------------------------------------------------------------
    t0 = time.time()
    logger.info("\n--- STEP 3: DATA CLEANING ---")
    cleaner_obj = cleaner.DataCleaner(validated_df)
    cleaned_df = cleaner_obj.clean()
    cleaned_path = processed_dir / "cleaned.csv"
    cleaner_obj.save(cleaned_path)
    d3 = time.time() - t0
    reporter.log_step(3, "Data Cleaning", len(validated_df), len(cleaned_df), cleaned_path, d3, "Removed non-positive quantities/prices & duplicates")
    logger.info("Step 3 complete in %.3fs. Saved -> %s", d3, cleaned_path)

    # -------------------------------------------------------------
    # Step 4: Feature Engineering
    # -------------------------------------------------------------
    t0 = time.time()
    logger.info("\n--- STEP 4: FEATURE ENGINEERING ---")
    engineer_obj = feature_engineering.FeatureEngineer(cleaned_df)
    engineered_df = engineer_obj.transform(
        procurement_multiplier=procurement_multiplier,
        holding_rate=holding_rate,
        default_lead_time_days=lead_time_days
    )
    engineered_path = processed_dir / "engineered.csv"
    engineer_obj.save(engineered_path)
    d4 = time.time() - t0
    reporter.log_step(4, "Feature Engineering", len(cleaned_df), len(engineered_df), engineered_path, d4, f"Synthesized revenue, unit_cost ({procurement_multiplier*100:.0f}%), holding_cost, lead_time")
    logger.info("Step 4 complete in %.3fs. Saved -> %s", d4, engineered_path)

    # -------------------------------------------------------------
    # Step 5: Demand Aggregation
    # -------------------------------------------------------------
    t0 = time.time()
    logger.info("\n--- STEP 5: DEMAND AGGREGATION ---")
    agg_obj = aggregator.DemandAggregator(engineered_df)
    daily_demand_df = agg_obj.aggregate_by_category(fill_zero_demand_dates=True)
    daily_demand_path = processed_dir / "daily_category_demand.csv"
    agg_obj.save(daily_demand_df, daily_demand_path)
    d5 = time.time() - t0
    reporter.log_step(5, "Demand Aggregation", len(engineered_df), len(daily_demand_df), daily_demand_path, d5, "Daily category demand time-series aggregation")
    logger.info("Step 5 complete in %.3fs. Saved -> %s", d5, daily_demand_path)

    # -------------------------------------------------------------
    # Step 6: Visual Charts Suite & Execution Report (MD & PDF)
    # -------------------------------------------------------------
    logger.info("\n--- STEP 6: GENERATING ANALYTICAL CHARTS & EXECUTION REPORT ---")
    plot_paths = reporter.generate_plots(daily_demand_df, engineered_df)
    etl_report_file, etl_pdf_file = reporter.generate_report(daily_demand_df, engineered_df, plot_paths)

    # -------------------------------------------------------------
    # Step 7: Pipeline Metadata Generation
    # -------------------------------------------------------------
    logger.info("\n--- STEP 7: GENERATING PIPELINE METADATA ---")
    metadata_path = processed_dir / "metadata.json"
    
    # Extract date ranges
    min_date = str(daily_demand_df["date"].min())
    max_date = str(daily_demand_df["date"].max())
    categories = sorted([str(c) for c in daily_demand_df["category"].unique()])

    metadata = {
        "pipeline_version": "1.0",
        "execution_timestamp": datetime.datetime.now().isoformat(),
        "dataset": {
            "dimensions": {
                "rows": int(daily_demand_df.shape[0]),
                "columns": int(daily_demand_df.shape[1])
            },
            "columns": list(daily_demand_df.columns),
            "categories": categories,
            "date_range": {
                "min_date": min_date,
                "max_date": max_date
            }
        },
        "features_generated": ["revenue", "unit_cost", "holding_cost", "lead_time"],
        "feature_engineering_parameters": {
            "procurement_multiplier": procurement_multiplier,
            "holding_rate": holding_rate,
            "default_lead_time_days": 7.0
        }
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Step 7 complete. Saved pipeline metadata -> %s", metadata_path)

    logger.info("=" * 70)
    logger.info("OPTERA ETL PIPELINE EXECUTED SUCCESSFULLY!")
    logger.info("ETL Markdown Report -> %s", etl_report_file)
    logger.info("ETL Executive PDF   -> %s", etl_pdf_file)
    logger.info("Visual Plots Saved  -> %s", reporter.plots_dir)
    logger.info("=" * 70)

    # Print Report Summary to Console
    print("\n" + "=" * 70)
    print("OPTERA ETL READY DATASET PREVIEW (Head 5 Rows)")
    print("=" * 70)
    print(daily_demand_df.head(5).to_string(index=False))
    print("=" * 70)
    print("\nREPORTS GENERATED IN 'optera/reports/' & 'optera/pdfs/':")
    print(f"  • ETL Markdown Report : {etl_report_file}")
    print(f"  • ETL Executive PDF   : {etl_pdf_file}")
    print(f"  • Pipeline Metadata    : {metadata_path}")
    print("=" * 70 + "\n")

    return daily_demand_path


if __name__ == "__main__":
    run_pipeline()