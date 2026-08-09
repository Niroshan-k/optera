"""
Optera Demand Analytics Orchestrator (v1.1)

Loads the aggregated daily category demand dataset, reads execution metadata,
runs category-wise statistical profiling, fits candidate probability distributions
(Normal, Poisson, Gamma, NegBinomial), evaluates goodness-of-fit (AIC/BIC),
generates distribution plots, compiles demand_analytical_report.md, and exports
demand_model.json for downstream optimization.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root & analytics paths are in sys.path
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optera.analytics.src.analytics.profiler import CategoryDemandProfiler, export_demand_model
from optera.analytics.src.analytics.reporter import AnalyticsReporter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
)
logger = logging.getLogger("OpteraAnalyticsPipeline")

# Paths
ETL_DIR = PROJECT_ROOT / "optera" / "etl"
PROCESSED_DIR = ETL_DIR / "data" / "processed"
INPUT_CSV = PROCESSED_DIR / "daily_category_demand.csv"
METADATA_JSON = PROCESSED_DIR / "metadata.json"
OPTERA_DIR = PROJECT_ROOT / "optera"
REPORTS_DIR = OPTERA_DIR / "reports"
PLOTS_DIR = OPTERA_DIR / "plots"
ANALYTICS_DATA_DIR = OPTERA_DIR / "analytics" / "data"


def run_analytics(
    data_path: Optional[Union[str, Path]] = None,
    input_csv: Optional[Union[str, Path]] = None,
    confidence_levels: Optional[List[float]] = None,
    distributions: Optional[List[str]] = None,
    goodness_of_fit_metric: str = "aic",
    output_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Executes the Demand Analytics layer.

    Parameters
    ----------
    data_path : Optional[Union[str, Path]]
        Path to processed daily category demand CSV.
    input_csv : Optional[Union[str, Path]]
        Alternative keyword argument for data_path.
    confidence_levels : Optional[List[float]]
        Custom confidence levels for mean demand (e.g., [0.95, 0.99]).
    distributions : Optional[List[str]]
        List of candidate distributions to fit.
    goodness_of_fit_metric : str
        Criterion metric to select the best distribution ("aic", "bic", or "ks").
    output_dir : Optional[Union[str, Path]]
        Target workspace directory.
    """
    logger.info("=" * 70)
    logger.info("STARTING OPTERA DEMAND ANALYTICS LAYER (v1.1)")
    logger.info("=" * 70)

    workspace = Path(output_dir).resolve() if output_dir else Path.cwd()
    processed_dir = workspace / "data" / "processed"
    reports_dir = workspace / "reports"
    pdfs_dir = workspace / "pdfs"
    plots_dir = workspace / "plots"
    analytics_data_dir = workspace / "analytics" / "data"

    reports_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    analytics_data_dir.mkdir(parents=True, exist_ok=True)

    src_csv = Path(data_path or input_csv).resolve() if (data_path or input_csv) else processed_dir / "daily_category_demand.csv"
    if not src_csv.exists():
        src_csv = INPUT_CSV

    if not src_csv.exists():
        raise RuntimeError(
            f"Prerequisite dataset missing: 'daily_category_demand.csv' not found at: {src_csv}\n"
            f"Cannot run Demand Analytics layer without ETL outputs. Please run 'optera.etl(...)' first."
        )
    input_csv = src_csv

    # 1. Load pipeline metadata for validation/logging
    meta_json = workspace / "data" / "processed" / "metadata.json"
    if not meta_json.exists():
        meta_json = METADATA_JSON

    if meta_json.exists():
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            logger.info("Loaded ETL pipeline metadata. Execution timestamp: %s", meta.get("execution_timestamp"))
            logger.info("Dataset dimensions: %s", meta.get("dataset", {}).get("dimensions"))
        except Exception as e:
            logger.warning("Failed to parse pipeline metadata JSON: %s", e)
    else:
        logger.warning("Pipeline metadata.json not found at: %s", METADATA_JSON)

    # 2. Run Category Demand Profiling & Distribution Fitting
    t0 = time.time()
    logger.info("Running Category Demand Profiler (Confidence levels: %s, Distributions: %s, Metric: %s)...",
                confidence_levels or [0.95, 0.99], distributions or ["normal", "poisson", "gamma", "nbinom"], goodness_of_fit_metric.upper())

    profiler = CategoryDemandProfiler(
        data=input_csv,
        confidence_levels=confidence_levels,
        distributions=distributions,
        goodness_of_fit_metric=goodness_of_fit_metric
    )
    summary_df, detailed_profiles = profiler.profile_all_categories()
    
    # 3. Export Structured Demand Model JSON for Optimization Layer
    demand_model_file = analytics_data_dir / "demand_model.json"
    cov_dict, corr_dict = profiler.compute_covariance_matrices()
    export_demand_model(detailed_profiles, demand_model_file, cov_dict=cov_dict, corr_dict=corr_dict)

    # 4. Generate Distribution Fitting & Q-Q Plots
    logger.info("Generating Category Demand Distribution Fitting Plots...")
    reporter = AnalyticsReporter(reports_dir=reports_dir, pdfs_dir=pdfs_dir, plots_dir=plots_dir)
    plot_paths = reporter.generate_plots(profiler.df, detailed_profiles)

    # 5. Compile Demand Analytical Report (MD & PDF)
    logger.info("Compiling Demand Analytical Report (MD & PDF)...")
    report_file, pdf_file = reporter.generate_report(summary_df, detailed_profiles, plot_paths)
    duration = time.time() - t0

    logger.info("Category demand modeling complete in %.3fs.", duration)
    logger.info("=" * 70)
    logger.info("OPTERA DEMAND ANALYTICS LAYER EXECUTED SUCCESSFULLY!")
    logger.info("Demand Markdown Report -> %s", report_file)
    logger.info("Demand Executive PDF   -> %s", pdf_file)
    logger.info("Demand Model JSON      -> %s", demand_model_file)
    logger.info("Distribution Plots     -> %s", PLOTS_DIR)
    logger.info("=" * 70)

    # Print Report Summary to Console
    print("\n" + "=" * 70)
    print("OPTERA DEMAND ANALYTICS LAYER OUTPUTS GENERATED")
    print("=" * 70)
    print(f"  • Markdown Report   : {report_file}")
    print(f"  • Executive PDF     : {pdf_file}")
    print(f"  • Demand Model JSON : {demand_model_file}")
    print(f"  • Distribution Plots: {PLOTS_DIR}")
    print("=" * 70 + "\n")

    return report_file


if __name__ == "__main__":
    run_analytics()
