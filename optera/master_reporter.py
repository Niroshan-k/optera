"""
Optera Master Report Compiler & Consolidator (v2.0)

Merges all three operational layer reports:
1. Layer 1: ETL & Demand Analytical Distribution Fitting Report
2. Layer 2: Particle Swarm Portfolio Optimization Report
3. Layer 3: Event-Driven Monte Carlo Simulation Execution Report

Generates:
- `output_dir/reports/optera_master_report.md`
- `output_dir/pdfs/optera_master_report.pdf`
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from optera.utils.pdf_generator import OpteraPDFBuilder

logger = logging.getLogger("OpteraMasterReporter")


class MasterReporter:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.reports_dir = self.workspace_dir / "reports"
        self.pdfs_dir = self.workspace_dir / "pdfs"
        self.plots_dir = self.workspace_dir / "plots"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)

    def compile_master_report(
        self,
        etl_report_file: Optional[Path] = None,
        opt_report_file: Optional[Path] = None,
        sim_report_file: Optional[Path] = None,
        sim_results: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Consolidates all layer Markdown reports into a single publication-ready Master Report."""
        master_md_file = self.workspace_dir / "optera_report.md"
        master_pdf_file = self.workspace_dir / "optera_report.pdf"

        # Default fallback report paths if not explicitly passed
        if etl_report_file is None or not etl_report_file.exists():
            etl_report_file = self.reports_dir / "etl_report.md"
            if not etl_report_file.exists():
                etl_report_file = self.reports_dir / "distribution_fitting_report.md"

        analytics_report_file = self.reports_dir / "demand_analytical_report.md"

        if opt_report_file is None or not opt_report_file.exists():
            opt_report_file = self.reports_dir / "optimization_report.md"
        if sim_report_file is None or not sim_report_file.exists():
            sim_report_file = self.reports_dir / "simulation_report.md"

        def read_content(path: Path) -> str:
            if path and path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            return f"*Report file not found at: {path}*\n"

        etl_content = read_content(etl_report_file)
        analytics_content = read_content(analytics_report_file) if analytics_report_file.exists() else ""
        opt_content = read_content(opt_report_file)
        sim_content = read_content(sim_report_file)

        master_content = f"""# OPTERA QUANTITATIVE SUPPLY CHAIN FRAMEWORK - MASTER EXECUTIVE REPORT

> **Comprehensive End-to-End Analysis**: Data Cleaning & Feature Engineering, Statistical Demand Profiling, Portfolio PSO Optimization, and Monte Carlo Event-Driven Simulation.

---

# LAYER 1: ETL DATA QUALITY & FEATURE ENGINEERING

{etl_content}

---

# LAYER 2: STATISTICAL DEMAND ANALYTICAL PROFILING

{analytics_content}

---

# LAYER 3: PORTFOLIO OPTIMIZATION & CAPITAL ALLOCATION

{opt_content}

---

# LAYER 4: MONTE CARLO EVENT-DRIVEN SIMULATION

{sim_content}

---

## Master Report Summary & Sign-off

- **Generated Workspace**: `{self.workspace_dir}`
- **PDF Artifact**: `{master_pdf_file}`
- **Markdown Master Artifact**: `{master_md_file}`
"""

        # Normalize file:/// image URIs with spaces into clean relative plot paths for Markdown compatibility
        import re
        def clean_img_paths(match):
            alt_text = match.group(1)
            raw_url = match.group(2)
            if "plots/" in raw_url:
                plot_name = raw_url.split("plots/")[-1]
                return f"![{alt_text}](plots/{plot_name})"
            return match.group(0)

        master_content_cleaned = re.sub(r"!\[(.*?)\]\((.*?)\)", clean_img_paths, master_content)

        with open(master_md_file, "w", encoding="utf-8") as f:
            f.write(master_content_cleaned)

        logger.info("Compiled Integrated Master Markdown Report -> %s", master_md_file)

        # Build PDF Master Report
        try:
            pdf_builder = OpteraPDFBuilder(
                title="Optera Integrated Supply Chain Executive Master Report",
                subtitle="End-to-End Demand Analytics, Portfolio Optimization & Monte Carlo Simulation"
            )

            pdf_builder.add_markdown(master_content_cleaned, workspace_dir=self.workspace_dir)
            pdf_builder.build(master_pdf_file)
            logger.info("Compiled Executive Master PDF Report -> %s", master_pdf_file)
        except Exception as e:
            logger.warning("PDF Master generation notice: %s", e)

        return master_md_file
