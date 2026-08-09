"""
Optera Master Executive Report Compiler & Consolidator (v2.0)

Merges all four operational layer reports:
1. Layer 1: ETL Data Quality & Feature Engineering Report
2. Layer 2: Demand Analytical Distribution Fitting Report
3. Layer 3: Particle Swarm Portfolio Optimization Report
4. Layer 4: Event-Driven Monte Carlo Simulation Execution Report

Generates:
- `output_dir/optera_report.md`
- `output_dir/optera_report.pdf`
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

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

        # Normalize image URIs into clean relative plot paths for Markdown rendering
        def clean_img_paths(match):
            alt_text = match.group(1)
            raw_url = match.group(2)
            if "plots/" in raw_url:
                plot_filename = raw_url.split("plots/")[-1]
                return f"![{alt_text}](plots/{plot_filename})"
            return match.group(0)

        master_content = re.sub(r"!\[(.*?)\]\((.*?)\)", clean_img_paths, master_content)

        with open(master_md_file, "w", encoding="utf-8") as f:
            f.write(master_content)

        logger.info("Compiled Integrated Master Markdown Report -> %s", master_md_file)

        # Build Executive PDF Report
        try:
            pdf_builder = OpteraPDFBuilder(
                title="Optera Quantitative Supply Chain Framework",
                subtitle="Master Executive Report & End-to-End Analytics"
            )
            pdf_builder.add_markdown(master_content, workspace_dir=self.workspace_dir)
            pdf_builder.build(master_pdf_file)
            logger.info("Compiled Executive Master PDF Report -> %s", master_pdf_file)
        except Exception as e:
            logger.error("Failed to generate Master PDF report: %s", e)

        return master_md_file
