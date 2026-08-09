"""
Optera Analytics Reporter & Visualization Engine

Generates crisp, quantitative distribution fitting plots and compiles the final
reports saved as:
• Markdown Report: optera/reports/demand_analytical_report.md
• Executive PDF Document: optera/pdfs/demand_analytical_report.pdf
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Ensure optera root is in path for utils import
optera_root = Path(__file__).resolve().parent.parent.parent.parent
if str(optera_root) not in sys.path:
    sys.path.insert(0, str(optera_root))

from optera.utils.pdf_generator import OpteraPDFBuilder

# Monospace quant styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "monospace"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

logger = logging.getLogger("OpteraAnalyticsReporter")


def to_rel_path(target: Path, start: Path) -> str:
    """Computes POSIX relative path for Markdown link parsing."""
    try:
        return os.path.relpath(target.resolve(), start.resolve()).replace("\\", "/")
    except Exception:
        return str(target)


class AnalyticsReporter:
    """
    Decoupled Analytics Reporting and Visualization Module.
    """

    def __init__(self, reports_dir: Optional[Path] = None, plots_dir: Optional[Path] = None, pdfs_dir: Optional[Path] = None):
        optera_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.reports_dir = Path(reports_dir) if reports_dir is not None else optera_dir / "reports"
        self.plots_dir = Path(plots_dir) if plots_dir is not None else optera_dir / "plots"
        self.pdfs_dir = Path(pdfs_dir) if pdfs_dir is not None else optera_dir / "pdfs"

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)

    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        """Converts a pandas DataFrame to a markdown table string."""
        headers = list(df.columns)
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_rows = []
        for _, row in df.iterrows():
            formatted_vals = []
            for val in row:
                if isinstance(val, float):
                    formatted_vals.append(f"{val:,.2f}")
                elif isinstance(val, int):
                    formatted_vals.append(f"{val:,}")
                else:
                    formatted_vals.append(str(val))
            body_rows.append("| " + " | ".join(formatted_vals) + " |")
        return "\n".join([header_row, separator_row] + body_rows)

    def generate_plots(self, raw_demand_df: pd.DataFrame, detailed_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Path]:
        """
        Generates distribution fitting and Q-Q plots for each category.
        """
        plot_paths = {}

        for cat, prof in detailed_profiles.items():
            cat_data = raw_demand_df[raw_demand_df["category"] == cat]["quantity"].values
            if len(cat_data) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

            # Subplot 1: Density & Fitted Distributions
            sns.histplot(
                cat_data,
                stat="density",
                kde=True,
                ax=ax1,
                color="#1f77b4",
                alpha=0.35,
                edgecolor="white",
                label="Empirical Demand"
            )

            x_axis = np.linspace(min(cat_data), max(cat_data), 200)
            best_dist_name = prof["best_fit"]["name"]
            fitted_dists = prof["fitted_distributions"]

            if "normal" in fitted_dists:
                mu_n = fitted_dists["normal"]["params"]["mu"]
                std_n = fitted_dists["normal"]["params"]["sigma"]
                pdf_n = stats.norm.pdf(x_axis, loc=mu_n, scale=std_n)
                is_best = ("Normal" in best_dist_name)
                ax1.plot(
                    x_axis,
                    pdf_n,
                    "r--" if is_best else "gray",
                    linewidth=2.2 if is_best else 1.0,
                    alpha=1.0 if is_best else 0.5,
                    label=f"Normal Fit {'(Best)' if is_best else ''}"
                )

            if "gamma" in fitted_dists:
                shape_g = fitted_dists["gamma"]["params"]["shape"]
                scale_g = fitted_dists["gamma"]["params"]["scale"]
                pdf_g = stats.gamma.pdf(x_axis, shape_g, loc=0, scale=scale_g)
                is_best = ("Gamma" in best_dist_name)
                ax1.plot(
                    x_axis,
                    pdf_g,
                    "g--" if is_best else "green",
                    linewidth=2.2 if is_best else 1.0,
                    alpha=1.0 if is_best else 0.5,
                    label=f"Gamma Fit {'(Best)' if is_best else ''}"
                )

            if "poisson" in fitted_dists:
                lam_p = fitted_dists["poisson"]["params"]["lambda"]
                x_disc = np.unique(np.round(x_axis).astype(int))
                pmf_p = stats.poisson.pmf(x_disc, mu=lam_p)
                is_best = ("Poisson" in best_dist_name)
                ax1.plot(
                    x_disc,
                    pmf_p,
                    "m--" if is_best else "purple",
                    linewidth=2.2 if is_best else 1.0,
                    alpha=1.0 if is_best else 0.5,
                    label=f"Poisson Fit {'(Best)' if is_best else ''}"
                )

            if "nbinom" in fitted_dists:
                n_nb = fitted_dists["nbinom"]["params"]["n"]
                p_nb = fitted_dists["nbinom"]["params"]["p"]
                x_disc = np.unique(np.round(x_axis).astype(int))
                pmf_nb = stats.nbinom.pmf(x_disc, n=n_nb, p=p_nb)
                is_best = ("Negative Binomial" in best_dist_name)
                ax1.plot(
                    x_disc,
                    pmf_nb,
                    "c--" if is_best else "orange",
                    linewidth=2.2 if is_best else 1.0,
                    alpha=1.0 if is_best else 0.5,
                    label=f"NegBinomial Fit {'(Best)' if is_best else ''}"
                )

            ax1.set_title(f"Category: {cat} - Distribution Fitting", fontsize=10.5, fontweight="bold")
            ax1.set_xlabel("Daily Demand Quantity", fontsize=8.5)
            ax1.set_ylabel("Density", fontsize=8.5)
            ax1.legend(fontsize=7.5, loc="upper right")

            # Subplot 2: Q-Q Plot
            stats.probplot(cat_data, dist="norm", plot=ax2)
            ax2.set_title(f"Category: {cat} - Normal Q-Q Plot", fontsize=10.5, fontweight="bold")
            ax2.set_xlabel("Theoretical Quantiles", fontsize=8.5)
            ax2.set_ylabel("Ordered Values", fontsize=8.5)
            ax2.get_lines()[0].set_color("#1f77b4")
            ax2.get_lines()[0].set_markersize(4)
            ax2.get_lines()[1].set_color("red")
            ax2.get_lines()[1].set_linewidth(1.5)

            plt.tight_layout()
            plot_file = self.plots_dir / f"demand_fit_{cat}.png"
            fig.savefig(plot_file, bbox_inches="tight")
            plt.close(fig)

            plot_paths[cat] = plot_file

        return plot_paths

    def generate_report(
        self,
        summary_df: pd.DataFrame,
        detailed_profiles: Dict[str, Dict[str, Any]],
        plot_paths: Optional[Dict[str, Path]] = None
    ) -> Tuple[Path, Path]:
        """
        Compiles both demand_analytical_report.md and demand_analytical_report.pdf.
        """
        report_file = self.reports_dir / "demand_analytical_report.md"
        pdf_file = self.pdfs_dir / "demand_analytical_report.pdf"

        table_md = self._df_to_markdown(summary_df)

        lines = [
            "# Optera Category Demand Analytical & Modeling Report",
            "",
            "> *Institutional-grade statistical demand profiling, probability distribution fitting (Normal, Poisson, Gamma, NegBinomial), AIC/BIC goodness-of-fit selection, confidence intervals, and pattern detection.*",
            "",
            "---",
            "",
            "## 1. Executive Portfolio Statistical Profile",
            "",
            table_md,
            "",
            "---",
            "",
            "## 2. Best Distribution Selection & Goodness-of-Fit Summary",
            ""
        ]

        gof_rows = []
        for cat, prof in detailed_profiles.items():
            best_info = prof["best_fit"]
            gof_rows.append({
                "Category": cat,
                "Best Model": best_info["name"],
                "Criterion": best_info["metric"],
                "AIC Score": round(prof["fitted_distributions"].get(best_info["key"], {}).get("aic", 0.0), 2),
                "BIC Score": round(prof["fitted_distributions"].get(best_info["key"], {}).get("bic", 0.0), 2),
                "Regime": prof["pattern_detection"]["regime_summary"]
            })
        gof_df = pd.DataFrame(gof_rows)
        lines.extend([
            self._df_to_markdown(gof_df),
            "",
            "---",
            "",
            "## 3. Detailed Category Profiling & Distribution Fits",
            ""
        ])

        for cat, prof in detailed_profiles.items():
            rel_plot = to_rel_path(plot_paths[cat], self.reports_dir) if plot_paths and cat in plot_paths else ""
            ci_str_list = [f"`{k}`: `[{v['lower']:.2f}, {v['upper']:.2f}]`" for k, v in prof["confidence_intervals"].items()]
            ci_formatted = ", ".join(ci_str_list)

            lines.extend([
                f"### Category: `{cat}`",
                "",
                f"- **Observations**: `{prof['observations']:,}` days",
                f"- **Total Quantity**: `{prof['total_quantity']:,.0f}` units",
                f"- **Central Tendency**: Mean = `{prof['mean']:.2f}`, Median = `{prof['median']:.2f}`",
                f"- **Volatility & Spread**: Std = `{prof['std']:.2f}`, Variance = `{prof['variance']:.2f}`, CV = `{prof['cv']:.2f}`, Min = `{prof['min']:.2f}`, Max = `{prof['max']:.2f}`, IQR = `[{prof['p25']:.2f}, {prof['p75']:.2f}]`",
                f"- **Higher Moments**: Skewness = `{prof['skewness']:.2f}`, Kurtosis = `{prof['kurtosis']:.2f}`",
                f"- **Mean Confidence Intervals**: {ci_formatted}",
                f"- **Statistically Best Distribution Fit**: **`{prof['best_fit']['name']}`** (Score = `{prof['best_fit']['score']:.2f}`)",
                f"- **Demand Pattern & Regime**: `{prof['pattern_detection']['regime_summary']}`",
                ""
            ])

            if rel_plot:
                lines.extend([
                    f"![Distribution Fit for {cat}]({rel_plot})",
                    ""
                ])

            fit_rows = []
            for dkey, dinfo in prof["fitted_distributions"].items():
                params_str = ", ".join([f"{k}={v:.2f}" for k, v in dinfo["params"].items()])
                fit_rows.append({
                    "Distribution": dinfo["name"],
                    "Parameters": params_str,
                    "Log-Likelihood": round(dinfo["log_likelihood"], 2),
                    "AIC": round(dinfo["aic"], 2),
                    "BIC": round(dinfo["bic"], 2),
                    "Is Best": "YES" if dkey == prof["best_fit"]["key"] else "No"
                })
            fit_df = pd.DataFrame(fit_rows)

            lines.extend([
                "**Candidate Distribution Goodness-of-Fit Comparison:**",
                "",
                self._df_to_markdown(fit_df),
                "",
                f"**Financials:** Mean Price = `${prof['financials']['unit_price_mean']:.2f}`, Mean Cost = `${prof['financials']['unit_cost_mean']:.2f}`, Gross Margin = `{prof['financials']['margin_pct']:.1f}%`, Total Revenue = `${prof['financials']['total_revenue']:,.2f}`",
                "",
                "---",
                ""
            ])

        report_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved Demand Analytical Report -> %s", report_file)

        # -------------------------------------------------------------
        # Compile PDF Document via OpteraPDFBuilder
        # -------------------------------------------------------------
        try:
            pdf_builder = OpteraPDFBuilder(
                title="Optera Category Demand Analytical & Modeling Report",
                subtitle="Institutional-grade statistical demand profiling, probability distribution fitting, AIC/BIC selection, and pattern detection."
            )

            pdf_builder.add_heading("1. Executive Portfolio Statistical Profile", level=1)
            pdf_builder.add_table(summary_df)

            pdf_builder.add_heading("2. Best Distribution Selection & Goodness-of-Fit Summary", level=1)
            pdf_builder.add_table(gof_df)

            pdf_builder.add_heading("3. Detailed Category Profiling & Distribution Fits", level=1)
            for cat, prof in detailed_profiles.items():
                pdf_builder.add_heading(f"Category: {cat}", level=2)
                pdf_builder.add_bullet("Central Tendency", f"Mean = {prof['mean']:.2f}, Median = {prof['median']:.2f}")
                pdf_builder.add_bullet("Volatility & Spread", f"Std = {prof['std']:.2f}, Variance = {prof['variance']:.2f}, CV = {prof['cv']:.2f}")
                pdf_builder.add_bullet("95% Mean Confidence Interval", f"[{prof['confidence_intervals']['95%']['lower']:.2f}, {prof['confidence_intervals']['95%']['upper']:.2f}]")
                pdf_builder.add_bullet("Statistically Best Model", f"{prof['best_fit']['name']} (AIC/BIC Score = {prof['best_fit']['score']:.2f})")
                pdf_builder.add_bullet("Demand Pattern Regime", prof["pattern_detection"]["regime_summary"])

                if plot_paths and cat in plot_paths:
                    pdf_builder.add_image(plot_paths[cat])

                # Fit comparison table
                fit_rows = []
                for dkey, dinfo in prof["fitted_distributions"].items():
                    params_str = ", ".join([f"{k}={v:.2f}" for k, v in dinfo["params"].items()])
                    fit_rows.append({
                        "Distribution": dinfo["name"],
                        "Parameters": params_str,
                        "Log-Likelihood": round(dinfo["log_likelihood"], 2),
                        "AIC": round(dinfo["aic"], 2),
                        "BIC": round(dinfo["bic"], 2),
                        "Is Best": "YES" if dkey == prof["best_fit"]["key"] else "No"
                    })
                pdf_builder.add_table(pd.DataFrame(fit_rows))
                pdf_builder.add_bullet("Financials", f"Mean Price = ${prof['financials']['unit_price_mean']:.2f}, Mean Cost = ${prof['financials']['unit_cost_mean']:.2f}, Gross Margin = {prof['financials']['margin_pct']:.1f}%")

            pdf_builder.build(pdf_file)
        except Exception as e:
            logger.warning("Failed to generate Analytics PDF report: %s", e)

        return report_file, pdf_file
