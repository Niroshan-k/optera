"""
Optera Optimization Reporter & Visualizer Module (v1.9)

Generates 6 high-resolution monospace visual plots, Markdown report (optimization_report.md),
and Executive PDF report (optimization_report.pdf).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from optera.utils.pdf_generator import OpteraPDFBuilder

logger = logging.getLogger("OpteraOptimizerReporter")

# Set Monospace aesthetic visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono", "Consolas", "Courier New"]
plt.rcParams["font.size"] = 9.5
plt.rcParams["axes.titlesize"] = 11.5
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["figure.titlesize"] = 13.0


class OptimizerReporter:
    """
    Generates optimization plots, Markdown report, and Executive PDF.
    """

    def __init__(
        self,
        reports_dir: Union[str, Path],
        pdfs_dir: Union[str, Path],
        plots_dir: Union[str, Path]
    ):
        self.reports_dir = Path(reports_dir)
        self.pdfs_dir = Path(pdfs_dir)
        self.plots_dir = Path(plots_dir)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(
        self,
        results: Dict[str, Any],
        stability_res: Dict[str, Any],
        frontier_res: Dict[str, Any],
        demand_model_dict: Dict[str, Any]
    ) -> Dict[str, Path]:
        """
        Generates 6 high-resolution monospace visual evaluation charts.
        """
        plot_paths = {}

        # 1. Efficient Frontier Scatter Plot
        plot_paths["efficient_frontier"] = self.plot_efficient_frontier(frontier_res, results)

        # 2. Correlation Heatmap Plot
        plot_paths["correlation_heatmap"] = self.plot_correlation_heatmap(demand_model_dict)

        # 3. Risk vs Profit Scatter Plot
        plot_paths["risk_profit_scatter"] = self.plot_risk_profit_scatter(demand_model_dict, results)

        # 4. PSO Convergence Curve Plot
        plot_paths["pso_convergence"] = self.plot_convergence_curve(results)

        # 5. PSO Independent Stability Distribution Plot
        plot_paths["stability_distribution"] = self.plot_stability_distribution(stability_res)

        # 6. Budget Allocation Donut Chart
        plot_paths["budget_allocation"] = self.plot_budget_allocation(results)

        return plot_paths

    def plot_efficient_frontier(self, frontier_res: Dict[str, Any], results: Dict[str, Any]) -> Path:
        """
        Plot 1: Markowitz Efficient Frontier Scatter Cloud colored by PES (Procurement Efficiency Score)
        Matching user reference image style with supply chain terminology & monospace fonts.
        """
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

        # 1. Extract Candidate Portfolios evaluated across PSO iterations
        candidates = results.get("candidate_portfolios", [])
        if not candidates:
            # Fallback if candidates not populated
            risks_std = [p["portfolio_std_dev"] for p in frontier_res["points"]]
            profits = [p["profit"] for p in frontier_res["points"]]
            pes_vals = [p["pes"] for p in frontier_res["points"]]
        else:
            risks_std = [c["risk_std"] for c in candidates]
            profits = [c["profit"] for c in candidates]
            pes_vals = [c["pes"] for c in candidates]

        risks_std = np.array(risks_std)
        profits = np.array(profits)
        pes_vals = np.array(pes_vals)

        # Scatter Cloud colored by PES (viridis colormap)
        scatter = ax.scatter(
            risks_std,
            profits,
            c=pes_vals,
            cmap="viridis",
            s=22,
            alpha=0.65,
            edgecolors="none"
        )

        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Procurement Efficiency Score (PES = Profit / Risk)", fontsize=9, fontweight="bold")

        # 2. Upper Efficient Frontier Curve
        fr_risks = np.array([p["portfolio_std_dev"] for p in frontier_res["points"]])
        fr_profits = np.array([p["profit"] for p in frontier_res["points"]])
        ax.plot(fr_risks, fr_profits, color="#2c3e50", linestyle="--", linewidth=1.8, label="Efficient Frontier Boundary")

        # 3. Highlight Optimal Max-PES Portfolio (Red Marker)
        opt_risk = results["portfolio_std_dev"]
        opt_profit = results["expected_gross_profit"]
        opt_pes = results["portfolio_pes"]

        ax.scatter([opt_risk], [opt_profit], color="#e74c3c", s=140, zorder=6, edgecolors="black", linewidth=1.5, label="Optimal Portfolio (Max PES)")
        ax.annotate(
            f" Optimal PSO Target\n (Profit: ${opt_profit:,.0f}, Risk: {opt_risk:.0f}, PES: {opt_pes:.2f})",
            xy=(opt_risk, opt_profit),
            xytext=(opt_risk + 10, opt_profit + 500),
            fontsize=8.5,
            fontweight="bold",
            color="#c0392b",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2)
        )

        # 4. Highlight Minimum Demand Risk Portfolio (Blue Marker)
        min_idx = np.argmin(fr_risks)
        min_risk = fr_risks[min_idx]
        min_prof = fr_profits[min_idx]

        ax.scatter([min_risk], [min_prof], color="#2980b9", s=140, zorder=6, edgecolors="black", linewidth=1.5, label="Minimum Risk Portfolio")
        ax.annotate(
            f" Min Risk Portfolio\n (Risk: {min_risk:.0f}, Profit: ${min_prof:,.0f})",
            xy=(min_risk, min_prof),
            xytext=(min_risk - 60 if min_risk > 100 else min_risk + 10, min_prof - 1500),
            fontsize=8.5,
            fontweight="bold",
            color="#1b4f72",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4e6f1", edgecolor="#2980b9", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.2)
        )

        ax.set_title("Markowitz Efficient Frontier: Procurement Profit vs Portfolio Demand Risk", pad=15)
        ax.set_xlabel("Portfolio Demand Risk (Standard Deviation - σ_portfolio)")
        ax.set_ylabel("Expected Procurement Gross Profit ($)")
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

        output_path = self.plots_dir / "optimization_efficient_frontier.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_correlation_heatmap(self, demand_model_dict: Dict[str, Any]) -> Path:
        """Plot 2: Category Demand Correlation Matrix Heatmap"""
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

        meta = demand_model_dict.get("metadata", {})
        corr_json = meta.get("correlation_matrix", {})

        categories = [k for k in demand_model_dict.keys() if k != "metadata"]
        dim = len(categories)

        corr_matrix = np.eye(dim)
        if corr_json:
            for i, c1 in enumerate(categories):
                for j, c2 in enumerate(categories):
                    corr_matrix[i, j] = corr_json.get(c1, {}).get(c2, 1.0 if i == j else 0.0)

        cat_labels = [demand_model_dict[cat]["category"] for cat in categories]

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-1.0,
            vmax=1.0,
            xticklabels=cat_labels,
            yticklabels=cat_labels,
            cbar_kws={"label": "Correlation Coefficient (r)"},
            ax=ax
        )
        ax.set_title("Category Demand Correlation Matrix (R)", pad=15)
        plt.xticks(rotation=30, ha="right")

        output_path = self.plots_dir / "optimization_correlation_heatmap.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_risk_profit_scatter(self, demand_model_dict: Dict[str, Any], results: Dict[str, Any]) -> Path:
        """Plot 3: Category Standalone Risk vs Expected Profit Scatter Plot"""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        categories = [k for k in demand_model_dict.keys() if k != "metadata"]

        cat_names = []
        profits = []
        stds = []
        weights = []

        for cat in categories:
            info = demand_model_dict[cat]
            name = info["category"]
            p = info["mean"] * (info["financials"]["unit_price_mean"] - info["financials"]["unit_cost_mean"])
            s = info["std"]
            w = results["allocations_percentage"].get(cat, 0.0)

            cat_names.append(name)
            profits.append(p)
            stds.append(s)
            weights.append(w)

        # Scatter individual categories (bubble size = weight)
        sizes = [max(120, w * 28) for w in weights]
        scatter = ax.scatter(stds, profits, s=sizes, c="#27ae60", alpha=0.75, edgecolors="black", linewidth=1.2, label="Standalone Categories")

        for i, txt in enumerate(cat_names):
            ax.annotate(f"{txt}\n({weights[i]:.1f}%)", (stds[i], profits[i]), fontsize=8.5, fontweight="bold", ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")

        # Plot Combined Portfolio Point
        opt_risk = results["portfolio_std_dev"]
        opt_profit = results["expected_gross_profit"]
        ax.scatter([opt_risk], [opt_profit], s=350, c="#e74c3c", marker="P", edgecolors="black", linewidth=1.5, zorder=5, label="Optimized Portfolio")
        ax.annotate(f"Combined Portfolio\n(PES={results['portfolio_pes']:.2f})", (opt_risk, opt_profit), fontsize=9, fontweight="bold", color="#c0392b", ha="center", va="bottom", xytext=(0, 8), textcoords="offset points")

        ax.set_title("Category Standalone Demand Risk vs Profit & Optimized Target", pad=15)
        ax.set_xlabel("Demand Volatility / Risk (σ_i units)")
        ax.set_ylabel("Expected Gross Profit ($)")
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

        output_path = self.plots_dir / "optimization_risk_profit_scatter.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_convergence_curve(self, results: Dict[str, Any]) -> Path:
        """
        Plot 4: Crisp PSO Convergence Curve displaying Global Best Fitness AND Swarm Average Fitness,
        marking Convergence Speed (Iteration index).
        """
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

        gbest_history = results.get("convergence_history", [])
        mean_history = results.get("mean_swarm_history", [])
        iterations = np.arange(1, len(gbest_history) + 1)

        # Plot Swarm Mean Fitness
        if mean_history and len(mean_history) == len(gbest_history):
            ax.plot(iterations, mean_history, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Swarm Mean Fitness")

        # Plot Global Best Fitness
        ax.plot(iterations, gbest_history, color="#2980b9", linewidth=2.5, label="Global Best Fitness F(w*)")

        # Highlight Convergence Speed
        conv_iter = results.get("convergence_iteration", len(gbest_history))
        if conv_iter <= len(gbest_history):
            conv_val = gbest_history[conv_iter - 1]
            ax.axvline(x=conv_iter, color="#e74c3c", linestyle=":", linewidth=1.8, label=f"Convergence Speed (Iter {conv_iter})")
            ax.plot(conv_iter, conv_val, "o", color="#e74c3c", markersize=8, zorder=5)

            ax.annotate(
                f" 99% Max Fitness\n (Iter {conv_iter}: {conv_val:,.1f})",
                xy=(conv_iter, conv_val),
                xytext=(conv_iter + 8, conv_val - 1200 if conv_val > 5000 else conv_val + 500),
                fontsize=8.5,
                fontweight="bold",
                color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2)
            )

        ax.set_title("Particle Swarm Optimization Fitness Convergence Curve", pad=15)
        ax.set_xlabel("Swarm Iteration Index")
        ax.set_ylabel("Portfolio Fitness Score F(w)")
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

        output_path = self.plots_dir / "optimization_pso_convergence.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_stability_distribution(self, stability_res: Dict[str, Any]) -> Path:
        """Plot 5: PSO Independent Stability Distribution Plot"""
        fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}, dpi=300)

        fits = stability_res["all_fitnesses"]
        mean_val = stability_res["mean_fitness"]
        ci = stability_res["fitness_confidence_interval"]

        # Boxplot
        ax_box.boxplot(fits, vert=False, patch_artist=True, boxprops=dict(facecolor="#3498db", color="#2980b9"), medianprops=dict(color="#e74c3c", linewidth=2))
        ax_box.set_title(f"PSO Stability Analysis Across {stability_res['num_runs']} Independent Runs", pad=10)
        ax_box.set_yticks([])

        # Histogram & Density
        sns.histplot(fits, kde=True, ax=ax_hist, color="#3498db", bins=12, stat="density", line_kws={"linewidth": 2})

        ax_hist.axvline(mean_val, color="#e74c3c", linestyle="-", linewidth=2, label=f"Mean Fitness: {mean_val:,.2f}")
        ax_hist.axvspan(ci[0], ci[1], color="#2ecc71", alpha=0.2, label=f"95% CI: [{ci[0]:,.2f}, {ci[1]:,.2f}]")

        ax_hist.set_xlabel("Portfolio Fitness Score F(w)")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(loc="upper left", frameon=True)
        ax_hist.grid(True, linestyle=":", alpha=0.6)

        output_path = self.plots_dir / "optimization_stability_distribution.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_budget_allocation(self, results: Dict[str, Any]) -> Path:
        """Plot 6: Category Budget Allocation Donut Chart"""
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

        allocs = results["allocations_percentage"]
        labels = list(allocs.keys())
        values = list(allocs.values())

        colors = ["#2ecc71", "#3498db", "#9b59b6", "#f1c40f", "#e67e22", "#e74c3c"]
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors[:len(values)],
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
            pctdistance=0.75
        )

        for autotext in autotexts:
            autotext.set_color("black")
            autotext.set_fontweight("bold")

        ax.set_title(f"Optimal Procurement Budget Allocation Vector (w*)\n(ENC = {results['effective_number_categories']:.2f} Categories)", pad=15)

        output_path = self.plots_dir / "optimization_budget_allocation.png"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        stability_res: Dict[str, Any],
        frontier_res: Dict[str, Any],
        plot_paths: Dict[str, Path]
    ) -> Path:
        """
        Compiles optera/reports/optimization_report.md.
        """
        md_file = self.reports_dir / "optimization_report.md"

        alloc_rows = ""
        for cat, pct in results["allocations_percentage"].items():
            w = results["allocations_weight"][cat]
            alloc_rows += f"| **{cat.capitalize()}** | **{pct:.2f}%** | `{w:.4f}` |\n"

        p_profit = results['portfolio_expected_profit']
        p_pes = results['portfolio_pes']
        p_var = results['portfolio_variance']
        p_std = results['portfolio_std_dev']
        p_hhi = results['herfindahl_index']
        p_enc = results['effective_number_categories']
        opt_fit = results['optimal_fitness']
        conv_it = results['convergence_iteration']
        stab_cv = stability_res['stability_cv']
        ci_0 = stability_res['fitness_confidence_interval'][0]
        ci_1 = stability_res['fitness_confidence_interval'][1]
        n_runs = stability_res['num_runs']
        m_fit = stability_res['mean_fitness']
        s_fit = stability_res['std_fitness']
        best_f = stability_res.get('best_fitness', opt_fit)
        runtime = stability_res.get('execution_time_seconds', results.get('execution_time_seconds', 0.0))

        md_content = f"""# Optera Optimization Evaluation & Portfolio Allocation Report

**Algorithm**: {results['algorithm']}  
**Evaluation Date**: 2026-07-26  
**Decision Space**: {results['dimension']} Product Categories  
**Execution Runtime**: {runtime:.3f} seconds  

---

## 1. Executive Portfolio Optimization Summary

| Metric Name | Value | Description |
| :--- | :---: | :--- |
| **Portfolio Expected Gross Profit** | **${p_profit:,.2f}** | Total expected dollar revenue margin |
| **Procurement Efficiency Score (PES)** | **{p_pes:.4f}** | Profit per unit of demand risk ($\text{{Profit}} / \sigma_{{\text{{portfolio}}}}$) |
| **Portfolio Demand Variance ($\mathbf{{w}}^T \mathbf{{\Sigma}} \mathbf{{w}}$)** | **{p_var:,.2f}** | Total quadratic portfolio demand risk |
| **Portfolio Demand Std Dev ($\sigma_{{\text{{portfolio}}}}$)** | **{p_std:,.2f} units** | Standard deviation of portfolio demand |
| **Herfindahl Concentration Index ($HHI$)** | **{p_hhi:.4f}** | Portfolio market concentration measure |
| **Effective Number of Categories ($ENC$) / Diversification Ratio ($DR$)** | **{p_enc:.2f} categories** | $DR = 1 / HHI$ (Effective funded categories) |
| **Optimal Portfolio Fitness $F(\mathbf{{w}}^*)$** | **{opt_fit:,.2f}** | Maximum objective fitness score achieved (unitless) |
| **Convergence Speed** | **Iteration {conv_it}** | Iteration index reaching 99% max fitness |

---

## 2. Mathematical Objective Function

$$F(\mathbf{{w}}) = \sum_{{i=1}}^N w_i \cdot \mu_i \cdot (p_i - c_i) - \lambda \cdot (\mathbf{{w}}^T \mathbf{{\Sigma}} \mathbf{{w}}) - \gamma \sum_{{i=1}}^N w_i^2$$

Where:
- $\lambda = {results['lambda_risk']}$: Risk aversion parameter for portfolio demand variance $\mathbf{{w}}^T \mathbf{{\Sigma}} \mathbf{{w}}$.
- $\\alpha = {results['alpha_diversification']}$: Herfindahl diversification factor deriving $\\gamma = {results['gamma_diversification']:,.2f}$.
- Bounds: $w_{{\min}} = {results['min_allocation_bound']*100:.0f}\%$, $w_{{\max}} = {results['max_allocation_bound']*100:.0f}\%$.

---

## 3. Optimal Category Budget Allocation Vector ($\mathbf{{w}}^*$)

| Product Category | Budget Allocation ($\%$) | Weight ($w_i$) |
| :--- | :---: | :---: |
{alloc_rows}

---

## 4. PSO Algorithm Stability & Performance Analysis ({n_runs} Independent PSO Trials)

| Stability & Performance Metric | Value | Description |
| :--- | :---: | :--- |
| **Independent PSO Trial Count** | **{n_runs} runs** | Total independent optimization trials |
| **Best Fitness Score** | **{best_f:,.2f}** | Highest fitness score achieved across runs |
| **Mean Fitness Score ($\mu$)** | **{m_fit:,.2f}** | Average fitness score achieved across runs |
| **Fitness Standard Deviation ($s$)** | **{s_fit:.4f}** | Standard deviation of fitness scores |
| **Stability Coefficient of Variation ($CV_{{fitness}}$)** | **`{stab_cv:.6f}`** | $CV = s / \mu$ (lower is more stable) |
| **95% Fitness Confidence Interval** | **[{ci_0:,.2f}, {ci_1:,.2f}]** | 95% Confidence Interval of fitness score |
| **Convergence Speed** | **Iteration {conv_it}** | Iteration reaching 99% max fitness |
| **Execution Runtime** | **{runtime:.3f} s** | Total stability evaluation runtime |

---

## 5. Optimization Visual Evaluation Charts

### Markowitz Efficient Frontier: Procurement Profit vs Portfolio Demand Risk
![Efficient Frontier](file:///{plot_paths['efficient_frontier'].as_posix()})

### Category Demand Correlation Heatmap
![Correlation Heatmap](file:///{plot_paths['correlation_heatmap'].as_posix()})

### Category Risk vs Expected Profit Scatter Plot
![Risk Profit Scatter](file:///{plot_paths['risk_profit_scatter'].as_posix()})

### PSO Fitness Convergence Curve
![PSO Convergence](file:///{plot_paths['pso_convergence'].as_posix()})

### PSO Independent Stability Distribution
![Stability Distribution](file:///{plot_paths['stability_distribution'].as_posix()})

### Optimal Budget Allocation Donut Chart
![Budget Allocation](file:///{plot_paths['budget_allocation'].as_posix()})
"""
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Saved Optimization Markdown Report -> %s", md_file)
        return md_file

    def generate_pdf_report(
        self,
        results: Dict[str, Any],
        stability_res: Dict[str, Any],
        plot_paths: Dict[str, Path]
    ) -> Path:
        """
        Compiles optera/pdfs/optimization_report.pdf using OpteraPDFBuilder.
        """
        pdf_file = self.pdfs_dir / "optimization_report.pdf"
        builder = OpteraPDFBuilder(
            title="Optera Executive Optimization Portfolio Evaluation Report",
            subtitle="Markowitz Mean-Variance Portfolio Optimization & PSO Stability Analysis"
        )

        builder.add_heading("1. Executive Portfolio Optimization Summary", level=1)

        summary_df = pd.DataFrame([
            {"Metric": "Portfolio Expected Profit", "Value": f"${results['portfolio_expected_profit']:,.2f}"},
            {"Metric": "Procurement Efficiency Score (PES)", "Value": f"{results['portfolio_pes']:.4f}"},
            {"Metric": "Portfolio Demand Variance", "Value": f"{results['portfolio_variance']:,.2f}"},
            {"Metric": "Portfolio Demand Std Dev", "Value": f"{results['portfolio_std_dev']:,.2f} units"},
            {"Metric": "Herfindahl Index (HHI)", "Value": f"{results['herfindahl_index']:.4f}"},
            {"Metric": "Effective Categories (ENC / DR)", "Value": f"{results['effective_number_categories']:.2f}"},
            {"Metric": "Optimal Fitness F(w*)", "Value": f"{results['optimal_fitness']:,.2f}"},
            {"Metric": "Convergence Speed", "Value": f"Iter {results['convergence_iteration']}"}
        ])
        builder.add_table(summary_df)

        builder.add_heading("2. Optimal Category Budget Allocation Vector (w*)", level=1)
        alloc_rows = []
        for cat, pct in results["allocations_percentage"].items():
            alloc_rows.append({
                "Category": cat.capitalize(),
                "Budget Allocation": f"{pct:.2f}%",
                "Weight (w_i)": f"{results['allocations_weight'][cat]:.4f}"
            })
        builder.add_table(pd.DataFrame(alloc_rows))

        builder.add_heading("3. PSO Algorithm Stability & Performance Analysis", level=1)
        stab_df = pd.DataFrame([
            {"Metric": "Independent PSO Trials", "Value": f"{stability_res['num_runs']} runs"},
            {"Metric": "Best Fitness Score", "Value": f"{stability_res.get('best_fitness', results['optimal_fitness']):,.2f}"},
            {"Metric": "Mean Fitness Score", "Value": f"{stability_res['mean_fitness']:,.2f}"},
            {"Metric": "Fitness Std Dev", "Value": f"{stability_res['std_fitness']:.4f}"},
            {"Metric": "Stability CV", "Value": f"{stability_res['stability_cv']:.6f}"},
            {"Metric": "95% Confidence Interval", "Value": f"[{stability_res['fitness_confidence_interval'][0]:,.2f}, {stability_res['fitness_confidence_interval'][1]:,.2f}]"},
            {"Metric": "Convergence Speed", "Value": f"Iter {results['convergence_iteration']}"},
            {"Metric": "Execution Runtime", "Value": f"{stability_res.get('execution_time_seconds', 0.0):.3f} s"}
        ])
        builder.add_table(stab_df)

        builder.add_heading("4. Optimization Visual Evaluation Charts", level=1)
        builder.add_image(plot_paths["efficient_frontier"])
        builder.add_image(plot_paths["correlation_heatmap"])
        builder.add_image(plot_paths["risk_profit_scatter"])
        builder.add_image(plot_paths["pso_convergence"])
        builder.add_image(plot_paths["stability_distribution"])
        builder.add_image(plot_paths["budget_allocation"])

        builder.build(pdf_file)
        logger.info("Saved Executive PDF Report -> %s", pdf_file)
        return pdf_file
