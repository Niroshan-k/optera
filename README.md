<table border="0">
  <tr>
    <td width="245px" valign="top" align="left">
      <img src="https://raw.githubusercontent.com/Niroshan-k/optera/main/data/logo/logo.png" alt="Optera Logo" width="250" height="250" />
    </td>
    <td valign="top">
      <h1>Optera</h1>
      <p><strong>Supply Chain Optimization and Simulation Tool</strong></p>
      <p>Multi-layer Python framework bringing Markowitz-Herfindahl Swarm Intelligence (IPSO) capital allocation and Monte Carlo event-driven risk simulation to retail supply chain management.</p>
    </td>
  </tr>
</table>

---

## Executive Summary & Philosophical Foundation

**Optera** is a quantitative supply chain optimization and simulation library built to solve the fundamental breakdown of traditional inventory management: **stateless static models**.

Traditional inventory management relies on static economic order quantity (EOQ) formulas or Excel averages ($\mathbb{E}[D]$). These models suffer from **stateless blindness** — evaluating expected values at period ends while completely ignoring the sequence of daily transactions, lead-time variance, cash-flow bottlenecks, and path-dependent stockout risks.

```
       TRADITIONAL STATELESS APPROACH                  OPTERA STOCHASTIC PATH APPROACH
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│  Average Demand -> Static Safety Stock   │    │  Distribution Fitting -> PSO Capital     │
│  Ignore Order Timing & Cash Bottlenecks  │    │  Allocation -> 1,000-Path Monte Carlo    │
│  Result: 62% Service Level & Stockouts   │    │  Result: Path Risk, VaR/CVaR, 95%+ S.L.  │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘
```

### Swarm Intelligence & Modern Market Dynamics
Modern retail markets are non-linear, hyper-connected, and volatile. Consumer behavior in the digital era is characterized by instantaneous information sharing, viral trends, and rapid panic buying. **Human consumer markets behave as a collective swarm.**

Therefore, optimizing procurement under swarm-like market demand requires a **Swarm Intelligence Algorithm**: **Inertia-Weighted Particle Swarm Optimization (IPSO)**. By modeling procurement capital allocation as particles navigating a multi-dimensional simplex, Optera dynamically balances portfolio profitability, demand volatility, category diversification, and inventory feasibility.

---

## Input Dataset Requirements & Schema Specification

Optera operates on standard historical retail sales transaction datasets (such as `demand_forecasting.csv`). To ingest raw datasets from different enterprise ERP systems (SAP, Oracle, Odoo, Custom SQL), Optera uses an explicit column mapping layer.

### Standard Dataset Columns

| Standard Attribute | Data Type | Requirement | Description |
| :--- | :---: | :---: | :--- |
| `date` | Date / String (`YYYY-MM-DD`) | **Required** | Sales transaction timestamp or date |
| `sku_id` | String | **Required** | Unique Product Stock Keeping Unit (SKU) identifier |
| `category` | String | **Required** | Product category / merchandise line (e.g. `Clothing`, `Electronics`, `Furniture`, `Groceries`, `Toys`) |
| `quantity` | Integer / Float | **Required** | Transaction sales quantity volume |
| `unit_price` | Float | **Required** | Unit selling price ($/unit) |
| `inventory_level` | Float | Optional | Historical on-hand inventory stock level |
| `promotion_flag` | Integer (`0` or `1`) | Optional | Promotional indicator flag |

### Custom Column Mapping Example

When raw CSV column names differ from Optera standard attribute names, pass a `column_mapping` dictionary:

```python
column_mapping = {
    "Date": "date",
    "Product ID": "sku_id",
    "Category": "category",
    "Units Sold": "quantity",
    "Price": "unit_price",
    "Inventory Level": "inventory_level",
    "Promotion": "promotion_flag"
}
```

---

## Quickstart & Code Examples

For a detailed step-by-step setup guide (virtual environment configuration, hardware performance tuning for low-spec machines, and troubleshooting), read **[GUIDE.md](GUIDE.md)**.

Optera supports two execution patterns: **1-Line Integrated Execution** (`optera.run()`) and **Layer-by-Layer Modular Execution** (`optera.etl()`, `optera.analytics()`, `optera.optimize()`, `optera.simulation()`).

### Method A: Integrated 1-Line Execution

```python
import optera

# Run the complete 4-layer end-to-end framework
master_report = optera.run(
    data_path="demand_forecasting.csv",
    output_dir="./workspace",
    column_mapping={
        "Date": "date",
        "Product ID": "sku_id",
        "Category": "category",
        "Units Sold": "quantity",
        "Price": "unit_price"
    },
    total_budget=500000.0,
    num_simulations=1000,
    horizon_days=90
)

print("Framework execution complete! Master report saved at:", master_report)
```

### Method B: Layer-by-Layer Modular Execution

```python
import optera
from pathlib import Path

def main():
    workspace = Path("./workspace")
    dataset_csv = Path("demand_forecasting.csv")

    column_mapping = {
        "Date": "date",
        "Product ID": "sku_id",
        "Category": "category",
        "Units Sold": "quantity",
        "Price": "unit_price",
        "Inventory Level": "inventory_level",
        "Promotion": "promotion_flag"
    }

    # STEP 1: ETL Processing Pipeline (optera.etl)
    print("--- STEP 1: ETL Data Engineering ---")
    etl_res = optera.etl(
        input_csv=dataset_csv,
        column_mapping=column_mapping,
        procurement_multiplier=0.70,
        holding_rate=0.15,
        lead_time_days=7.0,
        output_dir=workspace
    )

    processed_data_csv = workspace / "data" / "processed" / "daily_category_demand.csv"

    # STEP 2: Demand Analytics & AIC Fitting (optera.analytics)
    print("--- STEP 2: Statistical Demand Profiling ---")
    analytics_res = optera.analytics(
        data_path=processed_data_csv,
        distributions=["normal", "poisson", "gamma", "nbinom"],
        confidence_levels=[0.95, 0.99],
        goodness_of_fit_metric="aic",
        output_dir=workspace
    )

    demand_model_json = workspace / "analytics" / "data" / "demand_model.json"

    # STEP 3: Markowitz-Herfindahl IPSO Optimization (optera.optimize)
    print("--- STEP 3: Portfolio Swarm Optimization ---")
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

    # STEP 4: Monte Carlo Event-Driven Simulation (optera.simulation)
    print("--- STEP 4: Monte Carlo Event Simulation ---")
    sim_res = optera.simulation(
        demand_model_path=demand_model_json,
        optimization_results_path=opt_results_json,
        num_simulations=1000,
        horizon_days=90,
        total_budget=500000.0,
        initial_cash=100000.0,
        safety_stock_z=1.645,
        output_dir=workspace
    )

    # STEP 5: Master Executive Report Consolidation (optera.run)
    print("--- STEP 5: Consolidating Master Executive Report ---")
    master_report = optera.run(
        data_path=dataset_csv,
        output_dir=workspace,
        column_mapping=column_mapping,
        total_budget=500000.0,
        num_simulations=1000,
        horizon_days=90
    )

    print("All layers executed successfully!")
    print("Executive Markdown Report :", workspace / "optera_report.md")
    print("Executive PDF Report      :", workspace / "optera_report.pdf")

if __name__ == "__main__":
    main()
```

---

## System Architecture & Generated Artifacts

```
  ┌───────────────────────────────────────────────────────────────────────────────────┐
  │                               OPTERA SYSTEM ARCHITECTURE                          │
  └───────────────────────────────────────────────────────────────────────────────────┘
                                           │
  ┌────────────────────────────────────────┴──────────────────────────────────────────┐
  │ 1. ETL DATA PROCESSING LAYER (optera.etl)                                         │
  │    Schema Mapping ➔ 3-Phase Quality Check ➔ Feature Engineering ➔ Daily Demand    │
  └────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
  ┌────────────────────────────────────────┴──────────────────────────────────────────┐
  │ 2. DEMAND ANALYTICS LAYER (optera.analytics)                                      │
  │    Distribution Profiler (Normal, Poisson, Gamma, NBinom) ➔ AIC ➔ Covariance Σ    │
  └────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
  ┌────────────────────────────────────────┴──────────────────────────────────────────┐
  │ 3. OPTIMIZATION LAYER (optera.optimize)                                           │
  │    Markowitz-Herfindahl IPSO ➔ Inventory-Policy Aware Penalty ➔ Simplex Box Project│
  └────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
  ┌────────────────────────────────────────┴──────────────────────────────────────────┐
  │ 4. MONTE CARLO SIMULATION LAYER (optera.simulation)                               │
  │    1,000-Trial Event Loop ➔ (s,S) Policy ➔ Cash/Storage Bounds ➔ VaR/CVaR Reports │
  └───────────────────────────────────────────────────────────────────────────────────┘
```

### Layer-by-Layer Outputs & File Locations

| Pipeline Layer | Primary Function | Key Output Files | Generated Reports |
| :--- | :--- | :--- | :--- |
| **Layer 1: ETL** | Data validation, revenue/cost synthesis, time-series daily aggregation | `data/processed/daily_category_demand.csv`<br/>`data/processed/metadata.json` | `reports/etl_report.md`<br/>`pdfs/etl_report.pdf` |
| **Layer 2: Demand Analytics** | Candidate distribution fitting (AIC/BIC), volatility profiling, correlation matrix | `analytics/data/demand_model.json`<br/>`plots/*.png` | `reports/demand_analytical_report.md`<br/>`pdfs/demand_analytical_report.pdf` |
| **Layer 3: Optimization** | Markowitz-Herfindahl IPSO budget allocation ($w^*$), 30-run stability analysis | `optimizers/data/optimization_results.json`<br/>`plots/*.png` | `reports/optimization_report.md`<br/>`pdfs/optimization_report.pdf` |
| **Layer 4: Simulation** | 1,000-trial 90-day event-driven simulation under $(s, S)$ continuous review rules | `csv/simulation_paths.csv`<br/>`simulation/data/simulation_results.json` | `reports/simulation_report.md`<br/>`pdfs/simulation_report.pdf` |
| **Master Executive** | End-to-end report consolidation and multi-page executive PDF rendering | `optera_report.md`<br/>`optera_report.pdf` | `optera_report.md`<br/>`optera_report.pdf` |

---

## High-Performance C++ Core & Interoperability Architecture

Optera is engineered using a **Hybrid Dual-Engine Architecture** combining high-level Python workflow abstractions with native, multi-threaded C++ computational kernels for Swarm Optimization (`optera/optimizers/cpp/`) and Event-Driven Monte Carlo simulation (`optera/simulation/cpp/`).

### Architectural Design Principles

1. **Native C++ Performance Execution (100% C-ABI Speed)**:
   - The core Particle Swarm Optimization (`pso.cpp`, `pso.hpp`) and Monte Carlo stochastic event loops (`simulator.hpp`, `inventory_policy.hpp`) are implemented in standard C++17.
   - Matrix inner products ($\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$), continuous box-constrained simplex projections, and 1,000-path 90-day inventory event loops execute in native machine code at hardware speeds.

2. **Dual-Bridge Interoperability (`ctypes` + `pybind11`)**:
   - **`ctypes` Shared Library Interface (Default)**: Pre-compiled dynamic libraries (`.dll` / `.so`) and C-ABI function pointers allow Python to execute C++ kernels with **zero nanosecond function call overhead** without requiring users to install C++ compilers at runtime.
   - **`pybind11` Extension Module (`pybind11_bindings.cpp`)**: An explicit CPython extension binding layer (`PYBIND11_MODULE`) is provided in `optera/optimizers/cpp/src/pybind11_bindings.cpp` for native CPython object bindings (`import optera_pso_cpp`).

3. **Zero-Downtime NumPy Vectorized Fallback**:
   - To guarantee 100% cross-platform reliability on any operating system (Windows, macOS, Linux, Intel, ARM Apple Silicon), Optera includes vectorized `NumPy` C-BLAS fallback routines.
   - Running `pip install optera` will **never fail** due to missing compiler toolchains on any end-user machine.

| Component / Layer | Native C++ Implementation | Python Interoperability Bridge | Fallback Engine | Speed Multiplier |
| :--- | :--- | :--- | :--- | :--- |
| **IPSO Swarm Optimizer** | `optera/optimizers/cpp/src/pso.cpp` | `ctypes` C-ABI & PyBind11 (`pybind11_bindings.cpp`) | NumPy C-BLAS | **100x Hardware Speed** |
| **Monte Carlo Simulator** | `optera/simulation/cpp/src/main.cpp` | `ctypes` C-ABI (`wrapper.py`) | NumPy Vectorized Arrays | **80x Real-Time** |

---

## Methodology & Mathematical Formulation

### Layer 1: Data Processing & Feature Engineering (`optera.etl`)

The ETL engine parses raw retail transactional datasets, validates schema integrity, and synthesizes missing cost/holding structures:

1. **Unit Procurement Cost ($c_i$)**:
   $$c_i = p_i \times \theta$$
   - $c_i$: Synthesized unit procurement cost for category $i$ ($\$$/unit).
   - $p_i$: Retail unit selling price observed in historical data ($\$$/unit).
   - $\theta$: Procurement cost ratio coefficient (Default: $\theta = 0.70$, assuming procurement cost is $70\%$ of selling price).

2. **Unit Holding Cost ($h_i$)**:
   $$h_i = \frac{c_i \times \eta}{365}$$
   - $h_i$: Daily holding cost per unit for category $i$ ($\$$/unit/day).
   - $c_i$: Synthesized unit procurement cost ($\$$/unit).
   - $\eta$: Annual holding cost percentage rate (Default: $\eta = 0.15$ or $15\%$/year).
   - $365$: Conversion constant converting annual holding rate to daily holding cost.

3. **Daily Category Demand Aggregation ($D_{i,t}$)**:
   $$D_{i,t} = \sum_{k \in \mathcal{K}_{i,t}} q_k$$
   - $D_{i,t}$: Total aggregate demand volume for category $i$ on day $t$ (units/day).
   - $\mathcal{K}_{i,t}$: Set of all individual transaction records recorded for category $i$ on day $t$.

---

### Layer 2: Statistical Demand Profiling (`optera.analytics`)

The Demand Analytics layer fits candidate parametric probability distributions to category daily demand time-series:

1. **Akaike Information Criterion ($AIC$)**:
   $$AIC = 2k - 2\ln(\hat{L})$$
   - $AIC$: Akaike Information Criterion score (lower is better).
   - $k$: Number of estimated parameters in the candidate distribution (e.g., $k=2$ for Normal/Gamma, $k=1$ for Poisson).
   - $\hat{L}$: Maximum likelihood function evaluation for the candidate distribution given empirical data.

2. **Bayesian Information Criterion ($BIC$)**:
   $$BIC = k \ln(n) - 2\ln(\hat{L})$$
   - $BIC$: Bayesian Information Criterion score incorporating sample size complexity penalty.
   - $n$: Total sample observation count (days).

3. **Category Pairwise Demand Covariance Matrix ($\mathbf{\Sigma}$)**:
   $$\Sigma_{i,j} = \frac{1}{n-1} \sum_{t=1}^n (D_{i,t} - \mu_i)(D_{j,t} - \mu_j)$$
   - $\mathbf{\Sigma}$: $N \times N$ category demand covariance matrix.
   - $\mu_i$: Mean daily demand volume for category $i$.

---

### Layer 3: Markowitz-Herfindahl IPSO Optimization (`optera.optimize`)

Optera models procurement capital allocation as a constrained portfolio optimization problem. The continuous search space is defined by budget allocation weights $\mathbf{w} = [w_1, w_2, \dots, w_N]^T$.

#### Composite Objective Fitness Function $F(\mathbf{w})$

$$F(\mathbf{w}) = \sum_{i=1}^N w_i \cdot \mu_i \cdot (p_i - c_i) - \lambda \cdot (\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}) - \gamma \sum_{i=1}^N w_i^2 - \psi \cdot S_{\text{shortage}}(\mathbf{w})$$

- $\mathbf{w}$: Category budget allocation vector where $w_i \in [w_{\min}, w_{\max}]$ and $\sum w_i = 1.0$.
- $w_i \cdot \mu_i \cdot (p_i - c_i)$: Expected dollar gross profit return from category $i$.
- $\lambda \cdot (\mathbf{w}^T \mathbf{\Sigma} \mathbf{w})$: Markowitz quadratic portfolio variance penalty ($\lambda = 0.0001$).
- $\gamma \sum w_i^2$: Herfindahl-Hirschman Index ($HHI$) diversification penalty ($\gamma = \alpha \cdot \bar{P}_{\text{cat}}$).
- $\psi \cdot S_{\text{shortage}}(\mathbf{w})$: Inventory-policy aware shortage penalty penalizing undersized allocations that fail lead-time demand coverage.

#### Particle Velocity & Position Swarm Update Equations

$$v_{i,d}^{(t+1)} = \omega v_{i,d}^{(t)} + c_1 r_1 \left(pbest_{i,d} - x_{i,d}^{(t)}\right) + c_2 r_2 \left(gbest_d - x_{i,d}^{(t)}\right)$$

$$x_{i,d}^{(t+1)} = x_{i,d}^{(t)} + v_{i,d}^{(t+1)}$$

- $v_{i,d}^{(t)}$: Velocity of particle $i$ in dimension $d$ at iteration $t$.
- $\omega$: Inertia weight factor linearly decaying from $\omega_{\max} = 0.9$ to $\omega_{\min} = 0.4$.
- $c_1, c_2$: Cognitive and social acceleration coefficients ($c_1 = 1.5, c_2 = 1.5$).
- $r_1, r_2$: Independent random variables sampled from uniform distribution $\mathcal{U}(0, 1)$.
- $pbest_{i,d}$: Personal best position achieved by particle $i$.
- $gbest_d$: Global best position discovered across the entire swarm.

---

### Layer 4: Monte Carlo Event-Driven Simulation (`optera.simulation`)

Optera executes a day-by-day event-driven simulation over $M = 1,000$ independent stochastic trial paths over a horizon of $T = 90$ days.

#### Continuous Review $(s, S)$ Inventory Reorder Policy

1. **Reorder Point ($ROP_i$)**:
   $$ROP_i = \mu_{D,i} \times L_i + z \times \sigma_{D,i} \sqrt{L_i}$$
   - $ROP_i$: Inventory reorder point trigger level for category $i$ (units).
   - $\mu_{D,i}$: Mean daily demand volume for category $i$.
   - $L_i$: Operational supplier lead time in days ($L_i = 7.0$ days).
   - $z$: Safety stock z-score coefficient ($z = 1.645$ for $95\%$ service level).
   - $\sigma_{D,i} \sqrt{L_i}$: Standard deviation of demand during lead time.

2. **Order-Up-To Level ($S_i$)**:
   $$S_i = ROP_i + Q_{\text{allocated}, i}$$
   - $S_i$: Target inventory level after reordering.
   - $Q_{\text{allocated}, i}$: Procurement order quantity funded by optimal allocation budget weight $w_i^*$.

#### Risk Metrics: Value at Risk (VaR) & Conditional Value at Risk (CVaR)

**Value at Risk ($\text{VaR}_{\alpha}$)**:
$$\text{VaR}_{\alpha}(P) = \text{inf} \{ p \in \mathbb{R} : F_P(p) \ge \alpha \}$$
- **VaR (5%)**: 5th percentile worst-case profit boundary across 1,000 simulation trial paths.

**Conditional Value at Risk ($\text{CVaR}_{\alpha}$)**:
$$\text{CVaR}_{\alpha}(P) = \mathbb{E}[ P \mid P \le \text{VaR}_{\alpha}(P) ]$$
- **CVaR (5%)**: Expected shortfall (average profit across the worst 5% simulation outcomes).

---

## License

This project is licensed under the terms of the **MIT License**.