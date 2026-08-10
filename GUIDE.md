# Optera Execution & Deployment Guide

Welcome to the official deployment and user onboarding guide for **Optera** — the quantitative supply chain optimization and stochastic simulation tool.

This guide provides step-by-step instructions on setting up virtual environments, installing Optera from PyPI, tuning execution parameters for low-spec or high-performance hardware, and running full modular supply chain pipelines.

---

## Step 1: Environment Setup & Package Installation

To avoid library version conflicts, it is recommended to run Optera inside an isolated Python virtual environment (`venv`).

### 1.1 Create a Virtual Environment

Open your terminal or command prompt in your project root folder:

- **Windows**:
  ```cmd
  python -m venv venv
  ```

- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  ```

---

### 1.2 Activate the Virtual Environment

Activate your virtual environment based on your operating system:

- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\activate
  ```

- **Windows (Command Prompt / CMD)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

---

### 1.3 Install Optera via PyPI

Install the official release from PyPI:

```bash
pip install optera
```

> **Note**: `pip install optera` automatically resolves and installs all required scientific dependencies (`numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `reportlab`, `scikit-learn`).

---

## Step 2: Hardware Performance & Resource Tuning Advisory

Optera utilizes multi-threaded C++ / SciPy matrix routines for Particle Swarm Optimization (IPSO) and 1,000-trial Monte Carlo stochastic simulation paths.

### ⚙️ Performance Parameter Presets

Depending on your computer hardware specifications, adjust the optimization and simulation parameters:

| Hardware Spec | Swarm Size (`swarm_size`) | Max Iterations (`max_iterations`) | Monte Carlo Paths (`num_simulations`) | Simulation Horizon (`horizon_days`) | Expected Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low-Spec Machine** *(Dual-Core / 8GB RAM)* | `30` | `50` | `200` | `30 days` | ~5 - 10 seconds |
| **Standard Laptop** *(Quad-Core / 16GB RAM)* | `50` | `100` | `500` | `60 days` | ~15 - 25 seconds |
| **High-Performance** *(Multi-Core / 32GB RAM)* | `50 - 100` | `150 - 250` | `1000 - 5000` | `90 - 365 days` | ~30 - 60 seconds |

> **Advisory**: Setting extremely high trial path counts (e.g. `num_simulations=10,000` or `swarm_size=200`) on low-spec dual-core machines will increase CPU load and execution time. Start with smaller presets for quick testing.

---

## Step 3: Preparing Input Data & Column Mapping

Optera reads raw transaction sales datasets in `.csv` format (e.g. `example/demand_forecasting.csv`).

### Standard Required Attributes

| Attribute Name | Data Type | Required | Description |
| :--- | :---: | :---: | :--- |
| `date` | String (`YYYY-MM-DD`) | **Yes** | Transaction date timestamp |
| `sku_id` | String | **Yes** | Unique product SKU identifier |
| `category` | String | **Yes** | Product category / line (e.g., `Clothing`, `Electronics`, `Furniture`) |
| `quantity` | Float / Int | **Yes** | Transaction sales volume |
| `unit_price` | Float | **Yes** | Unit selling price ($/unit) |

### Setting Up Column Mapping

If your raw CSV column headers differ from standard names, pass a `column_mapping` dictionary:

```python
column_mapping = {
    "Date": "date",
    "Product ID": "sku_id",
    "Category": "category",
    "Units Sold": "quantity",
    "Price": "unit_price"
}
```

---

## Step 4: Running the Full Modular Execution Script

You can execute Optera using two methods: **1-Line Quickstart** or **Full Modular Layer Control** (matching `example/user_script.py`).

### Full Modular Script (`user_script.py`)

Create a Python script named `run_optera.py`:

```python
import logging
from pathlib import Path
import optera

# Enable clean logging output
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

def main():
    # Define workspace paths
    workspace = Path("./output_workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    # Path to historical sales dataset (CSV)
    dataset_csv = Path("./example/demand_forecasting.csv").resolve()

    # Column mapping for raw ERP headers
    column_mapping = {
        "Date": "date",
        "Product ID": "sku_id",
        "Category": "category",
        "Units Sold": "quantity",
        "Price": "unit_price"
    }

    print("=" * 80)
    print("STARTING OPTERA SUPPLY CHAIN OPTIMIZATION & SIMULATION")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # STEP 1: ETL Processing Pipeline (optera.etl)
    # --------------------------------------------------------------------------
    print("\n--- [STEP 1/5] Running ETL Pipeline ---")
    etl_res = optera.etl(
        input_csv=dataset_csv,
        column_mapping=column_mapping,
        procurement_multiplier=0.70,  # Procurement cost is 70% of price
        holding_rate=0.15,            # Annual holding cost rate (15%)
        lead_time_days=7.0,           # Operational lead time (7 days)
        output_dir=workspace
    )

    processed_data_csv = workspace / "data" / "processed" / "daily_category_demand.csv"

    # --------------------------------------------------------------------------
    # STEP 2: Statistical Demand Profiling (optera.analytics)
    # --------------------------------------------------------------------------
    print("\n--- [STEP 2/5] Running Statistical Demand Analytics ---")
    analytics_res = optera.analytics(
        data_path=processed_data_csv,
        distributions=["normal", "poisson", "gamma", "nbinom"],
        confidence_levels=[0.95, 0.99],
        goodness_of_fit_metric="aic",
        output_dir=workspace
    )

    demand_model_json = workspace / "analytics" / "data" / "demand_model.json"

    # --------------------------------------------------------------------------
    # STEP 3: Markowitz-Herfindahl IPSO Optimization (optera.optimize)
    # --------------------------------------------------------------------------
    print("\n--- [STEP 3/5] Running Swarm Portfolio Optimization ---")
    opt_res = optera.optimize(
        data_path=demand_model_json,
        total_budget=500000.0,         # Total procurement budget ($500k)
        swarm_size=50,                 # Swarm size (tune per hardware advisory)
        max_iterations=150,            # Max iterations
        lambda_risk=0.0001,            # Markowitz risk aversion factor
        alpha_diversification=0.10,    # Herfindahl diversification factor
        shortage_penalty_weight=1.0,   # Inventory shortage penalty
        output_dir=workspace
    )

    opt_results_json = workspace / "optimizers" / "data" / "optimization_results.json"

    # --------------------------------------------------------------------------
    # STEP 4: Monte Carlo Event-Driven Simulation (optera.simulation)
    # --------------------------------------------------------------------------
    print("\n--- [STEP 4/5] Running Monte Carlo Stochastic Simulation ---")
    sim_res = optera.simulation(
        demand_model_path=demand_model_json,
        optimization_results_path=opt_results_json,
        num_simulations=1000,          # Number of trial paths (tune per hardware advisory)
        horizon_days=90,               # Simulation horizon
        total_budget=500000.0,
        initial_cash=100000.0,         # Initial cash buffer ($100k)
        safety_stock_z=1.645,          # 95% service level safety stock z-score
        output_dir=workspace
    )

    # --------------------------------------------------------------------------
    # STEP 5: Consolidating Integrated Master Report (optera.run)
    # --------------------------------------------------------------------------
    print("\n--- [STEP 5/5] Consolidating Executive Master Reports ---")
    master_md = optera.run(
        data_path=dataset_csv,
        output_dir=workspace,
        column_mapping=column_mapping,
        total_budget=500000.0,
        num_simulations=1000,
        horizon_days=90
    )

    print("=" * 80)
    print("ALL OPTERA PIPELINE LAYERS EXECUTED SUCCESSFULLY!")
    print("=" * 80)
    print(f"  • Executive Markdown Report : {workspace / 'optera_report.md'}")
    print(f"  • Executive PDF Report      : {workspace / 'optera_report.pdf'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

Run your script:

```bash
python run_optera.py
```

---

## Step 5: Output Artifacts & Executive Reports

Upon successful execution, Optera generates publication-ready artifacts inside your specified `workspace` directory:

```
output_workspace/
├── optera_report.md                # Integrated Executive Master Markdown Report
├── optera_report.pdf               # Multi-page Executive PDF Report (ReportLab)
├── data/
│   └── processed/
│       ├── daily_category_demand.csv
│       └── metadata.json
├── analytics/
│   └── data/
│       └── demand_model.json
├── optimizers/
│   └── data/
│       └── optimization_results.json
├── simulation/
│   └── data/
│       └── simulation_results.json
├── csv/
│   └── simulation_paths.csv        # 1,000-trial stochastic path export
├── reports/                        # Individual layer Markdown reports
├── pdfs/                           # Individual layer PDF reports
└── plots/                          # PNG visual evaluation charts & heatmaps
```

---

## Troubleshooting & FAQs

- **Q: `ImportError` or missing package errors?**
  - Ensure your virtual environment is activated (`.\venv\Scripts\activate` or `source venv/bin/activate`) and run `pip install optera`.

- **Q: Script runs slowly during step 3 or step 4?**
  - Lower `swarm_size` to `30`, `max_iterations` to `50`, and `num_simulations` to `200` as per the hardware tuning advisory.

- **Q: Raw CSV headers don't match standard names?**
  - Pass the exact mapping in `column_mapping` as demonstrated in Step 3.
