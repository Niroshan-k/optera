<table border="0">
  <tr>
    <td width="245px" valign="top" align="left">
      <img src="data/UI/logo.png" alt="Optera Logo" width="250" height="250" />
    </td>
    <td valign="top">
      <h1>Optera</h1>
      <p><strong>Quantitative Supply Chain Framework & Stochastic Portfolio Optimization Engine</strong></p>
      <p>An institutional-grade, multi-layer Python framework bringing Markowitz-Herfindahl Swarm Intelligence (IPSO) capital allocation and Monte Carlo event-driven risk simulation to retail supply chain management.</p>
      <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-C96442.svg" alt="Python" /></a>
      <a href="https://scipy.org"><img src="https://img.shields.io/badge/SciPy-Accelerated-B0A6DF.svg" alt="SciPy" /></a>
      <img src="https://img.shields.io/badge/Status-Production%20Ready-green.svg" alt="Status" />
      <br/>
      <em>Privacy-First · 100% On-Premise · Fully Modular Python API (`import optera`)</em>
    </td>
  </tr>
</table>

---

## 📌 Executive Summary & Philosophical Foundation

**Optera** is a quantitative supply chain optimization and simulation library built to solve the fundamental breakdown of traditional inventory management: **stateless static models**.

Traditional inventory management relies on static economic order quantity (EOQ) formulas or Excel averages ($\mathbb{E}[D]$). These models suffer from **stateless blindness** — they evaluate expected values at period ends while completely ignoring the sequence of daily transactions, lead-time variance, cash-flow bottlenecks, and path-dependent stockout risks.

```
       TRADITIONAL STATELESS APPROACH                  OPTERA STOCHASTIC PATH APPROACH
┌──────────────────────────────────────────┐    ┌──────────────────────────────────────────┐
│  Average Demand -> Static Safety Stock   │    │  Distribution Fitting -> PSO Capital     │
│  Ignore Order Timing & Cash Bottlenecks  │    │  Allocation -> 1,000-Path Monte Carlo    │
│  Result: 62% Service Level & Stockouts   │    │  Result: Path Risk, VaR/CVaR, 95%+ S.L.  │
└──────────────────────────────────────────┘    └──────────────────────────────────────────┘
```

### 🧠 Swarm Intelligence & Modern Market Dynamics
Modern retail markets are non-linear, hyper-connected, and volatile. Consumer behavior in the digital era is characterized by instantaneous information sharing, viral trends, and rapid panic buying. **Human consumer markets behave as a collective swarm.**

Therefore, optimizing procurement under swarm-like market demand requires a **Swarm Intelligence Algorithm**: **Inertia-Weighted Particle Swarm Optimization (IPSO)**. By modeling procurement capital allocation as particles navigating a multi-dimensional simplex, Optera dynamically balances portfolio profitability, demand volatility, category diversification, and inventory feasibility.

---

## 🏗️ System Architecture (The 4 Layers)

Optera is engineered as a clean 4-layer decoupled pipeline:

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
  │    Markowitz-Herfindahl IPSO ➔ Inventory-Policy Aware Penalty ➔ Simplex Box Project
  └────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
  ┌────────────────────────────────────────┴──────────────────────────────────────────┐
  │ 4. MONTE CARLO SIMULATION LAYER (optera.simulation)                               │
  │    1,000-Trial Event Loop ➔ (s,S) Policy ➔ Cash/Storage Bounds ➔ VaR/CVaR Reports │
  └───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology & Mathematical Formulation

### Layer 1: Data Processing & Feature Engineering (`optera.etl`)

The ETL engine parses raw retail transactional datasets, validates schema integrity, and synthesizes missing cost/holding structures:

1. **Unit Procurement Cost ($c_i$)**:
   $$c_i = p_i \times \theta$$
   > **Variable Definitions**:
   > - $c_i$: Synthesized unit procurement cost for category $i$ ($\$$/unit).
   > - $p_i$: Retail unit selling price observed in historical data for category $i$ ($\$$/unit).
   > - $\theta$: Procurement cost ratio coefficient (Default: $\theta = 0.70$, assuming procurement cost is $70\%$ of selling price).

2. **Unit Holding Cost ($h_i$)**:
   $$h_i = \frac{c_i \times \eta}{365}$$
   > **Variable Definitions**:
   > - $h_i$: Daily holding cost per unit for category $i$ ($\$$/unit/day).
   > - $c_i$: Synthesized unit procurement cost ($\$$/unit).
   > - $\eta$: Annual holding cost percentage rate (Default: $\eta = 0.15$ or $15\%$/year).
   > - $365$: Conversion constant converting annual holding rate to daily holding cost.

3. **Daily Category Demand Aggregation ($D_{i,t}$)**:
   $$D_{i,t} = \sum_{k \in \mathcal{K}_{i,t}} q_k$$
   > **Variable Definitions**:
   > - $D_{i,t}$: Total aggregate demand volume for category $i$ on day $t$ (units/day).
   > - $\mathcal{K}_{i,t}$: Set of all individual transaction records recorded for category $i$ on day $t$.
   > - $q_k$: Quantity of units sold in customer transaction $k$.

---

### Layer 2: Statistical Demand Profiling & Covariance Analysis (`optera.analytics`)

Rather than assuming all product categories follow a Normal distribution, Optera fits **4 candidate probability distributions** per category $i$:
$$\mathcal{M}_i = \{\text{Normal}(\mu, \sigma^2), \, \text{Poisson}(\lambda), \, \text{Gamma}(\alpha, \beta), \, \text{NegativeBinomial}(r, p)\}$$

The optimal probability density function $f_i^*(x)$ is selected by minimizing the **Akaike Information Criterion (AIC)**:
$$\text{AIC}_i = 2k - 2\ln(\hat{L}_i)$$
> **Variable Definitions**:
> - $\text{AIC}_i$: Akaike Information Criterion score for candidate model on category $i$ (lower is better).
> - $k$: Number of estimated parameters in the candidate distribution (e.g., $k=2$ for Normal/Gamma, $k=1$ for Poisson).
> - $\hat{L}_i$: Maximum likelihood estimation (MLE) value of the fitted distribution parameters given observed historical data.

#### Cross-Category Covariance Matrix ($\boldsymbol{\Sigma}$)
To account for inter-category demand coupling and portfolio risk, Optera computes the empirical covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$:
$$\Sigma_{i,j} = \frac{1}{T-1} \sum_{t=1}^T (D_{i,t} - \mu_i)(D_{j,t} - \mu_j)$$
> **Variable Definitions**:
> - $\Sigma_{i,j}$: Empirical covariance between daily demand of category $i$ and category $j$ ($\text{units}^2$).
> - $T$: Total number of historical observation days (e.g., $T = 730$ days).
> - $D_{i,t}, D_{j,t}$: Daily demand volume recorded for category $i$ and category $j$ on day $t$ (units).
> - $\mu_i, \mu_j$: Mean daily demand for category $i$ and category $j$ across all $T$ days (units/day).

---

### Layer 3: Markowitz-Herfindahl PSO Capital Allocation (`optera.optimize`)

#### Philosophical Rationale for Particle Swarm Optimization (PSO)
Gradient-based optimizers (e.g., Sequential Least Squares Programming) fail in multi-category inventory allocation due to non-convex penalty surfaces, discrete unit constraints, and box bounds. Particle Swarm Optimization explores the $N$-dimensional budget simplex through cooperative swarm intelligence, avoiding local optima.

#### Mathematical Master Objective Function
The objective function maximizes total portfolio fitness $F(w)$ for allocation weights $w = [w_1, w_2, \dots, w_N]^T$:

$$\max_{w \in \Delta^N} F(w) = P(w) - V(w) - D(w) - \mathcal{S}(w)$$
> **Variable Definitions**:
> - $F(w)$: Master portfolio fitness score for candidate allocation weight vector $w$ (unitless objective value).
> - $w = [w_1, w_2, \dots, w_N]^T$: Portfolio budget allocation weights vector where $w_i \in [w_{\min}, w_{\max}]$ and $\sum_{i=1}^N w_i = 1.0$.
> - $P(w)$: Total expected portfolio gross profit ($\$$).
> - $V(w)$: Markowitz portfolio demand volatility risk penalty ($\$$).
> - $D(w)$: Herfindahl market concentration diversification penalty ($\$$).
> - $\mathcal{S}(w)$: Inventory policy-aware lead-time shortage penalty ($\$$).

1. **Expected Gross Profit Component $P(w)$**:
   $$P(w) = \sum_{i=1}^N w_i \cdot \mu_i \cdot (p_i - c_i)$$
   > **Variable Definitions**:
   > - $P(w)$: Expected total portfolio daily gross profit ($\$$/day).
   > - $w_i$: Capital allocation weight assigned to category $i$ ($w_i \in [0.05, 0.40]$).
   > - $\mu_i$: Mean daily demand for category $i$ (units/day).
   > - $p_i$: Mean retail selling price per unit for category $i$ ($\$$/unit).
   > - $c_i$: Mean procurement cost per unit for category $i$ ($\$$/unit).
   > - $(p_i - c_i)$: Unit gross profit margin for category $i$ ($\$$/unit).

2. **Markowitz Volatility Risk Penalty $V(w)$**:
   $$V(w) = \lambda \cdot \left( w^T \boldsymbol{\Sigma} w \right)$$
   > **Variable Definitions**:
   > - $V(w)$: Markowitz demand volatility risk penalty term ($\$$).
   > - $\lambda$: Risk aversion penalty parameter (Default: $\lambda = 0.0001$).
   > - $w$: Capital allocation weight vector ($N \times 1$).
   > - $\boldsymbol{\Sigma}$: Empirical cross-category demand covariance matrix ($N \times N$).
   > - $w^T \boldsymbol{\Sigma} w$: Total portfolio demand variance ($\text{units}^2$).

3. **Herfindahl Diversification Penalty $D(w)$**:
   $$D(w) = \gamma \cdot \text{HHI}(w) = \gamma \sum_{i=1}^N w_i^2$$
   > **Variable Definitions**:
   > - $D(w)$: Portfolio concentration penalty term ($\$$).
   > - $\text{HHI}(w) = \sum_{i=1}^N w_i^2$: Herfindahl-Hirschman Index measuring portfolio allocation concentration ($\text{HHI} \in [1/N, 1.0]$).
   > - $\gamma = \alpha \cdot \bar{P}_{\text{cat}}$: Dynamic penalty multiplier where $\alpha = 0.10$ and $\bar{P}_{\text{cat}} = \frac{1}{N}\sum \mu_i(p_i - c_i)$ is average category profit.

4. **Inventory Policy-Aware Shortage Penalty $\mathcal{S}(w)$ (CRITICAL INVENTIVE STEP)**:
   A major flaw in naive optimizers is allocating capital without considering reorder points $s_i$. If allocated initial inventory $I_{0,i} = \lfloor \frac{w_i B}{c_i} \rfloor$ falls below reorder point $s_i$, the simulation is forced into an immediate Day 1 stockout.

   Optera prevents this by embedding a **proportional lead-time shortage penalty**:
   $$\mathcal{S}(w) = \psi \cdot \bar{P}_{\text{cat}} \sum_{i=1}^N \max\left(0, \frac{s_i - I_{0,i}}{s_i}\right)$$
   > **Variable Definitions**:
   > - $\mathcal{S}(w)$: Inventory policy-aware lead-time shortage penalty term ($\$$).
   > - $\psi$: Dimensionless shortage penalty multiplier (Default: $\psi = 1.0$).
   > - $\bar{P}_{\text{cat}}$: Average category profit scaling factor ($\$$).
   > - $s_i$: Reorder point inventory threshold for category $i$ (units).
   > - $I_{0,i} = \lfloor \frac{w_i B}{c_i} \rfloor$: Initial inventory units affordable under weight $w_i$, total budget $B$, and unit cost $c_i$.
   > - $\max\left(0, \frac{s_i - I_{0,i}}{s_i}\right)$: Normalized proportional shortfall ratio measuring how far initial stock $I_{0,i}$ is below reorder point $s_i$.

#### Simplex Box Projection ($\mathcal{P}_{\Delta}$)
At each particle velocity step, continuous weights are projected onto the constrained budget simplex ($w_i \in [w_{\min}, w_{\max}], \sum w_i = 1.0$):
$$w_i^{(k+1)} = \mathcal{P}_{\Delta}\left( w_i^{(k)} + v_i^{(k+1)} \right)$$
> **Variable Definitions**:
> - $w_i^{(k+1)}$: Updated budget allocation weight for particle on iteration $k+1$.
> - $w_i^{(k)}$: Current budget allocation weight on iteration $k$.
> - $v_i^{(k+1)}$: Particle velocity vector step computed from cognitive and social swarm vectors.
> - $\mathcal{P}_{\Delta}$: Simplex box projection operator enforcing $w_{\min} \le w_i \le w_{\max}$ and $\sum_{i=1}^N w_i = 1.0$.

---

### Layer 4: Monte Carlo Event-Driven Stochastic Simulator (`optera.simulation`)

The simulation engine runs $M = 1,000$ independent stochastic trials across $T = 90$ days. Each day $t$, daily demand $d_{i,t}^{(m)} \sim f_i^*(x)$ is sampled.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STOCHASTIC DAY-BY-DAY SIMULATION LOOP                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
  Day t starts ➔ 1. Receive Supplier Deliveries (if Lead Time Elapsed)
               ➔ 2. Sample Stochastic Demand d_i,t ~ f_i*(x)
               ➔ 3. Fulfill Orders: Sales = min(On-Hand, Demand), Stockout = Demand - Sales
               ➔ 4. Update Financials: Cash += Sales * Price - On-Hand * Holding
               ➔ 5. Check (s, S) Reorder Trigger:
                    If (On-Hand + On-Order) <= s_i:
                        q_needed     = S_i - InvPosition
                        q_target     = μ_i * T_review
                        q_affordable = floor(Cash / c_i)
                        q_storage    = max(0, C_i - InvPosition)
                        
                        Q_order = min(q_needed, q_target, q_affordable, q_storage)
                        If Q_order >= MOQ_i: Place Supplier Order (Cash -= Q_order * c_i)
```

#### Realistic Inventory Policy Parameters
- **Reorder Point ($s_i$)**:
  $$s_i = \lceil \mu_i \cdot L_i + z \cdot \sigma_i \sqrt{L_i} \rceil$$
  > **Variable Definitions**:
  > - $s_i$: Reorder point inventory level for category $i$ (units).
  > - $\mu_i$: Mean daily demand for category $i$ (units/day).
  > - $L_i$: Supplier replenishment lead time in days (e.g., $L_i = 7.0$ days).
  > - $\mu_i L_i$: Expected lead-time demand volume (units).
  > - $z$: Safety stock z-score coefficient (e.g., $z = 1.645$ for $95\%$ target service level).
  > - $\sigma_i \sqrt{L_i}$: Standard deviation of demand over lead time $L_i$ (units).
  > - $z \cdot \sigma_i \sqrt{L_i}$: Safety stock buffer volume (units).

- **Order-Up-To Level ($S_i$)**:
  $$S_i = s_i + \lceil \mu_i \cdot T_{\text{review}} \rceil$$
  > **Variable Definitions**:
  > - $S_i$: Target order-up-to maximum inventory level for category $i$ (units).
  > - $s_i$: Reorder point inventory level (units).
  > - $T_{\text{review}}$: Configurable review cycle period in days (e.g., $T_{\text{review}} = 14$ days).
  > - $\mu_i T_{\text{review}}$: Expected demand volume over one review cycle (units).

- **Warehouse Capacity Constraint ($C_i$)**:
  $$C_i = \lceil 2.5 \times S_i \rceil$$
  > **Variable Definitions**:
  > - $C_i$: Physical warehouse storage space limit for category $i$ (units).
  > - $2.5$: Physical warehouse capacity expansion factor.
  > - $S_i$: Order-up-to level for category $i$ (units).

- **Minimum Order Quantity (MOQ)**:
  $$\text{Replenish if } Q_{\text{order}} \ge \text{MOQ}_i$$
  > **Variable Definitions**:
  > - $Q_{\text{order}} = \min(q_{\text{needed}}, q_{\text{target}}, q_{\text{affordable}}, q_{\text{storage}})$: Calculated replenishment order size (units).
  > - $\text{MOQ}_i$: Minimum Order Quantity specified by supplier for category $i$ (Default: $\text{MOQ}_i = 100$ units).

#### Risk Metrics (VaR & CVaR)
From the $M$ simulated profit outcomes $\pi^{(1)}, \pi^{(2)}, \dots, \pi^{(M)}$ sorted in ascending order:
- **Value at Risk (VaR 5%)**:
  $$\text{VaR}_{0.05} = -\text{Percentile}(\boldsymbol{\pi}, 5\%)$$
  > **Variable Definitions**:
  > - $\text{VaR}_{0.05}$: $5\%$ Value-at-Risk threshold ($\$$), representing maximum cumulative loss expected at $95\%$ confidence level.
  > - $\boldsymbol{\pi} = [\pi^{(1)}, \pi^{(2)}, \dots, \pi^{(M)}]$: Vector of total cumulative net profit outcomes across all $M = 1,000$ simulation trials.

- **Conditional Value at Risk (CVaR 5% / Expected Shortfall)**:
  $$\text{CVaR}_{0.05} = \frac{1}{\lfloor 0.05 M \rfloor} \sum_{k=1}^{\lfloor 0.05 M \rfloor} \pi^{(k)}$$
  > **Variable Definitions**:
  > - $\text{CVaR}_{0.05}$: Conditional Value-at-Risk / Expected Shortfall ($\$$), representing the mean net cumulative profit across the worst $5\%$ simulation paths.
  > - $\lfloor 0.05 M \rfloor$: Number of worst-case trial paths (e.g., 50 worst trials out of $M = 1,000$).
  > - $\pi^{(k)}$: Trial cumulative net profit sorted in ascending order.

---

## 💻 Quickstart & API Usage (`import optera`)

### Option A: Single Integrated Execution (`optera.run`)

```python
import optera
from pathlib import Path

# Run entire integrated framework in 1 call
master_report = optera.run(
    data_path="./demand_forecasting.csv",
    output_dir=".",
    column_mapping={
        "Date": "date",
        "Product ID": "sku_id",
        "Category": "category",
        "Units Sold": "quantity",
        "Price": "unit_price",
        "Inventory Level": "inventory_level",
        "Promotion": "promotion_flag"
    },
    total_budget=500000.0,
    num_simulations=1000,
    horizon_days=90
)
```

### Option B: Modular Layer-by-Layer Execution

```python
import optera

# 1. ETL Layer
etl_res = optera.etl(
    input_csv="./demand_forecasting.csv",
    procurement_multiplier=0.70,
    holding_rate=0.15,
    output_dir="."
)

# 2. Demand Analytics Layer
analytics_res = optera.analytics(
    distributions=["normal", "poisson", "gamma", "nbinom"],
    confidence_levels=[0.95, 0.99],
    goodness_of_fit_metric="aic",
    output_dir="."
)

# 3. Optimization Layer (Particle Swarm Optimization)
opt_res = optera.optimize(
    total_budget=500000.0,
    swarm_size=50,
    max_iterations=150,
    lambda_risk=0.0001,
    alpha_diversification=0.10,
    shortage_penalty_weight=1.0,
    output_dir="."
)

# 4. Monte Carlo Simulation Layer
sim_res = optera.simulation(
    num_simulations=1000,
    horizon_days=90,
    total_budget=500000.0,
    initial_cash=100000.0,
    safety_stock_z=1.645,
    review_cycle_days=14,
    min_order_quantity=100.0,
    output_dir="."
)
```

---

## 📖 Mathematical Notation & Variable Dictionary

| Symbol | Mathematical Definition | Default Value / Unit | Description |
| :--- | :--- | :--- | :--- |
| $B$ | Total Procurement Capital | $\$500,000.00$ | Total available budget for inventory allocation |
| $w_i$ | Budget Allocation Weight | $w_i \in [0.05, 0.40]$ | Fraction of total budget $B$ allocated to category $i$ |
| $p_i$ | Mean Selling Price | $\$/\text{unit}$ | Average retail selling price per unit for category $i$ |
| $c_i$ | Mean Procurement Cost | $c_i = 0.70 \cdot p_i$ | Average unit acquisition cost from supplier |
| $h_i$ | Daily Holding Cost | $\$/\text{unit}/\text{day}$ | Daily cost of holding one unit in inventory |
| $\mu_i$ | Mean Daily Demand | $\text{units}/\text{day}$ | Expected daily customer demand |
| $\sigma_i$ | Demand Standard Deviation | $\text{units}$ | Standard deviation of daily demand |
| $\boldsymbol{\Sigma}$ | Covariance Matrix | $\mathbb{R}^{N \times N}$ | Inter-category demand covariance matrix |
| $\lambda$ | Risk Aversion Weight | $0.0001$ | Markowitz portfolio variance risk penalty weight |
| $\alpha$ | Diversification Factor | $0.10$ | Herfindahl concentration penalty factor |
| $\psi$ | Shortage Penalty Weight | $1.0$ | Proportional lead-time shortage penalty weight |
| $L_i$ | Supplier Lead Time | $7.0 \text{ days}$ | Time between order placement and inventory arrival |
| $z$ | Safety Stock Z-Score | $1.645$ | Multiplier for target service level ($95\%$) |
| $s_i$ | Reorder Point | $\text{units}$ | Inventory level threshold triggering replenishment |
| $S_i$ | Order-Up-To Level | $\text{units}$ | Target inventory restoration level |
| $T_{\text{review}}$ | Review Cycle Period | $14 \text{ days}$ | Periodic review cycle timeframe |
| $\text{MOQ}_i$ | Minimum Order Quantity | $100 \text{ units}$ | Minimum replenishment batch size from supplier |
| $C_i$ | Storage Capacity | $2.5 \cdot S_i$ | Maximum warehouse physical storage capacity |

---

## 📄 License & Citation

Optera is open-source under the **MIT License**.

If you use Optera in academic research or industrial projects, please cite:

```bibtex
@software{optera_supply_chain_2026,
  author = {Optera Core Development Team},
  title = {Optera: High-Performance Swarm-Optimized Supply Chain Simulation Engine},
  year = {2026},
  publisher = {GitHub},
  journal = {Final Year FYP Quantitative Research Project},
  url = {https://github.com/Niroshan-k/Event-Driven-Simulation-Framework}
}
```