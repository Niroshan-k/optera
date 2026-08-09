"""
Optera Supply Chain Inventory Simulation Auditor (v2.3)

Single-category or multi-category day-by-day audit tool printing a detailed trace table:
Day | Demand | Starting Inv | Sales | Lost Sales | Ending Inv | Inv Position | Pending Orders | Cash | Action
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEMAND_MODEL_JSON = PROJECT_ROOT / "optera" / "analytics" / "data" / "demand_model.json"
OPTIMIZATION_RESULTS_JSON = PROJECT_ROOT / "optera" / "optimizers" / "data" / "optimization_results.json"


def audit_simulation(
    category_name: str = "clothing",
    horizon_days: int = 30,
    total_budget: float = 500000.0,
    initial_cash: float = 100000.0,
    safety_stock_z: float = 1.65,
    review_cycle_days: int = 14,
    moq: float = 100.0,
    warehouse_capacity: float | None = None
):
    """
    Runs a deterministic single-trial day-by-day audit for a specific category
    using exact PSO initial inventory allocation and Base-Stock (s, S) min-max reordering
    with MOQ, working capital, and optional warehouse capacity constraints.
    """
    with open(DEMAND_MODEL_JSON, "r", encoding="utf-8") as f:
        demand_model = json.load(f)

    with open(OPTIMIZATION_RESULTS_JSON, "r", encoding="utf-8") as f:
        opt_results = json.load(f)

    categories = [k for k in demand_model.keys() if k != "metadata"]
    if category_name.lower() not in [c.lower() for c in categories]:
        print(f"Category '{category_name}' not found. Available: {categories}")
        category_name = categories[0]

    cat_key = [c for c in categories if c.lower() == category_name.lower()][0]
    info = demand_model[cat_key]
    alloc_w = opt_results.get("allocations_weight", {}).get(cat_key, 0.20)

    fin = info.get("financials", {})
    cost = float(fin.get("unit_cost_mean", 50.0))
    price = float(fin.get("unit_price_mean", 70.0))
    holding = float(fin.get("holding_cost_mean", cost * 0.15)) / 365.0
    lead = max(1.0, float(fin.get("lead_time_mean", 7.0)))

    mean_d = float(info["mean"])
    std_d = float(info["std"])

    # 1. Base-Stock (s, S) Parameters
    lead_demand_std = std_d * np.sqrt(lead)
    safety_stock = safety_stock_z * lead_demand_std
    reorder_point_s = np.ceil(mean_d * lead + safety_stock)
    order_up_to_S = reorder_point_s + np.ceil(mean_d * review_cycle_days)
    max_warehouse_capacity = warehouse_capacity if warehouse_capacity is not None else float("inf")

    # 2. Exact PSO Initial Allocation (I_0)
    cat_budget = total_budget * alloc_w
    initial_units_I0 = float(np.floor(cat_budget / max(1.0, cost)))
    actual_spent = initial_units_I0 * cost

    # 3. Initial Lead-Time Coverage Ratio
    coverage_ratio = initial_units_I0 / max(1.0, reorder_point_s)

    print("\n" + "=" * 115)
    print(f"OPTERA SINGLE-CATEGORY INVENTORY SIMULATION AUDIT (v2.3 - {cat_key.upper()})")
    print("=" * 115)
    print(f"  • Category                         : {info.get('category', cat_key)}")
    print(f"  • PSO Optimal Budget Weight (w*)    : {alloc_w*100:.2f}% (${cat_budget:,.2f})")
    print(f"  • Exact PSO Initial Inv (I_0)      : {initial_units_I0:,.0f} units (${actual_spent:,.2f})")
    print(f"  • Reorder Point (s)                : {reorder_point_s:,.0f} units")
    print(f"  • Initial Lead-Time Coverage Ratio : {coverage_ratio*100:.1f}% ({initial_units_I0:,.0f} / {reorder_point_s:,.0f})")
    print(f"  • Order-Up-To Level (S)            : {order_up_to_S:,.0f} units")
    print(f"  • Max Warehouse Capacity Limit     : {'UNCONSTRAINED (INF)' if max_warehouse_capacity == float('inf') else f'{max_warehouse_capacity:,.0f} units'}")
    print(f"  • Minimum Order Quantity (MOQ)     : {moq:,.0f} units")
    print("=" * 115)

    cash = initial_cash
    on_hand = initial_units_I0
    pending_orders = []  # list of (arrival_day, qty, cost)

    rng = np.random.default_rng(42)

    rows = []

    for t in range(1, horizon_days + 1):
        # 1. Process Arrivals
        arrived_qty = 0.0
        rem_orders = []
        for (arr_day, q_qty, c_cost) in pending_orders:
            if arr_day <= t:
                arrived_qty += q_qty
            else:
                rem_orders.append((arr_day, q_qty, c_cost))
        pending_orders = rem_orders
        starting_inv = on_hand + arrived_qty

        # 2. Stochastic Demand
        demand = np.round(max(0.0, rng.normal(mean_d, std_d)))

        # 3. Sales & Stockout Execution
        sales = min(starting_inv, demand)
        lost_sales = demand - sales
        ending_inv = starting_inv - sales
        on_hand = ending_inv

        # 4. Inventory Position & Base-Stock Reorder Logic
        pending_qty = sum(q for (_, q, _) in pending_orders)
        inv_position = ending_inv + pending_qty

        action = "HOLD"
        if inv_position <= reorder_point_s:
            q_needed = max(0.0, order_up_to_S - inv_position)
            q_target = np.ceil(mean_d * review_cycle_days)
            q_affordable = np.floor(cash / max(1.0, cost))
            q_storage_space = max(0.0, max_warehouse_capacity - inv_position) if max_warehouse_capacity != float("inf") else float("inf")

            q_order = min(q_needed, q_target, q_affordable, q_storage_space)

            if q_order >= moq:
                order_cost = q_order * cost
                cash -= order_cost
                arr_day = t + int(np.round(lead))
                pending_orders.append((arr_day, q_order, order_cost))
                action = f"REORDER (+{q_order:,.0f} units, arr Day {arr_day})"
            elif q_affordable * cost < moq * cost:
                action = f"INSUFFICIENT CASH FOR MOQ ({moq:.0f})"
            elif q_storage_space < moq:
                action = "WAREHOUSE FULL"
            else:
                action = f"BELOW MOQ ({q_order:.0f} < {moq:.0f})"

        rev = sales * price
        hold_cost = ending_inv * holding
        cash += (rev - hold_cost)

        rows.append({
            "Day": t,
            "Demand": f"{demand:,.0f}",
            "Start Inv": f"{starting_inv:,.0f}",
            "Sales": f"{sales:,.0f}",
            "Lost Sales": f"{lost_sales:,.0f}",
            "End Inv": f"{ending_inv:,.0f}",
            "Inv Position": f"{inv_position:,.0f}",
            "Pending Orders": f"{pending_qty:,.0f}",
            "Cash ($)": f"${cash:,.2f}",
            "Action": action
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.to_string(index=False))
    print("=" * 115 + "\n")


if __name__ == "__main__":
    cat = sys.argv[1] if len(sys.argv) > 1 else "clothing"
    audit_simulation(category_name=cat, horizon_days=30)
