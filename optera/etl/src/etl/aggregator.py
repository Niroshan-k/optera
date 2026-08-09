"""
Optera Demand Aggregator Module

Aggregates transactional datasets into daily time-series aggregated by
Category and SKU, serving as the foundational input for statistical demand
modeling and stochastic simulations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger("OpteraAggregator")


class DemandAggregator:
    """
    Aggregates transactional data into time-series demand formats.
    """

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, (str, Path)):
            path = Path(data)
            self.df = pd.read_csv(path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Input must be a CSV file path or pandas DataFrame.")

        self.df["date"] = pd.to_datetime(self.df["date"])

    def aggregate_by_category(self, fill_zero_demand_dates: bool = True) -> pd.DataFrame:
        logger.info("Aggregating transaction data into daily category demand time-series...")

        self.df["date"] = self.df["date"].dt.floor("D")

        agg_rules = {
            "quantity": "sum",
            "unit_price": "mean"
        }
        if "revenue" in self.df.columns:
            agg_rules["revenue"] = "sum"
        if "unit_cost" in self.df.columns:
            agg_rules["unit_cost"] = "mean"
        if "holding_cost" in self.df.columns:
            agg_rules["holding_cost"] = "mean"
        if "lead_time" in self.df.columns:
            agg_rules["lead_time"] = "mean"

        grouped = self.df.groupby(["date", "category"]).agg(agg_rules).reset_index()

        if fill_zero_demand_dates and not grouped.empty:
            categories = grouped["category"].unique()
            min_date = grouped["date"].min()
            max_date = grouped["date"].max()

            full_grid = pd.MultiIndex.from_product(
                [pd.date_range(min_date, max_date, freq="D"), categories],
                names=["date", "category"]
            ).to_frame().reset_index(drop=True)

            grouped = pd.merge(full_grid, grouped, on=["date", "category"], how="left")
            grouped["quantity"] = grouped["quantity"].fillna(0.0)
            if "revenue" in grouped.columns:
                grouped["revenue"] = grouped["revenue"].fillna(0.0)

            for col in ["unit_price", "unit_cost", "holding_cost", "lead_time"]:
                if col in grouped.columns:
                    grouped[col] = grouped.groupby("category")[col].ffill().bfill()

        grouped = grouped.sort_values(by=["category", "date"]).reset_index(drop=True)
        logger.info("Demand aggregation complete. Category series records: %d", len(grouped))
        return grouped

    def save(self, df: pd.DataFrame, output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved aggregated category demand to: %s", output_path)


def aggregate_to_category(df: pd.DataFrame) -> pd.DataFrame:
    aggregator = DemandAggregator(df)
    return aggregator.aggregate_by_category()
