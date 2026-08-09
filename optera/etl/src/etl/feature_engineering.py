"""
Optera Feature Engineering Module

Generates derived financial and operational variables required by
optimization algorithms (PSO) and Monte Carlo simulations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger("OpteraFeatureEngineering")


class FeatureEngineer:
    """
    Feature engineering pipeline for Optera datasets.
    """

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, (str, Path)):
            path = Path(data)
            self.df = pd.read_csv(path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Input must be a CSV file path or pandas DataFrame.")

    def transform(
        self,
        procurement_multiplier: float = 0.70,
        holding_rate: float = 0.15,
        default_lead_time_days: float = 7.0
    ) -> pd.DataFrame:
        logger.info("Starting Feature Engineering Pipeline...")

        if "quantity" in self.df.columns and "unit_price" in self.df.columns:
            self.df["revenue"] = self.df["quantity"] * self.df["unit_price"]
            logger.info("Generated 'revenue' feature (quantity * unit_price).")

        if "unit_cost" not in self.df.columns or self.df["unit_cost"].isna().all():
            logger.warning(
                "'unit_cost' column missing or empty. Synthesizing unit_cost = unit_price * %.2f",
                procurement_multiplier
            )
            self.df["unit_cost"] = self.df["unit_price"] * procurement_multiplier
        else:
            self.df["unit_cost"] = self.df["unit_cost"].fillna(self.df["unit_price"] * procurement_multiplier)

        if "holding_cost" not in self.df.columns or self.df["holding_cost"].isna().all():
            logger.info("Synthesizing 'holding_cost' = unit_cost * %.2f", holding_rate)
            self.df["holding_cost"] = self.df["unit_cost"] * holding_rate
        else:
            self.df["holding_cost"] = self.df["holding_cost"].fillna(self.df["unit_cost"] * holding_rate)

        if "lead_time" not in self.df.columns or self.df["lead_time"].isna().all():
            logger.info("Synthesizing 'lead_time' with default value of %.1f days", default_lead_time_days)
            self.df["lead_time"] = default_lead_time_days
        else:
            self.df["lead_time"] = self.df["lead_time"].fillna(default_lead_time_days)

        logger.info("Feature Engineering complete.")
        return self.df

    def save(self, output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info("Saved feature engineered dataset to: %s", output_path)
