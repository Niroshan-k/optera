"""
Optera Data Cleaner Module

Cleans validated retail/supply chain transactions to ensure reliable downstream analytical modeling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger("OpteraCleaner")


class DataCleaner:
    """
    Cleans a validated Optera DataFrame.
    """

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, (str, Path)):
            path = Path(data)
            self.df = pd.read_csv(path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Input must be a CSV file path or pandas DataFrame.")

    def clean(
        self,
        remove_duplicates: bool = True,
        filter_negative_quantity: bool = True,
        filter_negative_price: bool = True,
        fill_missing_category: str = "Uncategorized"
    ) -> pd.DataFrame:
        logger.info("Starting Data Cleaning Pipeline...")
        initial_rows = len(self.df)

        if remove_duplicates:
            self.df = self.df.drop_duplicates()
            dupes_removed = initial_rows - len(self.df)
            if dupes_removed > 0:
                logger.info("Removed %d duplicate rows.", dupes_removed)

        if filter_negative_quantity and "quantity" in self.df.columns:
            prev = len(self.df)
            self.df = self.df[self.df["quantity"] > 0]
            removed = prev - len(self.df)
            if removed > 0:
                logger.info("Filtered out %d non-positive quantity rows.", removed)

        if filter_negative_price and "unit_price" in self.df.columns:
            prev = len(self.df)
            self.df = self.df[self.df["unit_price"] >= 0]
            removed = prev - len(self.df)
            if removed > 0:
                logger.info("Filtered out %d negative unit price rows.", removed)

        if "sku_id" in self.df.columns:
            self.df["sku_id"] = self.df["sku_id"].astype(str).str.strip().str.upper()

        if "category" in self.df.columns:
            self.df["category"] = self.df["category"].fillna(fill_missing_category).astype(str).str.strip().str.title()
        else:
            self.df["category"] = fill_missing_category

        essential = [c for c in ["date", "sku_id", "quantity", "unit_price"] if c in self.df.columns]
        prev = len(self.df)
        self.df = self.df.dropna(subset=essential)
        dropped_nulls = prev - len(self.df)
        if dropped_nulls > 0:
            logger.info("Dropped %d rows with missing essential fields.", dropped_nulls)

        logger.info("Data Cleaning complete. Rows retained: %d / %d", len(self.df), initial_rows)
        return self.df

    def save(self, output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info("Saved cleaned dataset to: %s", output_path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaner = DataCleaner(df)
    return cleaner.clean()
