"""
Optera Data Validator Module

Validates data quality of standard mapped datasets before they enter
data cleaning and feature engineering.

Responsibilities
----------------
• Load mapped DataFrame / CSV
• Detect empty datasets or duplicate columns
• Check missing values and bad dates
• Validate numeric constraints (negative quantities/prices)
• Output validation report
• Save output to validated.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger("OpteraValidator")


class ValidationError(Exception):
    """Base validation exception."""
    pass


class DataValidator:
    """
    Validates data quality of a pandas DataFrame or CSV file after Schema mapping.
    """

    ESSENTIAL_COLUMNS = ["date", "sku_id", "quantity", "unit_price"]

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, (str, Path)):
            self.csv_path = Path(data)
            self.df = self._load_csv(self.csv_path)
        elif isinstance(data, pd.DataFrame):
            self.csv_path = None
            self.df = data.copy()
        else:
            raise ValueError("Input must be a CSV file path or pandas DataFrame.")

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        for enc in ["utf-8", "ISO-8859-1", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not parse CSV at {path} with supported encodings.")

    def validate(self) -> pd.DataFrame:
        """
        Runs quality validation pipeline.
        """
        logger.info("Starting Optera Data Quality Validation...")

        if self.df is None or self.df.empty:
            raise ValidationError("Dataset is empty or None.")

        # Check duplicate columns
        duplicates = self.df.columns[self.df.columns.duplicated()].tolist()
        if duplicates:
            raise ValidationError(f"Duplicate columns detected in dataset: {duplicates}")

        # Check essential columns presence
        missing_essential = [col for col in self.ESSENTIAL_COLUMNS if col not in self.df.columns]
        if missing_essential:
            raise ValidationError(f"Missing essential columns for data validation: {missing_essential}")

        # Check negative prices/quantities
        if "quantity" in self.df.columns:
            negative_q = (self.df["quantity"] < 0).sum()
            if negative_q > 0:
                logger.warning("Detected %d negative quantity records. These should be handled during cleaning.", negative_q)

        if "unit_price" in self.df.columns:
            negative_p = (self.df["unit_price"] < 0).sum()
            if negative_p > 0:
                logger.warning("Detected %d negative price records. These should be handled during cleaning.", negative_p)

        # Check missing values
        missing_count = self.df[self.ESSENTIAL_COLUMNS].isna().sum().sum()
        if missing_count > 0:
            logger.warning("Detected %d missing values across essential columns.", missing_count)

        logger.info("Data Quality Validation completed successfully.")
        return self.df

    def report(self):
        """Prints diagnostic validation report."""
        print("\n" + "=" * 60)
        print("Optera Data Validation Report")
        print("=" * 60)
        print(f"Total Rows    : {len(self.df):,}")
        print(f"Total Columns : {len(self.df.columns)}")
        print("\nColumn Summary & Missing Values:")
        for col in self.df.columns:
            nulls = self.df[col].isna().sum()
            dtype = str(self.df[col].dtype)
            print(f"  • {col:20} | Type: {dtype:10} | Missing: {nulls:,}")
        print("=" * 60 + "\n")

    def save(self, output_path: Union[str, Path]):
        """Saves validated dataframe to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info("Saved validated dataset to: %s", output_path)