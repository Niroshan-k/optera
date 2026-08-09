from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("OpteraSchema")


class Schema:

    """
    Optera Standard Schema

    Maps user dataset columns into Optera's standard schema
    and validates that essential columns exist, while logging
    warnings for missing derivable standard columns.
    """

    # ==========================================================
    # Optera Standard Schema Definitions
    # ==========================================================

    STANDARD_COLUMNS = {
        "date": "datetime",
        "sku_id": "string",
        "quantity": "numeric",
        "unit_price": "numeric",
        "category": "string",
        "unit_cost": "numeric",
        # Optional / Derivable
        "transaction": "string",
        "inventory_level": "numeric",
        "lead_time": "numeric",
        "holding_cost": "numeric",
        "promotion_flag": "numeric"
    }

    # Essential columns: required for fundamental time-series processing
    ESSENTIAL_COLUMNS = [
        "date",
        "sku_id",
        "quantity",
        "unit_price"
    ]

    # Derivable standard columns: can be calculated/generated in feature_engineering
    DERIVABLE_COLUMNS = [
        "category",
        "unit_cost",
        "transaction",
        "inventory_level",
        "lead_time",
        "holding_cost",
        "promotion_flag"
    ]

    # ==========================================================

    def __init__(self):
        self.mapping = {}

    # ==========================================================

    def define(self, mapping: Dict[str, str]):
        """
        Register column mapping from user dataset column names to Optera standard schema.

        Example
        -------
        schema.define({
            "Order Date": "date",
            "Product ID": "sku_id",
            "Demand": "quantity",
            "Price": "unit_price"
        })
        """
        for source, target in mapping.items():
            if target not in self.STANDARD_COLUMNS:
                raise ValueError(
                    f"'{target}' is not a valid Optera standard column. "
                    f"Allowed standard columns: {list(self.STANDARD_COLUMNS.keys())}"
                )
            self.mapping[source] = target

    # ==========================================================

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply mapping to dataframe and run schema validation checks.
        """
        if not self.mapping:
            raise RuntimeError(
                "No schema mapping has been defined. Call schema.define() first."
            )

        df = df.copy()

        df.rename(
            columns=self.mapping,
            inplace=True
        )

        self.__check_columns(df)

        self.__validate_types(df)

        return df

    # ==========================================================

    def __check_columns(self, df: pd.DataFrame):
        missing_essential = [col for col in self.ESSENTIAL_COLUMNS if col not in df.columns]

        if missing_essential:
            logger.error("Missing essential Optera columns: %s", missing_essential)
            raise ValueError(
                f"Missing required essential Optera columns: {missing_essential}\n"
                f"Please ensure your dataset or schema mapping defines these essential columns."
            )

        missing_derivable = [col for col in self.DERIVABLE_COLUMNS if col not in df.columns]
        for col in missing_derivable:
            logger.warning(
                "Missing standard column '%s'. Ensure this column is synthesized during Feature Engineering if required downstream.",
                col
            )

    # ==========================================================

    def __validate_types(self, df: pd.DataFrame):
        for column, expected in self.STANDARD_COLUMNS.items():
            if column not in df.columns:
                continue

            try:
                if expected == "datetime":
                    df[column] = pd.to_datetime(
                        df[column],
                        errors="raise"
                    )

                elif expected == "numeric":
                    df[column] = pd.to_numeric(
                        df[column],
                        errors="raise"
                    )

                elif expected == "string":
                    df[column] = df[column].astype(str)

            except Exception as e:
                actual = df[column].dtype
                raise TypeError(
                    f"\nSchema validation failed for column '{column}'.\n"
                    f"Expected type : {expected}\n"
                    f"Detected type : {actual}\n"
                    f"Error details : {str(e)}"
                )

    # ==========================================================

    def show_standard_schema(self):
        print("\n" + "=" * 60)
        print("Optera Standard Schema")
        print("=" * 60)

        for col, dtype in self.STANDARD_COLUMNS.items():
            if col in self.ESSENTIAL_COLUMNS:
                status = "Essential (Required)"
            else:
                status = "Derivable / Optional"

            print(f"{col:20} {dtype:12} {status}")

        print("=" * 60 + "\n")
