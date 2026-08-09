"""
Optera ETL Core Package

Modules
-------
• Schema: Column mapping & schema validation
• DataValidator: Quality checks & duplicate detection
• DataCleaner: Data cleaning & filtering
• FeatureEngineer: Financial feature generation & imputation
• DemandAggregator: Daily time-series demand aggregation
"""

from .Schema import Schema
from .validator import DataValidator
from .cleaner import DataCleaner, clean_dataframe
from .feature_engineering import FeatureEngineer
from .aggregator import DemandAggregator, aggregate_to_category

__all__ = [
    "Schema",
    "DataValidator",
    "DataCleaner",
    "clean_dataframe",
    "FeatureEngineer",
    "DemandAggregator",
    "aggregate_to_category"
]
