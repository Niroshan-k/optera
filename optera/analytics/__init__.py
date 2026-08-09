"""
Optera Demand Analytics Package

Provides category-wise statistical demand profiling, hypothesis testing,
intermittent demand classification (Croston/Syntetos-Boylan framework),
distribution fitting, and portfolio-style risk metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src package is in path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analytics.profiler import CategoryDemandProfiler, export_demand_model
from analytics.reporter import AnalyticsReporter

__all__ = ["CategoryDemandProfiler", "export_demand_model", "AnalyticsReporter"]
