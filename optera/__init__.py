"""Optera Quantitative Supply Chain Package Initialization."""

from optera.etl.main import run_pipeline as etl
from optera.analytics.main import run_analytics as analytics
from optera.optimizers.main import run_optimization as optimization
from optera.optimizers.main import run_optimization as optimize
from optera.simulation.main import run_simulation as simulation
from optera.simulation.audit import audit_simulation as audit
from optera.main import run
from optera.utils.ascii_art import print_ascii_banner

# Auto-print high-impact ASCII Logo Banner ONCE when Optera starts
try:
    print_ascii_banner(version="1.0.2")
except Exception:
    pass

__version__ = "1.0.2"

__all__ = [
    "run",
    "etl",
    "analytics",
    "optimization",
    "optimize",
    "simulation",
    "audit",
    "print_ascii_banner"
]
