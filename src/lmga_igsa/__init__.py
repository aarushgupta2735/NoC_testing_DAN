"""LMGA_IGSA: neural-guided genetic algorithm for core-to-IO mapping
with a learned, conflict-aware preemption policy (v2 architecture —
see model_v1_backup.py for the retired LSTM version)."""

from .model import PointerNetV2
from .ga import Individual, Population
from .run import run_lmga

__all__ = ["PointerNetV2", "Individual", "Population", "run_lmga"]

__version__ = "0.1.0"