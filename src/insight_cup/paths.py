"""Canonical project paths shared by runtime and maintenance tools."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
TRAINING_ROOT = PROJECT_ROOT / "training"

