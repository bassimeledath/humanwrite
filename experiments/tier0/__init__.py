"""Tier-0 training-side metrics and diagnostics."""

from .metrics import TRAINING_ONLY_NOTICE, batch_diagnostics, distribution_gap

__all__ = ["TRAINING_ONLY_NOTICE", "batch_diagnostics", "distribution_gap"]

