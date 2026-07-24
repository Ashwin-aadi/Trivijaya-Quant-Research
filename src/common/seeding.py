"""Global seeding, so every stochastic step is reproducible.

The charter mandates a single global seed (42) and forbids unseeded randomness. This module seeds
Python's ``random`` and NumPy in one call; any library added later that carries its own RNG
(e.g. a gradient-boosting or torch model) must be seeded explicitly at its call site and the seed
recorded in the run manifest — a global switch that silently covers new libraries is exactly the
kind of hidden state that breaks reproducibility.
"""

from __future__ import annotations

import random

import numpy as np

from src.common.log import get_logger

_log = get_logger(__name__)


def seed_everything(seed: int) -> int:
    """Seed Python and NumPy RNGs. Returns the seed so callers can log it in the manifest."""
    random.seed(seed)
    np.random.seed(seed)
    _log.info("seeded Python and NumPy RNGs with %d", seed)
    return seed
