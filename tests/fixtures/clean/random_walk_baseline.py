"""Pick names at random from a fixed seed, as a null-hypothesis baseline."""

from __future__ import annotations

import random

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible

# Project-wide seed. Every stochastic operation in this repository takes an explicit seed; an
# unseeded baseline would produce a different answer on every run and could not be compared
# against anything.
SEED = 42


class RandomWalkBaseline(Strategy):
    """Selects a random subset each rebalance, reproducibly."""

    rationale = (
        "A random portfolio is the honest null: any strategy that cannot beat one has "
        "demonstrated no skill. It also exposes how much of a backtest's spread comes from "
        "selection versus from the market itself."
    )

    def __init__(self, holdings: int = 10, seed: int = SEED) -> None:
        self._holdings = holdings
        self._seed = seed

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        candidates = sorted(view.symbols)
        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        # Seeded per decision date rather than once at construction, so the sequence of picks does
        # not depend on how many times the strategy has previously been called. The same date
        # always yields the same portfolio, whatever order the backtest runs in.
        rng = random.Random(f"{self._seed}-{stamp.isoformat()}")
        count = min(self._holdings, len(candidates))
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(rng.sample(candidates, count)),
        )
