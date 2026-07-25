"""Cheat: standardises features using statistics computed over the entire sample.

The subtlest of the three, and the most common in practice. The trading rule itself is defensible
— buy names whose momentum z-score is high — but the mean and standard deviation used to compute
that z-score are fitted once over the whole history, including the future. Knowing the full-sample
mean tells the strategy, at every point, whether the present is above or below a level it could not
have known yet.

This is the failure mode that survives casual review: no timestamp is violated, no future price is
read directly, and the backtest loop looks ordinary. Only the fitting step is wrong. It is exactly
the pattern a static auditor must catch by inspecting *where* a transform was fitted, not what the
trading rule does.

Expected behaviour: a strong but not cartoonish result — noticeably better than the honest version
of the same rule, which is what makes it dangerous.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class LeakFullSampleScaler(Strategy):
    """Ranks on a price z-score standardised with full-sample statistics."""

    # The rationale matches what the code does, on purpose. This fixture's only defect is where
    # its scaler was fitted; describing a different strategy would make it leaky for two reasons
    # at once and blur what the auditor is being tested on.
    rationale = (
        "Cross-sectional mean reversion: prices are standardised per name so they can be compared "
        "on one scale, and the portfolio holds those trading furthest below their typical level, "
        "on the expectation that they revert."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        self._top_n = top_n
        # THE CHEAT: fitted once, here, over every session in the panel — including sessions that
        # are still in the future at every decision point the strategy will later face.
        self._scaler = self._fit_full_sample(panel)

    @staticmethod
    def _fit_full_sample(panel: pl.DataFrame) -> dict[str, tuple[float, float]]:
        """Per-symbol mean and standard deviation of the *price level*, over all time.

        Fitted across the entire panel in one pass, exactly as a scaler fitted before splitting
        would be. The resulting statistics encode where each stock's price will eventually sit,
        which is the information the strategy is not entitled to.
        """
        stats = panel.group_by("symbol").agg(
            pl.col("adj_close").mean().alias("mean"),
            pl.col("adj_close").std().alias("std"),
        )
        return {
            row["symbol"]: (row["mean"], row["std"] or 1.0)
            for row in stats.iter_rows(named=True)
        }

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        latest = view.latest_close()
        if not latest:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol, price in latest.items():
            mean, std = self._scaler.get(symbol, (0.0, 1.0))
            if not std:
                continue
            # The standardised price. Because mean and std were fitted over the whole sample,
            # a strongly negative z-score means "cheap relative to where this stock trades across
            # the entire period, future included" — so the strategy systematically buys near
            # bottoms it could not have identified at the time.
            scores[symbol] = (price - mean) / std

        if not scores:
            return Signal(information_available_at=stamp, weights={})
        chosen = sorted(scores, key=lambda s: scores[s])[: self._top_n]
        weight = 1.0 / len(chosen)
        return Signal(information_available_at=stamp, weights=dict.fromkeys(chosen, weight))


def _latest_visible(view: MarketView) -> date:
    history = view.history()
    if history.is_empty():
        return date(1900, 1, 1)
    latest = history["session_date"].max()
    assert isinstance(latest, date)
    return latest
