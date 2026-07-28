"""A plain equal-weight basket restricted to the most liquid names by trailing volume.

Negative control: average daily volume is computed with a ``group_by("symbol")`` aggregation over
``view.history(adv_window)``, already bounded to sessions strictly before the decision date, and
the candidate set is intersected with ``view.symbols``, the point-in-time investable universe the
engine handed this strategy.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


def _latest_visible_session(view: MarketView) -> date:
    """The newest session already printed to the tape, the only date a signal may cite."""
    recent = view.history(1)
    if recent.is_empty():
        return date.min
    latest = recent["session_date"].max()
    assert isinstance(latest, date)
    return latest


class LiquidityScreenedEqualWeight(Strategy):
    """Equal-weight the most liquid names in the visible universe by trailing ADV."""

    rationale = (
        "Restricting an equal-weight basket to names with sufficient trailing average daily "
        "volume is a standard tradability screen: it keeps the strategy out of names it could "
        "not realistically enter or exit at size, independent of any view on future returns. "
        "This is a liquidity filter, not a return signal."
    )

    def __init__(
        self, adv_window: int = 20, min_adv_shares: float = 100_000.0, max_names: int = 20
    ) -> None:
        self.adv_window = adv_window
        self.min_adv_shares = min_adv_shares
        self.max_names = max_names

    def generate(self, view: MarketView) -> Signal:
        decision_date = _latest_visible_session(view)
        frame = view.history(self.adv_window)
        if frame.is_empty():
            return Signal(information_available_at=decision_date, weights={})
        adv = (
            frame.group_by("symbol")
            .agg(pl.col("volume").mean().alias("average_daily_volume"))
            .filter(pl.col("symbol").is_in(list(view.symbols)))
            .filter(pl.col("average_daily_volume") >= self.min_adv_shares)
            .sort("average_daily_volume", descending=True)
            .head(self.max_names)
        )
        symbols = adv["symbol"].to_list()
        weights: dict[str, float] = {}
        if symbols:
            share = 1.0 / len(symbols)
            for symbol in symbols:
                weights[symbol] = share
        return Signal(information_available_at=decision_date, weights=weights)
