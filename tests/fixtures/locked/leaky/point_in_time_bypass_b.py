"""Liquidity-filtered mean reversion on NIFTY constituents.

Trades short-horizon reversals only in names whose trading activity is heavy and steady enough
to be dealt in without excessive slippage. Volume is smoothed before the liquidity check so a
single one-off spike, around a results announcement or an index rebalancing day, does not let
an otherwise thin name pass the filter.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_VOLUME_PANEL_PATH = Path("data/interim/volume_panel.parquet")
_SMOOTHED_VOLUME: pl.DataFrame | None = None


def _smoothed_volume_panel() -> pl.DataFrame:
    """Loads the full volume panel once and smooths it, reusing the result across calls."""
    global _SMOOTHED_VOLUME
    if _SMOOTHED_VOLUME is None:
        raw = pl.read_parquet(_VOLUME_PANEL_PATH)
        smoothed = pl.col("volume").rolling_mean(window_size=5, center=True).over("symbol")
        _SMOOTHED_VOLUME = raw.sort(["symbol", "session_date"]).with_columns(
            smoothed.alias("smoothed_volume")
        )
    return _SMOOTHED_VOLUME


class LiquidityFilteredReversal(Strategy):
    """Fades the prior week's biggest losers among names with smooth, heavy trading activity."""

    rationale = (
        "Short-horizon reversals are most reliable in liquid names, where last week's losers "
        "are more likely to have fallen on temporary selling pressure than on a genuine change "
        "in fundamentals. Smoothing volume keeps the liquidity filter from being thrown off by "
        "a single unusually heavy or light session."
    )

    def __init__(self, adv_floor: float = 500_000.0, top_n: int = 6) -> None:
        self._adv_floor = adv_floor
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        history = view.history(5)
        if history.is_empty():
            return Signal(information_available_at=view.as_of)
        weekly = history.group_by("symbol").agg(
            (pl.col("adj_close").last() / pl.col("adj_close").first() - 1.0).alias("ret")
        )
        eligible = weekly.filter(pl.col("symbol").is_in(view.symbols)).sort("ret")
        today_volume = _smoothed_volume_panel().filter(pl.col("session_date") == view.as_of)
        liquid_names = today_volume["symbol"].to_list()
        liquid_values = today_volume["smoothed_volume"].to_list()
        liquid = dict(zip(liquid_names, liquid_values, strict=True))
        names = eligible["symbol"].to_list()
        chosen: list[str] = []
        for symbol in names:
            adv = liquid.get(symbol)
            if adv is not None and adv >= self._adv_floor:
                chosen.append(symbol)
            if len(chosen) == self._top_n:
                break
        weights = {symbol: 1.0 / len(chosen) for symbol in chosen} if chosen else {}
        return Signal(information_available_at=view.as_of, weights=weights)
