"""Adaptive-horizon momentum rotation for NIFTY constituents.

Different businesses trend at different speeds: some names respond within a month of fresh
news, others only show a durable trend over two or three quarters. Rather than forcing every
stock through the same lookback, this strategy lets each name's own trending behaviour decide
which horizon best captures its momentum, then ranks eligible names on that basis.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_HISTORY_PATH = Path("data/interim/full_price_panel.parquet")
_CANDIDATE_WINDOWS = (21, 63, 126, 252)
_BEST_HORIZON: dict[str, int] | None = None


def _best_horizons() -> dict[str, int]:
    """Determines, per symbol, which candidate window produced the strongest total return."""
    global _BEST_HORIZON
    if _BEST_HORIZON is not None:
        return _BEST_HORIZON
    panel = pl.read_parquet(_HISTORY_PATH).sort(["symbol", "session_date"])
    scores: dict[str, list[tuple[int, float]]] = {}
    for window in _CANDIDATE_WINDOWS:
        tail = panel.group_by("symbol", maintain_order=True).tail(window)
        rets = tail.group_by("symbol").agg(
            (pl.col("adj_close").last() / pl.col("adj_close").first() - 1.0).alias("ret")
        )
        names = rets["symbol"].to_list()
        values = rets["ret"].to_list()
        for symbol, ret in zip(names, values, strict=True):
            scores.setdefault(symbol, []).append((window, ret))
    _BEST_HORIZON = {
        sym: max(cand, key=lambda pair: pair[1])[0] for sym, cand in scores.items()
    }
    return _BEST_HORIZON


class AdaptiveHorizonRotation(Strategy):
    """Ranks names by momentum measured over each name's own best-performing horizon."""

    rationale = (
        "Forcing every stock through the same lookback window ignores that some businesses "
        "trend quickly while others take quarters to show a durable move. Letting each name's "
        "own history indicate which horizon best expresses its momentum should give a cleaner "
        "cross-sectional ranking than one fixed window applied uniformly."
    )

    def __init__(self, top_n: int = 6) -> None:
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        horizons = _best_horizons()
        scored: list[tuple[str, float]] = []
        for symbol in view.symbols:
            window = horizons.get(symbol)
            if window is None:
                continue
            history = view.history(window)
            filtered = history.filter(pl.col("symbol") == symbol).sort("session_date")
            prices = filtered["adj_close"]
            if prices.len() < 2:
                continue
            scored.append((symbol, prices[-1] / prices[0] - 1.0))
        scored.sort(key=lambda row: row[1], reverse=True)
        chosen = scored[: self._top_n]
        weights = {symbol: 1.0 / len(chosen) for symbol, _ in chosen} if chosen else {}
        return Signal(information_available_at=view.as_of, weights=weights)
