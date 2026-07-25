"""Helpers shared by the clean fixtures.

Every strategy needs the same two things: the timestamp of the most recent session it is allowed
to have seen, and a way to turn a list of picks into weights. Writing them once keeps each fixture
to its actual idea, and means the point-in-time stamping is correct in one place rather than
twenty.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from src.backtest.strategy import MarketView

# Returned when a strategy has no usable history. Any date far in the past satisfies the engine's
# "strictly before the fill" rule while carrying an empty portfolio.
_NO_DATA = date(1900, 1, 1)


def latest_visible(view: MarketView) -> date:
    """The most recent session in the visible window.

    This — never ``view.as_of`` — is the correct information timestamp. ``as_of`` is the session
    being traded, so claiming it would assert knowledge of the future and the engine would reject
    the order.
    """
    history = view.history()
    if history.is_empty():
        return _NO_DATA
    latest = history["session_date"].max()
    assert isinstance(latest, date)
    return latest


def equal_weight(symbols: Sequence[str]) -> dict[str, float]:
    """Spread one unit of capital evenly. Empty in, empty out."""
    if not symbols:
        return {}
    weight = 1.0 / len(symbols)
    return dict.fromkeys(symbols, weight)


def window_return(view: MarketView, lookback: int) -> dict[str, float]:
    """Total return per symbol over the last ``lookback`` visible sessions."""
    closes = view.closes(lookback=lookback + 1)
    if closes.height < 2:
        return {}
    out: dict[str, float] = {}
    for symbol in view.symbols:
        if symbol not in closes.columns:
            continue
        series = closes[symbol].drop_nulls()
        if series.len() < 2 or series[0] <= 0:
            continue
        out[symbol] = series[-1] / series[0] - 1.0
    return out


def daily_returns(view: MarketView, lookback: int) -> dict[str, list[float]]:
    """Session-over-session returns per symbol across the visible window."""
    closes = view.closes(lookback=lookback + 1)
    if closes.height < 2:
        return {}
    out: dict[str, list[float]] = {}
    for symbol in view.symbols:
        if symbol not in closes.columns:
            continue
        values = closes[symbol].drop_nulls().to_list()
        if len(values) < 2:
            continue
        out[symbol] = [
            values[i] / values[i - 1] - 1.0
            for i in range(1, len(values))
            if values[i - 1] > 0
        ]
    return out


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation; zero when there is nothing to disperse."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return float(variance**0.5)


def top_n(scores: dict[str, float], count: int, *, largest: bool = True) -> list[str]:
    """The ``count`` best-scoring symbols, ranked deterministically."""
    if not scores:
        return []
    # Symbol breaks ties so the selection is reproducible run to run.
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1] if largest else kv[1], kv[0]))
    return [symbol for symbol, _ in ordered[:count]]


def column_or_none(frame: pl.DataFrame, name: str) -> pl.Series | None:
    """Fetch a column if the pivot produced one for this symbol."""
    return frame[name] if name in frame.columns else None
