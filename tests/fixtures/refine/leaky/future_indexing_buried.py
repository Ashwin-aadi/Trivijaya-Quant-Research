"""Category ``future_indexing``, variant ``buried``.

Defect: one line in ``_index_level`` reads the benchmark bar stamped ``view.as_of`` — the session
the engine is about to fill. The index close for that session does not exist at the moment the
decision is taken, so the relative-strength hurdle the whole strategy is built around is computed
against a number from the future.

Everything else here is ordinary: a trailing-return screen, a dispersion filter, a warm-up guard,
and volatility-scaled weights. The defect is one filter comparison inside a helper, surrounded by
code that does exactly what it says. This variant measures whether detection survives the cheat not
being the most interesting thing in the file.

The author's mistake is a common one: index levels feel like public reference data, so reaching for
"the index close on the day I trade" does not feel like reading the future the way reaching for a
stock's own close does.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthScreen(Strategy):
    """Holds names outperforming the index, sized inversely to their own volatility."""

    rationale = (
        "Relative strength against the broad index separates names that are rising because the "
        "whole market is rising from names that are rising on their own account. Only the second "
        "kind is worth paying costs for, so the screen keeps names whose trailing return beats the "
        "index over the same window, and sizes them inversely to volatility so that one violent "
        "name cannot dominate the book."
    )

    def __init__(
        self,
        benchmark: pl.DataFrame,
        lookback: int = 63,
        top_n: int = 8,
        min_dispersion: float = 0.005,
    ) -> None:
        self._benchmark = benchmark
        self._lookback = lookback
        self._top_n = top_n
        self._min_dispersion = min_dispersion

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=self._lookback + 1)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        trailing = self._trailing_returns(view, closes)
        if not trailing:
            return Signal(information_available_at=stamp, weights={})

        hurdle = self._index_return(view, closes)
        beating = {s: r for s, r in trailing.items() if r > hurdle}
        if not beating:
            return Signal(information_available_at=stamp, weights={})

        ranked = sorted(beating.items(), key=lambda kv: (-kv[1], kv[0]))[: self._top_n]
        risk = {s: self._volatility(closes, s) for s, _ in ranked}
        return Signal(information_available_at=stamp, weights=_risk_weights(risk))

    def _trailing_returns(self, view: MarketView, closes: pl.DataFrame) -> dict[str, float]:
        """Total return per symbol over the lookback, skipping names that are too quiet to trade."""
        out: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback or values[0] <= 0:
                continue
            span = max(values) - min(values)
            if span / values[0] < self._min_dispersion:
                # Flat names produce noise-dominated rankings and churn the book for nothing.
                continue
            out[symbol] = values[-1] / values[0] - 1.0
        return out

    def _index_return(self, view: MarketView, closes: pl.DataFrame) -> float:
        """Index return over the same window, used as the hurdle every name must clear."""
        opening = self._index_level(view, _first_session(closes))
        closing = self._index_level(view, view.as_of)
        if opening is None or closing is None or opening <= 0:
            return 0.0
        return closing / opening - 1.0

    def _index_level(self, view: MarketView, when: date) -> float | None:
        """Benchmark close on ``when``, or ``None`` when the index did not trade that session."""
        # THE CHEAT: `when` is `view.as_of` on the second call above — the session the engine fills.
        # That bar has not printed when this decision is made, so the hurdle is computed against a
        # future index level. No timestamp is violated and the signal is stamped honestly; the leak
        # is entirely in which row of the benchmark frame is read.
        row = self._benchmark.filter(pl.col("session_date") == when)
        if row.is_empty():
            return None
        return float(row["adj_close"][0])

    @staticmethod
    def _volatility(closes: pl.DataFrame, symbol: str) -> float:
        """Standard deviation of session returns, floored so the reciprocal stays finite."""
        values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
        steps = [
            values[i] / values[i - 1] - 1.0 for i in range(1, len(values)) if values[i - 1] > 0
        ]
        if len(steps) < 2:
            return 1.0
        mean = sum(steps) / len(steps)
        variance = sum((s - mean) ** 2 for s in steps) / (len(steps) - 1)
        return max(float(variance**0.5), 1e-6)


def _risk_weights(risk: dict[str, float]) -> dict[str, float]:
    """Inverse-volatility weights, normalised to one unit of capital."""
    if not risk:
        return {}
    raw = {symbol: 1.0 / vol for symbol, vol in risk.items()}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {symbol: value / total for symbol, value in raw.items()}


def _first_session(closes: pl.DataFrame) -> date:
    """Oldest session in the pivoted close frame."""
    oldest = closes["session_date"].min()
    assert isinstance(oldest, date)
    return oldest


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
