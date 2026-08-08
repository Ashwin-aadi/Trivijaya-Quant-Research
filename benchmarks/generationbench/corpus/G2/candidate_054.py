from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength tend to continue their momentum over time. "
        "By identifying stocks that are outperforming the broad market, we can capture this "
        "momentum effect."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        # Compute relative strength
        rel_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            # Get the closes and their ranks within the window
            adj_closes = [float(v) for v in closes[symbol].to_list()]
            rank = (pl.DataFrame({"adj_close": adj_closes}).with_columns(
                pl.col("adj_close").rank(method="ordinal", descending=True)
            )["adj_close"].to_list())[0]

            # Compute the average close
            avg_close = sum(adj_closes) / len(adj_closes)

            # Compute relative strength as rank divided by average price
            rel_strength = rank / avg_close

            if rel_strength > 1.0:  # Only consider stocks with positive relative strength
                rel_strengths[symbol] = rel_strength

        # Sort symbols by their relative strengths in descending order
        sorted_symbols = sorted(rel_strengths, key=lambda k: rel_strengths[k], reverse=True)

        # Select the top performing stocks
        picks = sorted_symbols[:5]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest