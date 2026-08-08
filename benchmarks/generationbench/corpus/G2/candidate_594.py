from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that have outperformed their peers in recent history tend to continue "
        "outperforming them over the next few days. This strategy seeks to identify and "
        "capitalize on such relative strengths."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        relative_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in history["symbol"].to_list():
                continue

            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._window:
                continue

            current_close = view.latest_close()[symbol]
            past_closes = [
                float(row["adj_close"])
                for row in history.filter(pl.col("symbol") == symbol).sort(
                    "session_date", descending=False
                ).rows()
            ]
            if len(past_closes) < self._lookback:
                continue

            recent_return = (current_close - recent_closes[-1]) / recent_closes[-1]
            past_return = sum((pct_change / 100.0 for pct_change in past_closes)) / len(
                past_closes
            )
            relative_strength = (recent_return + 1) / (past_return + 1)
            if not pl.is_nan(relative_strength):
                relative_strengths[symbol] = relative_strength

        sorted_symbols = [
            k for k, _ in sorted(relative_strengths.items(), key=lambda item: -item[1])
        ][:5]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest