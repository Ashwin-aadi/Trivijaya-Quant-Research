from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeFeatureStrategy(Strategy):
    rationale = (
        "Combining momentum and volatility features can provide a more robust signal. "
        "Momentum helps identify trending stocks, while low volatility suggests stable performance."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in history["symbol"]:
                continue

            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            vol_values = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("adj_close").change().abs()
                    .alias("volatility_change")
                )
                .sort("session_date", descending=False)["volatility_change"]
                .to_list()
            )

            if len(close_values) < self._window or len(vol_values) < self._window:
                continue

            momentum = close_values[-1] - close_values[0]
            volatility = sum(vol_values) / self._window
            score = (momentum / 10.0 + 1.0 / volatility)

            if score >= max(score for symbol in view.symbols):
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest