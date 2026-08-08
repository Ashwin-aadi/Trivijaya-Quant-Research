from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceReversion(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency of asset prices to return to levels "
        "near their recent average. This can be particularly effective in volatile markets."
    )

    def __init__(self, window: int = 20, k: float = 1.5) -> None:
        self._window = window
        self._k = k

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = float(closes.mean().select(pl.col("adj_close")).item())
        std_dev_close = float(closes.select((pl.col("adj_close") - pl.col("adj_close").mean()).stddev()).item())

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            z_score = (values[-1] - mean_close) / std_dev_close
            if abs(z_score) > self._k and values[-1] != mean_close:
                picks.append(symbol)

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