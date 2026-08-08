from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of financial assets to revert to their "
        "historical mean price over time. When an asset's price is significantly above its historical "
        "mean, it is expected to fall back; vice versa."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_price = history.groupby("symbol").agg(
            (pl.col("adj_close").mean().alias("mean_price"))
        )
        recent_close = view.closes()
        price_diffs: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in recent_close.columns or symbol not in mean_price["symbol"]:
                continue
            recent_adj_close = float(recent_close[symbol].drop_nulls().to_list()[-1])
            mean_price_val = float(mean_price.filter(pl.col("symbol") == symbol)[
                                        "mean_price"].to_list()[0])

            if abs(recent_adj_close - mean_price_val) / mean_price_val > self._threshold:
                price_diffs[symbol] = recent_adj_close - mean_price_val

        signals: list[str] = [s for s, v in price_diffs.items() if v < 0]
        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest