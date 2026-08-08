from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy aims to screen for highly liquid stocks before applying equal weighting. "
        "High liquidity ensures that trading the stock will not significantly impact its price, "
        "making it a more stable investment choice."
    )

    def __init__(self, min_volume: int = 100000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)  # Consider the last year for liquidity and price data
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        filtered_history = history.filter(
            (pl.col("volume") > self._min_volume).all()
        ).sort("session_date", descending=False)

        if filtered_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weights: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in filtered_history.columns:
                continue
            volume_series = filtered_history[symbol]["volume"]
            volume_ranked = volume_series.rank(method="average", descending=True).to_list()
            weight = 1.0 / len(symbols)
            weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights={s: weights.get(s, 0.0) for s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest