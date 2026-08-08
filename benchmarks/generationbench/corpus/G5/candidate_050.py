from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends by scaling them with the recent volatility. "
        "Long positions are taken in symbols that have been trending positively and "
        "have low historical volatility, while short positions are taken in those trending negatively."
    )

    def __init__(self, window: int = 20, n_trends: int = 5) -> None:
        self._window = window
        self._n_trends = n_trends

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique():
                continue
            adj_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(adj_closes) < self._window + 1:
                continue

            trend_score = (adj_closes[-1] - adj_closes[0]) / (len(adj_closes) - 1)
            volatility = adj_closes.std()  # Use standard deviation as a measure of volatility
            trends[symbol] = trend_score / max(volatility, 1e-6)

        sorted_trends = sorted(trends.items(), key=lambda x: x[1], reverse=True)
        longs = [k for k, v in sorted_trends[:self._n_trends]]
        shorts = [k for k, v in sorted_trends[-self._n_trends:]]

        weights = {s: 0.5 / len(longs) for s in longs}
        if shorts:
            weights.update({s: -0.5 / len(shorts) for s in shorts})

        return Signal(information_available_at=stamp, weights={k: v for k, v in weights.items() if v != 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest