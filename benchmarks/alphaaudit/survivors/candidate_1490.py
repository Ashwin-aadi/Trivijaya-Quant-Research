from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following involves identifying stocks that are trending "
        "in a particular direction while considering the volatility of those trends. "
        "During periods of high volatility, small price movements can be amplified, "
        "potentially leading to higher returns if the trend continues."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + self._vol_window - 1)
        if closes.height < self._window + self._vol_window - 1:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window + self._vol_window - 1:
                continue

            # Calculate returns
            returns = [(prices[i] / prices[i-1] - 1.0) for i in range(1, self._window + self._vol_window - 1)]

            # Calculate volatility
            vol = ((pl.Series(returns).rank(method="dense", descending=False) /
                    (self._vol_window - 1)).to_list())

            # Calculate trend score
            if sum(vol[-self._window:]) > 0.5 * self._window:
                trends[symbol] = 1.0
            elif sum(vol[-self._window:]) < 0.5 * self._window:
                trends[symbol] = -1.0

        positive_trends = [s for s, t in trends.items() if t == 1.0]
        negative_trends = [s for s, t in trends.items() if t == -1.0]

        weight_positive = len(positive_trends) / (len(positive_trends) + len(negative_trends))
        weight_negative = len(negative_trends) / (len(positive_trends) + len(negative_trends))

        return Signal(
            information_available_at=stamp,
            weights={s: weight_positive for s in positive_trends} |
            {s: -weight_negative for s in negative_trends}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest