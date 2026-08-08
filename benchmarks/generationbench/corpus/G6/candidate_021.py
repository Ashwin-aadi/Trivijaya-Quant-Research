from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "The strategy focuses on identifying and investing in stocks with lower historical "
        "volatility to reduce market risk. It ranks all listed Indian stocks by their 20-day "
        "rolling standard deviation of daily log returns and selects the bottom quartile for "
        "portfolio inclusion, rebalancing monthly."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = view.closes()
        adjusted_closes = [float(v) for v in closes.to_vertical().select("adj_close").to_dict()[0].values()]

        log_returns = [(pl.col(f"{symbol}_adj_close") / pl.col(f"{symbol}_adj_close").shift(1) - 1.0).alias(f"r_{symbol}") for symbol in symbols]
        volatilities = (history.select(log_returns)
                        .group_by("session_date")
                        .agg([pl.col(f"r_{symbol}").std().alias(f"vol_{symbol}") for symbol in symbols])
                        .sort("session_date", descending=False)
                        .filter(pl.col("session_date") == stamp - pl.duration(days=self._window))
                        .select([f"vol_{symbol}" for symbol in symbols])
                        .to_numpy()[0].tolist())

        sorted_indices = [i[0] for i in sorted(enumerate(volatilities), key=lambda x: x[1])]
        bottom_quartile_symbols = [symbols[i] for i in sorted_indices[:self._top_n]]

        if not bottom_quartile_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(bottom_quartile_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in bottom_quartile_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest