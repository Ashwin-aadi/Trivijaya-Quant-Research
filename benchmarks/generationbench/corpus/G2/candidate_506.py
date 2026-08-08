from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the phenomenon where high volatility "
        "periods are followed by prolonged trends in a given direction. By scaling the position"
        " size based on recent volatility, we can potentially benefit from these trends while "
        "reducing risk."
    )

    def __init__(self, window: int = 20, trend_factor: float = 1.5) -> None:
        self._window = window
        self._trend_factor = trend_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        returns = (
            closes
            .sort("session_date", descending=True)
            .select([
                pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0,
            ])
            .drop_nulls()
            .head(self._window)
            ["arr_0"]
            .to_list()
        )

        volatility = sum(abs(r) for r in returns) / self._window
        mean_return = sum(returns) / self._window

        if not returns or abs(mean_return) < 0.01:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            history = view.history(symbol=symbol).sort("session_date", descending=True)
            recent_closes = history.select(pl.col("adj_close")).to_numpy()[-self._window:]

            # Calculate the trend direction
            recent_returns = [
                (recent_closes[i] - recent_closes[i + 1]) / recent_closes[i + 1]
                for i in range(self._window - 2, -1, -1)
            ]
            if not recent_returns or max(recent_returns) < 0:
                continue

            weight = self._trend_factor * volatility
            picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = self._trend_factor * volatility / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest