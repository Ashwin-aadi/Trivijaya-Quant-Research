from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two simple characteristics: the 20-day closing price momentum "
        "and the 10-day volume trend to identify potentially strong stocks."
    )

    def __init__(self, window_price: int = 20, window_volume: int = 10) -> None:
        self._window_price = window_price
        self._window_volume = window_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_price, self._window_volume))
        if history.height < max(self._window_price, self._window_volume):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window_price)
        volumes = view.closes(lookback=self._window_volume).with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(self._window_volume) - 1.0).alias("volume_ratio")
        )

        if closes.height < self._window_price or volumes.height < self._window_volume:
            return Signal(information_available_at=stamp, weights={})

        price_moments = {}
        volume_trends = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            price_moments[symbol] = float(close_values[-1] / close_values[0] - 1)

            volume_values = [float(v) for v in volumes[symbol].select("volume_ratio").to_series().drop_nulls().to_list()]
            if len(volume_values) < self._window_volume:
                continue
            volume_trends[symbol] = float(sum(volume_values[-self._window_volume:]) / self._window_volume)

        combined_scores = {symbol: price_moments[symbol] + volume_trends[symbol] for symbol in view.symbols}
        sorted_symbols = [k for k, v in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols[:5]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest