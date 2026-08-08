from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakout(Strategy):
    rationale = (
        "This strategy aims to capture significant price changes due to strong buying or "
        "selling pressures. It triggers entries when a stock's closing price exceeds its 5-day "
        "moving average by at least 2%, accompanied by a 30% increase in volume compared to the "
        "previous day, indicating strong buying pressure. The exit rules ensure timely exits "
        "to manage risk effectively."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        volumes = _volumes(view, self._window)

        picks: list[str] = []
        for symbol in view.symbols:
            if (
                symbol not in history.columns
                or symbol not in closes.columns
                or symbol not in volumes.columns
            ):
                continue

            close_series = pl.col(symbol)
            volume_series = pl.col(symbol + "_volume")

            close_values = [float(v) for v in close_series.drop_nulls().to_list()]
            volume_values = [float(v) for v in volume_series.drop_nulls().to_list()]

            if len(close_values) < self._window:
                continue

            moving_avg_close = sum(close_values[-self._window:]) / self._window
            latest_close = close_values[-1]

            if (latest_close - moving_avg_close) / moving_avg_close >= 0.02 and volume_values[0] * 1.3 <= volume_values[-1]:
                picks.append(symbol)

        picks = picks[:10]
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


def _volumes(view: MarketView, window: int) -> pl.DataFrame:
    history = view.history(lookback=window + 1)
    if history.is_empty():
        return pl.DataFrame()

    symbols = [symbol for symbol in view.symbols]
    volume_history = history.select(symbols).with_columns(
        (pl.col(symbol) * pl.col("volume")).alias(symbol + "_volume")
        for symbol in symbols
    )

    return volume_history.sort("session_date").tail(window)