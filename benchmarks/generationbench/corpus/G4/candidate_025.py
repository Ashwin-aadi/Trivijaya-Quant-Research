from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion50d(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by taking positions in stocks "
        "that deviate significantly from their 50-day simple moving average (SMA). Stocks above "
        "their SMA are sold short, while those below it are bought long."
    )

    def __init__(self, window: int = 50, threshold: float = 0.03, top_n: int = 20) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_sma = (closes["session_date"], closes.groupby("symbol").mean().select(
            pl.col("adj_close").alias("sma")).to_pandas()["sma"].to_dict())

        def get_deviation(symbol: str, close_price: float) -> float:
            return abs(close_price - mean_sma[symbol]) / mean_sma[symbol]

        deviations = [(symbol, get_deviation(symbol, closes[symbol].max())) for symbol in view.symbols
                      if symbol in mean_sma and symbol in closes.columns]
        deviations.sort(key=lambda x: x[1], reverse=True)

        picks: list[str] = []
        for symbol, deviation in deviations:
            if len(picks) >= self._top_n:
                break
            if deviation > self._threshold or deviation < -self._threshold:
                picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: -weight for s in picks[:int(self._top_n/2)]} | {s: weight for s in picks[int(self._top_n/2):]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest