from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy identifies trends in stocks while scaling positions based on market volatility "
        "to mitigate risk. It combines elements from both a simple trend-following approach with stop-losses and a diversified portfolio to balance simplicity and robustness."
    )

    def __init__(self, sma_window: int = 20, atr_window: int = 14, max_held: int = 15) -> None:
        self._sma_window = sma_window
        self._atr_window = atr_window
        self._max_held = max_held

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._atr_window + self._sma_window - 1)
        if closes.height < self._atr_window + self._sma_window - 1:
            return Signal(information_available_at=stamp, weights={})

        sma: pl.DataFrame = view.history().select(
            "session_date",
            (pl.col("adj_close").over(pl.range(0, self._sma_window).sum()) / self._sma_window)
                .alias("sma"),
        )
        atr: pl.DataFrame = _atr(view.history(), lookback=self._atr_window)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in sma.columns or symbol not in atr.columns:
                continue
            sma_val = float(sma.filter(pl.col("symbol") == symbol)["sma"].item())
            close_val = float(atr.filter(pl.col("symbol") == symbol)["atr"].item())

            if closes[symbol].to_list()[-1] > sma_val and min(closes[symbol].to_list()) < sma_val:
                picks.append(symbol)

        picks = picks[: self._max_held]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight
                for s in picks
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _atr(history: pl.DataFrame, lookback: int) -> pl.DataFrame:
    high = history.select("high").to_numpy().ravel()
    low = history.select("low").to_numpy().ravel()
    close_prev = history.select("close").shift(1).to_numpy().ravel()
    tr = [max(max(h[i], l[i]) - min(l[i], h[i + 1]), abs(h[i] - c_p[i])) for i in range(len(high) - 1)]
    atr = sum(tr[-lookback:]) / lookback
    return pl.DataFrame({"symbol": history["symbol"].to_list()[-lookback:], "atr": [atr]})