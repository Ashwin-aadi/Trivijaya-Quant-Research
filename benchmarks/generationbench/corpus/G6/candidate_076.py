from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedReversion(Strategy):
    rationale = (
        "This strategy combines mean-reverting behavior with short-term trend analysis to identify "
        "overvalued or undervalued stocks. It buys when prices are below their SMAs and sells when they exceed them."
    )

    def __init__(self, sma_window_50: int = 50, sma_window_20: int = 20, threshold: float = 0.02) -> None:
        self._sma_window_50 = sma_window_50
        self._sma_window_20 = sma_window_20
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._sma_window_50, self._sma_window_20))
        if closes.height < max(self._sma_window_50, self._sma_window_20):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            sma_50 = _rolling_mean(closes[symbol], window=self._sma_window_50)
            sma_20 = _rolling_mean(closes[symbol], window=self._sma_window_20)

            latest_close = float(view.latest_close()[symbol])
            if (latest_close - sma_50[-1]) / sma_50[-1] < -self._threshold and \
                    (latest_close - sma_20[-1]) / sma_20[-1] < -self._threshold:
                picks.append(symbol)

        picks = picks[:30]
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


def _rolling_mean(series: pl.Series, window: int) -> pl.Series:
    return series.rolling_mean(window=window).fill_null(strategy="backfill")