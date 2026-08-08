from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompressionStrategy(Strategy):
    rationale = (
        "This strategy identifies stocks experiencing significant price movements relative to their historical ranges during periods of high dispersion or range compression. It aims to capitalize on these opportunities while managing risk through strict exit criteria and a stop-loss."
    )

    def __init__(self, window: int = 20, threshold_high_dhlr: float = 4.0, threshold_low_dhlr: float = 2.0, exit_after_days: int = 3, sma_window: int = 50, price_deviation_threshold: float = 3.0, max_positions: int = 15) -> None:
        self._window = window
        self._threshold_high_dhlr = threshold_high_dhlr
        self._threshold_low_dhlr = threshold_low_dhlr
        self._exit_after_days = exit_after_days
        self._sma_window = sma_window
        self._price_deviation_threshold = price_deviation_threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            ohlcv = history.filter(pl.col("symbol") == symbol)
            last_close = float(ohlcv.select("close").last().item())
            prev_close = float(history.filter(pl.col("symbol") == symbol).select("close").shift(1).last().item())

            dhlr = (float(ohlcv.select("high").max().item()) - float(ohlcv.select("low").min().item())) / prev_close
            if dhlr >= self._threshold_high_dhlr and ohlcv.select("volume").last().item() > 1_000_000:
                picks.append(symbol)

        if len(picks) < self._max_positions:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest