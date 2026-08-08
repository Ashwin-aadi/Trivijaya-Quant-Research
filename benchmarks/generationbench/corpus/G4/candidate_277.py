from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "This strategy aims to exploit short-term mean reversion in Indian equity markets by "
        "identifying stocks that have deviated significantly from their historical price levels. "
        "It uses Bollinger Bands to detect overbought or oversold conditions and capitalizes on "
        "the anticipated reversion of prices."
    )

    def __init__(self, window: int = 20, band_width: float = 2.0, max_positions: int = 20) -> None:
        self._window = window
        self._band_width = band_width
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            sma = sum(values[-self._window:]) / self._window
            std_dev = (sum((v - sma) ** 2 for v in values[-self._window:]) / self._window) ** 0.5
            upper_band = sma + self._band_width * std_dev
            lower_band = sma - self._band_width * std_dev

            if closes[symbol][closes[symbol].height - 1] > upper_band:
                picks.append(symbol)
            elif closes[symbol][closes[symbol].height - 1] < lower_band:
                picks.append(symbol)

        picks = sorted(picks, key=lambda s: abs(closes[s][closes[s].height - 1] - sma), reverse=True)[: self._max_positions]
        if not picks:
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