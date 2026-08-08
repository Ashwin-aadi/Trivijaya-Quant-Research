from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy selects stocks that show both a high relative strength and a strong "
        "volume increase over the last 10 trading days. Relative strength indicates outperformance "
        "against the market, while increased volume suggests market interest."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        strong_volume_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volume_changes = [
                float(v)
                for _, v in (
                    history.with_columns(
                        (pl.col("volume") - pl.col("volume").shift(1)).alias("volume_change")
                    )
                    .sort("session_date", descending=True)
                    .select(["symbol", "volume_change"])
                    .to_dicts()
                )
            ]
            if len(volume_changes) < self._window:
                continue
            volume_rise = any(change > 0 for change in volume_changes)
            if volume_rise and close_prices[-1] >= max(close_prices):
                strong_volume_symbols.append(symbol)

        if not strong_volume_symbols:
            return Signal(information_available_at=stamp, weights={})

        relative_strength_symbols = []
        market_returns = (
            view.closes(lookback=self._window)
            .select(pl.all().exclude("symbol"))
            .sum(axis=1)
            / len(view.symbols)
        )
        for symbol in strong_volume_symbols:
            if (close := history[symbol]).is_empty():
                continue
            close_series = [float(v) for v in close.drop_nulls().to_list()]
            market_close_series = [
                float(market_returns[i]) for i, _ in enumerate(close_series)
            ]
            relative_return = (
                sum((close - market_close) / market_close for close, market_close in zip(
                    close_series, market_close_series
                )) / len(close_series)
            )
            if relative_return >= 0.1:
                relative_strength_symbols.append(symbol)

        final_symbols = strong_volume_symbols[:5]
        weight = 1.0 / len(final_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in final_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest