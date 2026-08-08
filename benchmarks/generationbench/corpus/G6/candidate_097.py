from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their 30-day "
        "moving average and are likely to revert. It uses a threshold of more than 2 standard "
        "deviations from the moving average for entry, ensuring only highly overvalued or undervalued "
        "stocks are considered."
    )

    def __init__(self, window: int = 30, deviation_threshold: float = 2.0, holding_days: int = 5) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold
        self._holding_days = holding_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        sma = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("sma"))
            .join(closes, on="symbol", how="inner")
        )

        std_dev = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close") - pl.col("sma")).std().alias("std_dev"),
                sma / 2.0,
            )
        ).join(sma, on="symbol", how="inner")

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in std_dev.columns or "std_dev" not in std_dev.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            sma_value = float(std_dev.filter(pl.col("symbol") == symbol)["sma"].item())
            std_dev_value = float(std_dev.filter(pl.col("symbol") == symbol)["std_dev"].item())

            if len(recent_closes) < self._window:
                continue
            latest_close = recent_closes[-1]
            if abs(latest_close - sma_value) / std_dev_value > self._deviation_threshold:
                signals.append(symbol)

        signals = signals[:30]
        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest