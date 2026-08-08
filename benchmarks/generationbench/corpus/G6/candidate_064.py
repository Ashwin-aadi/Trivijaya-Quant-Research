from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion200d(Strategy):
    rationale = (
        "This strategy leverages mean-reverting behavior in stock prices over time using a 200-day "
        "simple moving average (SMA) as the trailing reference. It enters positions when stocks deviate "
        "from their SMA by more than 3%, ensuring that only clear signs of overvaluation or undervaluation are considered."
    )

    def __init__(self, window: int = 200, threshold: float = 0.03, top_n: int = 50) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close") / pl.col("adj_close").rolling_mean(self._window).shift(1) - 1.0).alias("deviation")
        )
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in sma.select("symbol").to_numpy().flatten():
                continue
            deviation = float(sma.filter(pl.col("symbol") == symbol)["deviation"].item())
            if abs(deviation) > self._threshold and history.filter(pl.col("symbol") == symbol).select(
                    pl.col("volume").max()).item() >= 10_000:
                picks.append(symbol)

        picks = picks[: self._top_n]
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