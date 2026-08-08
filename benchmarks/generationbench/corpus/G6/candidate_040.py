from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion50d(Strategy):
    rationale = (
        "The strategy is based on mean reversion, leveraging historical price levels to identify "
        "opportunities in the Indian stock market. It uses a 50-day simple moving average (SMA) as "
        "the trailing reference level for both measurement and entry/exit conditions."
    )

    def __init__(self, window: int = 50, deviation_threshold: float = 0.03, max_positions: int = 30) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma = history.group_by("symbol").agg(
            (pl.col("close").mean()).alias("sma")
        ).with_columns(
            ((pl.col("adj_close") - pl.col("sma")) / pl.col("sma")).abs().alias("deviation"),
            ((pl.col("adj_close") - pl.col("sma")).rolling_std(window=self._window)).over("symbol").alias("std_dev")
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in sma.columns or len(sma[symbol].to_list()) < self._window:
                continue
            values = [float(v) for v in sma[symbol].drop_nulls().to_list()]
            last_price = view.latest_close()[symbol]
            deviation = float(values[-1])
            std_dev = float(values[-1 + len(sma["std_dev"])])

            if abs(deviation) > self._deviation_threshold:
                picks.append(symbol)

        picks = picks[: self._max_positions]
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