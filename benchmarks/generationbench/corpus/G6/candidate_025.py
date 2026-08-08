from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion identifies stocks that have deviated significantly from their mean "
        "price. By entering positions in such stocks, we aim to capture the return towards "
        "the historical average price."
    )

    def __init__(self, window_short: int = 10, window_long: int = 60, z_score_threshold: float = 1.5, max_positions: int = 20) -> None:
        self._window_short = window_short
        self._window_long = window_long
        self._z_score_threshold = z_score_threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_long + self._window_short - 1)
        if closes.height < self._window_long + self._window_short - 1:
            return Signal(information_available_at=stamp, weights={})

        history = view.history(lookback=self._window_long + self._window_short - 1)
        symbols = [s for s in view.symbols if s in closes.columns and s in history.columns]

        signal = pl.DataFrame(
            {
                "symbol": [],
                "mean_return": [],
                "std_deviation": [],
                "z_score": [],
                "weight": []
            }
        )

        for symbol in symbols:
            history_row = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            closes_row = closes.filter(pl.col("symbol") == symbol)

            mean_return = float(history_row.select(
                (pl.col("adj_close").tail(self._window_short) - pl.col("adj_close").shift(1)).mean()
            ).item())
            std_deviation = float(closes_row.select(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).std()
            ).item())

            z_score = (float(closes_row.select(pl.col("close").last()).item()) - mean_return) / std_deviation

            if abs(z_score) >= self._z_score_threshold:
                signal = pl.concat([signal, pl.DataFrame({
                    "symbol": [symbol],
                    "mean_return": [mean_return],
                    "std_deviation": [std_deviation],
                    "z_score": [z_score],
                    "weight": [1.0 / self._max_positions]
                })])

        if signal.height == 0:
            return Signal(information_available_at=stamp, weights={})

        signal = signal.sort(by="z_score", descending=True).tail(self._max_positions)
        weights = {row["symbol"]: row["weight"] for _, row in signal.iter_rows()}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest