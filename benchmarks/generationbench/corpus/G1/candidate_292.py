from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying symbols that have deviated "
        "significantly from their trailing average, we can identify potential mean-reverting "
        "opportunities."
    )

    def __init__(self, window: int = 60, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()
        symbols_with_data = set(history["symbol"]) & set(latest_closes.keys())
        
        signal_weights: dict[str, float] = {}
        for symbol in symbols_with_data:
            if symbol not in history.columns:
                continue
            close_series = pl.col("adj_close").filter(pl.col("symbol") == symbol)
            mean_close = close_series.mean()
            std_dev_close = close_series.std()

            z_score = (latest_closes[symbol] - mean_close) / std_dev_close if std_dev_close > 0 else 0
            if abs(z_score) >= self._z_score_threshold:
                signal_weights[symbol] = 1.0 / len(signal_weights)

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signal_weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest