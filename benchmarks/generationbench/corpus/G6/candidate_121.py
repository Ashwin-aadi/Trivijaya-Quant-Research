from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Utilizing a short-horizon mean reversion strategy involves identifying stocks with "
        "significant deviations from their 20-day moving average and entering long positions. "
        "This approach aims to capitalize on the tendency of stock prices to revert to historical "
        "averages, balancing statistical significance with practical liquidity criteria."
    )

    def __init__(self, window: int = 20, threshold_zscore: float = -1.5, max_positions: int = 30) -> None:
        self._window = window
        self._threshold_zscore = threshold_zscore
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_data = [symbol for symbol in view.symbols if symbol in history.columns]
        filtered_history = history[symbols_with_data]

        ma20 = filtered_history.select(
            pl.col("adj_close").rolling_mean(self._window).alias(f"ma_{self._window}")
        )
        std_dev = filtered_history.select(
            pl.col("adj_close").rolling_std(self._window).alias(f"std_{self._window}")
        )
        z_scores = (filtered_history["adj_close"] - ma20[f"ma_{self._window}"]) / std_dev[f"std_{self._window}"]

        picks: list[str] = []
        for symbol in symbols_with_data:
            if z_scores[symbol].is_nan().sum() > 0 or z_scores[symbol][-1] < self._threshold_zscore:
                continue
            daily_volume = view.closes(lookback=30)[symbol].to_list()[-1]
            if daily_volume > 5_000_000:  # Assuming 1 lakh as the base for volume
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