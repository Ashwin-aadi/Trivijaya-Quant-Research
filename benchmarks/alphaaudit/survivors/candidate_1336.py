from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which deviate significantly from their "
        "historical average will tend to return. By using a trailing window to establish this "
        "mean, we can identify undervalued stocks for entry."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.select(pl.col("adj_close").mean().alias("mean"))
            .select("mean")
            .to_series()
            .to_list()[0]
        )
        symbol_scores: dict[str, float] = {
            row["symbol"]: (row["close"] - mean_close) / mean_close
            for _, row in history.to_pandas().iterrows()
        }

        # Identify symbols with the smallest deviation from their trailing mean
        sorted_symbols = sorted(symbol_scores.items(), key=lambda x: abs(x[1]))
        picks = [symbol for symbol, _ in sorted_symbols[:5]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest