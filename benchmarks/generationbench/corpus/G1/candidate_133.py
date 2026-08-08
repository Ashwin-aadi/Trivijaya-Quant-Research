from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion strategies aim to profit from the tendency of prices to return to their mean. "
        "Short-term deviations from this mean can provide trading opportunities."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .to_pandas()["mean"]
        )
        
        recent_closes = view.closes(lookback=self._window)
        z_scores = []
        for symbol in symbols:
            if symbol not in recent_closes.columns:
                continue
            symbol_history = history.filter(pl.col("symbol") == symbol).sort(
                "session_date"
            ).to_pandas()
            symbol_mean = mean_close[symbol]
            z_score = (recent_closes[symbol][-1] - symbol_mean) / symbol_history[
                "adj_close"
            ].std(skipna=True)
            z_scores.append((symbol, z_score))

        z_scores.sort(key=lambda x: abs(x[1]), reverse=True)

        top_symbols = [s for s, _ in z_scores[:5]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest