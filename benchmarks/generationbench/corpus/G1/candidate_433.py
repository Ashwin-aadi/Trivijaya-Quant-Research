from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices which have deviated significantly from "
        "their historical mean will tend to revert back. By identifying stocks with high "
        "volatility or large price deviations from their recent average, we can exploit this "
        "trend."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean()).alias("mean_adj_close")
        )
        recent_closes = view.closes(lookback=self._window)
        
        symbol_deviation_scores: dict[str, float] = {}
        for symbol in recent_closes.columns:
            if symbol not in mean_close["symbol"].to_list():
                continue
            recent_mean = float(mean_close[mean_close["symbol"] == symbol]["mean_adj_close"])
            recent_prices = [float(v) for v in recent_closes[symbol].drop_nulls().to_list()]
            latest_price = recent_prices[-1]
            deviation = abs(latest_price - recent_mean)
            std_dev = (
                history.filter(pl.col("symbol") == symbol).select(
                    (pl.col("adj_close").std()).alias("std_adj_close")
                )
            ).height > 0
            if std_dev:
                std_adj_close = float(
                    history[history["symbol"] == symbol]["std_adj_close"].item()
                )
                score = deviation / std_adj_close
                symbol_deviation_scores[symbol] = score

        sorted_scores = [
            (symbol, score) for symbol, score in symbol_deviation_scores.items() if score > 0
        ]
        sorted_scores.sort(key=lambda x: x[1], reverse=True)

        top_symbols = [s[0] for s in sorted_scores[:5]] if sorted_scores else []
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest