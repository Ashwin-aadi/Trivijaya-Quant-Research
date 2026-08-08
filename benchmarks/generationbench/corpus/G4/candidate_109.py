from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks with "
        "significant price deviations from their historical means. It aims to capitalize on the "
        "tendency of stock prices to revert to their long-term averages following temporary noise."
    )

    def __init__(self, lookback_period: int = 30, threshold: float = 2.0) -> None:
        self._lookback_period = lookback_period
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history.select("close").to_series().drop_nulls().to_list()]
        mean_price = pl.Series(closes).mean()
        std_dev = pl.Series(closes).std()

        z_scores: dict[str, float] = {}
        for symbol in view.symbols:
            latest_close = view.latest_close()[symbol]
            if latest_close is not None:
                z_score = (latest_close - mean_price) / std_dev
                if abs(z_score) > self._threshold:
                    z_scores[symbol] = z_score

        if not z_scores:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest