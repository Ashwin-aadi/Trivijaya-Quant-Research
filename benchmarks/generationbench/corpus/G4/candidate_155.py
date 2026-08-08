from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityGovernance(Strategy):
    rationale = (
        "This strategy exploits a composite of two weakly related characteristics: "
        "seasonal price patterns and corporate governance metrics. By identifying stocks with "
        "historical price patterns during specific seasons and high ESG scores, we aim to capture "
        "potential arbitrage opportunities."
    )

    def __init__(self, window: int = 4, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window * 4:  # Assuming each quarter has roughly the same number of data points
                continue

            avg_returns_q1 = sum(values[0:self._window]) / self._window
            avg_returns_q2 = sum(values[self._window : (self._window * 2)]) / self._window
            avg_returns_q3 = sum(values[(self._window * 2) : (self._window * 3)]) / self._window
            avg_returns_q4 = sum(values[(self._window * 3):]) / self._window

            seasonality_scores[symbol] = max(avg_returns_q1, avg_returns_q2, avg_returns_q3, avg_returns_q4)

        governance_scores = view.closes("ESG_SCORE").to_dict()["ESG_SCORE"]
        if len(governance_scores) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        combined_scores = {symbol: seasonality_scores.get(symbol, 0.0) * governance_scores[symbol] for symbol in view.symbols}
        top_symbols = sorted(combined_scores.keys(), key=lambda s: combined_scores[s], reverse=True)[:self._top_n]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest