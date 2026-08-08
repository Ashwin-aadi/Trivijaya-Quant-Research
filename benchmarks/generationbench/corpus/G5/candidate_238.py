from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit stronger performance during specific months of the year. "
        "This strategy aims to capitalize on these seasonal patterns by allocating capital to symbols that have historically performed well in certain months."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_closes = view.closes(lookback=self._window).transpose()
        symbols = symbol_closes.columns

        seasonal_effect_scores: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in history.columns or symbol not in symbol_closes.columns:
                continue
            close_series = [float(v) for v in symbol_closes[symbol].drop_nulls().to_list()]
            month_effect_score = 0.0
            for i in range(self._window):
                month = (close_series[i] / close_series[max(0, i - self._window)] - 1.0)
                if i % 12 < 6:  # 前半年表现更好
                    month_effect_score += max(month, 0.0)
                else:
                    month_effect_score -= min(month, 0.0)

            seasonal_effect_scores[symbol] = month_effect_score

        sorted_symbols = [
            symbol for _, symbol in sorted(seasonal_effect_scores.items(), key=lambda item: abs(item[1]), reverse=True)
        ]
        top_n_symbols = sorted_symbols[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest