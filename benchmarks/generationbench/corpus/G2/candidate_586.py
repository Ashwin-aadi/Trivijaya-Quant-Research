from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stock market patterns exhibit seasonality due to predictable changes in "
        "economic conditions or investor behavior. For instance, the Indian stock market may show "
        "a pattern of higher returns during festive seasons such as Diwali or Durga Puja. This strategy "
        "aims to capitalize on these seasonal effects by allocating capital towards stocks with historically "
        "strong performance in specific months."
    )

    def __init__(self, holiday_months: tuple[int, ...] = (10, 9)) -> None:
        self._holiday_months = holiday_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=365)
        if closes.height < 365:
            return Signal(information_available_at=stamp, weights={})

        seasonality_scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            month_scores = []
            for i in range(0, len(values), 30):
                month_values = values[i:i+30]
                if 10 <= (i // 30 + 1) <= 12 or 9 <= (i // 30 + 1) <= 10:
                    mean_value = sum(month_values) / len(month_values)
                    month_scores.append((mean_value, i))
            seasonality_scores[symbol] = max(month_scores)[0] if month_scores else None

        top_symbols = sorted(seasonality_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in (sym for sym, _ in top_symbols)}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest