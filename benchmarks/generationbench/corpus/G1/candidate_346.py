from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality effects can significantly impact stock prices. This strategy "
        "exploits historical patterns by identifying stocks that perform well during specific times of the year."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or (history["session_date"].max() - history["session_date"].min()).days < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(closes) < self._window:
                continue

            monthly_returns = []
            for i in range(0, 12):
                start_month_index = next(j for j, date_str in enumerate([d.strftime("%Y-%m-%d") for d in history["session_date"]]) if int(date_str.split("-")[1]) == (i + 1))
                end_month_index = min(start_month_index + self._window // 12, len(closes))

                monthly_returns.append((closes[end_month_index - 1] / closes[start_month_index - 1] - 1.0) if start_month_index > 0 else 0)

            max_return = max(monthly_returns)
            seasonal_trends[symbol] = max_return

        top_symbols = sorted(seasonal_trends, key=seasonal_trends.get, reverse=True)[:5]
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