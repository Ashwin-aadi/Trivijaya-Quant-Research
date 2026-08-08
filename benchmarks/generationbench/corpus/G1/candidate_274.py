from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Equities often exhibit seasonality based on calendar effects. For instance, "
        "certain sectors might perform better in specific months due to economic or "
        "market conditions. By identifying and trading these seasonal trends, we can "
        "potentially generate excess returns."
    )

    def __init__(self, window: int = 10, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            monthly_closes = _monthly_aggregate(view, symbol)
            avg_close = pl.DataFrame(monthly_closes).select(pl.col("adj_close").mean()).item()
            strength = (values[-1] - avg_close) / avg_close
            seasonal_strengths[symbol] = strength

        sorted_symbols = sorted(seasonal_strengths.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_symbols[: self._top_n]]

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


def _monthly_aggregate(view: MarketView, symbol: str) -> list[dict[str, float]]:
    monthly_closes = []
    history = view.history(lookback=None).filter(pl.col("symbol") == symbol)
    for i in range(0, len(history), 21):  # Approximating a month with 21 trading days
        month_close = (
            history[i : i + 21]
            .select(["session_date", "adj_close"])
            .sort(by="session_date")
            .tail(n=1)
            .row(0)[1]
        )
        monthly_closes.append({"date": history["session_date"][i], "adj_close": month_close})
    return monthly_closes