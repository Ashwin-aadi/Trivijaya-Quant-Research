from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "This strategy exploits calendar effects by identifying months with historically "
        "higher returns for specific sectors. It buys stocks in these sectors a few weeks "
        "before the favorable period and sells them shortly after to capitalize on expected "
        "price increases."
    )

    def __init__(self, lookback_years: int = 5, window_days: int = 20, top_n: int = 10) -> None:
        self._lookback_years = lookback_years
        self._window_days = window_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)
        if history.height < self._lookback_years * 252:
            return Signal(information_available_at=stamp, weights={})

        sectors = ["consumer_discretionary"]  # Example sector
        signals = {sector: [] for sector in sectors}
        for symbol in view.symbols:
            if any(symbol.startswith(sect) for sect in sectors):
                df = history.filter(pl.col("symbol") == symbol)
                prices = [float(v) for v in df["adj_close"].drop_nulls().to_list()]
                if len(prices) < self._lookback_years * 252:
                    continue
                # Calculate monthly returns and rank based on past performance
                monthly_returns = [
                    (prices[i + 1] - prices[i]) / prices[i]
                    for i in range(len(prices) - 1)
                ]
                monthly_dates = [date.fromordinal(int(d)) for d in df["session_date"].to_list()]
                monthly_performance = {m: r for m, r in zip(monthly_dates[::252], monthly_returns)}
                top_months = sorted(
                    monthly_performance.items(), key=lambda x: x[1], reverse=True
                )[: self._top_n]
                signals[symbol] = [m for m, _ in top_months]

        picks = []
        for symbol, months in signals.items():
            if view.as_of.month in months:
                picks.append(symbol)

        picks = list(set(picks))[: self._top_n]
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