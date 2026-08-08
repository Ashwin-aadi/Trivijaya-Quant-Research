from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two simple characteristics: the 20-day closing price momentum "
        "and the 10-day volatility. Entries are triggered when both conditions are favorable."
    )

    def __init__(self, window_c: int = 20, window_v: int = 10) -> None:
        self._window_c = window_c
        self._window_v = window_v

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_c)
        if closes.height < self._window_c:
            return Signal(information_available_at=stamp, weights={})

        vol_history = view.history(lookback=self._window_v).select(
            pl.col("symbol"), (pl.col("adj_close").std().alias("volatility"))
        )
        picks: list[str] = []

        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in vol_history.select("symbol"):
                continue
            values_c = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            values_v = [float(v) for v in vol_history.filter(pl.col("symbol") == symbol)["volatility"].to_list()]

            if len(values_c) < self._window_c or len(values_v) < self._window_v:
                continue

            close_slope = (values_c[-1] - values_c[0]) / max(1, self._window_c)
            volatility = float(max(values_v))
            if close_slope > 0 and volatility <= max(values_v):
                picks.append(symbol)

        picks = picks[:5]
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