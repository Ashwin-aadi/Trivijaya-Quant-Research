from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks based on their relative strength against the broader market "
        "can provide a momentum-based edge. This strategy favors stocks that are outperforming "
        "the NIFTY 100 index."
    )

    def __init__(self, lookback: int = 20) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)

        if closes.height < self._lookback or "NIFTY 100" not in closes.columns:
            return Signal(information_available_at=stamp, weights={})

        nifty_close = closes["NIFTY 100"].to_list()[-self._lookback:]
        stock_closes = [float(v) for v in closes.drop_columns("NIFTY 100").select(pl.col("*").to_list())]

        if len(stock_closes) < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        performance_ratios: list[float] = []
        for stock_close in stock_closes:
            ratio = max((close / nifty_close[i] - 1.0 for i, close in enumerate(stock_close)), default=-1)
            if ratio > 0:
                performance_ratios.append(ratio)

        top_stocks = sorted(view.symbols, key=lambda sym: sum([ratio for i, ratio in enumerate(performance_ratios) if stock_closes[i][sym] / nifty_close[i] - 1.0 == ratio]), reverse=True)[:5]

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest