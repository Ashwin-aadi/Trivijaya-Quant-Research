from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Securities with higher relative strength compared to the broad market index "
        "(NIFTY 100) are more likely to outperform in the near future. This is based on the "
        "idea that high relative strength indicates strong underlying fundamentals or positive"
        " sentiment, which can drive prices up."
    )

    def __init__(self, lookback_days: int = 60, top_n: int = 10) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty100_closes = history.filter(pl.col("symbol") == "NIFTY 100")
        if nifty100_closes.height < self._lookback_days:
            return Signal(information_available_at=stamp, weights={})

        other_symbols = view.symbols.difference(["NIFTY 100"])
        relative_strength: dict[str, float] = {}
        for symbol in other_symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            symbol_closes = history.filter(pl.col("symbol") == symbol).select(
                "adj_close"
            )
            nifty100_closes = nifty100_closes.select("adj_close")
            ratio = (symbol_closes["adj_close"] / nifty100_closes["adj_close"]).to_list()
            if len(ratio) < self._lookback_days:
                continue
            last_ratio = float(ratio[-1])
            relative_strength[symbol] = last_ratio

        sorted_ranks = sorted(relative_strength.items(), key=lambda x: x[1], reverse=True)
        picks: list[str] = [symbol for symbol, _ in sorted_ranks[: self._top_n]]
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