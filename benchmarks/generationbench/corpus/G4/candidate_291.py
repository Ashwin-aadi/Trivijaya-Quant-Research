from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the inefficiencies in smaller-cap Indian stocks by "
        "screening for sufficient liquidity and then equally weighting these stocks to "
        "capture mean reversion effects. Smaller-cap equities often exhibit higher "
        "idiosyncratic volatility, making them less followed and potentially less efficiently priced."
    )

    def __init__(self, lookback_days: int = 30, min_volume: float = 1_000_000, top_n: int = 200) -> None:
        self._lookback_days = lookback_days
        self._min_volume = min_volume
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback_days)
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        volume_history = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("volume") / 24).alias("average_volume")
        )

        ranked_symbols = _rank_symbols(symbols, closes, volume_history)
        if len(ranked_symbols) < self._top_n:
            top_symbols = ranked_symbols
        else:
            top_symbols = ranked_symbols[:self._top_n]

        weight_per_stock = 1.0 / len(top_symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_stock for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _rank_symbols(symbols: list[str], closes: pl.DataFrame, volume_history: pl.DataFrame) -> list[str]:
    ranked = []
    for symbol in symbols:
        latest_close = float(view.latest_close()[symbol])
        history = closes.select(pl.col(symbol).alias("price")).with_columns(
            (pl.col("price") / pl.col("average_volume").shift(1) - 1.0).alias("ranked_return")
        ).sort("session_date", descending=True)
        if history.height < 60:
            continue
        last_60_days = history.head(60)

        avg_ranked_return = float(last_60_days.select(pl.col("ranked_return").mean()).item())
        ranked.append((symbol, latest_close * avg_ranked_return))

    return [s[0] for s in sorted(ranked, key=lambda x: -x[1])][:200]