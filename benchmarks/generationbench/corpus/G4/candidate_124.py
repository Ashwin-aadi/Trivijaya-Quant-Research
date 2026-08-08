from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion52w(Strategy):
    rationale = (
        "This strategy aims to capitalize on mean-reverting behavior in stock prices around key "
        "support and resistance levels, relative to their trailing 52-week average. The economic "
        "mechanism exploits the tendency for stock prices to revert back to a historical average "
        "price level after deviating from it."
    )

    def __init__(self, threshold_buy: float = -0.1, threshold_sell: float = 0.1, top_n: int = 20) -> None:
        self._threshold_buy = threshold_buy
        self._threshold_sell = threshold_sell
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=52 * 5)  # 52 weeks in trading days (approx. 260 business days)
        if history.is_empty() or history.height < 260:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        closes = view.closes(lookback=self._threshold_sell + 1)
        if closes.height < 260 or any(symbol not in closes.columns for symbol in symbols):
            return Signal(information_available_at=stamp, weights={})

        avg_prices: pl.DataFrame = history.group_by("symbol").agg(
            (pl.col("adj_close") / pl.col("adj_close").shift(259) - 1.0).alias("return_52w")
        ).sort("return_52w", descending=True)

        picks: list[str] = []
        for symbol in symbols:
            if symbol not in avg_prices.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < 260:
                continue
            latest_close = recent_closes[-1]
            trailing_avg = avg_prices.select(pl.col(symbol)).item()
            deviation_from_avg = (latest_close - trailing_avg) / trailing_avg

            if deviation_from_avg > self._threshold_sell:
                picks.append(symbol)
            elif deviation_from_avg < self._threshold_buy:
                picks.append(symbol)

        picks = picks[: self._top_n]
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