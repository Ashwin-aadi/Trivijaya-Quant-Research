from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy identifies stocks with price levels deviating significantly from "
        "their 20-day moving average by more than two standard deviations. These stocks are "
        "expected to revert to their mean prices, providing profitable trading opportunities."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(symbol) for symbol in view.symbols]
        filtered_history = history.filter(pl.col("symbol").is_in(symbols))
        closes = filtered_history.select(["session_date", "adj_close"]).pivot(
            values="adj_close", index="session_date", columns="symbol"
        ).fill_null(None)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in closes.columns:
                continue
            adj_close_series = [float(v) for v in closes[symbol].to_list()]
            mean = sum(adj_close_series) / len(adj_close_series)
            std_dev = (sum((x - mean) ** 2 for x in adj_close_series) / len(
                adj_close_series)) ** 0.5
            latest_price = view.latest_close()[symbol]
            score = abs(latest_price - mean) / std_dev
            if score > 2:
                mean_reversion_scores[symbol] = score

        picks: list[str] = []
        for symbol, score in sorted(mean_reversion_scores.items(), key=lambda x: x[1], reverse=True)[:self._top_n]:
            picks.append(symbol)

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