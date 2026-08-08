from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "This strategy capitalizes on the tendency for stock prices to revert to their historical "
        "average levels based on mean reversion in P/E ratios. Stocks with extreme deviations from "
        "their long-term valuations are expected to correct over a short period."
    )

    def __init__(self, lookback: int = 5, window: int = 10) -> None:
        self._lookback = lookback
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate mean P/E ratio
        symbols = view.symbols
        history = view.history(lookback=self._lookback)

        means: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in history.columns:
                continue

            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            eps = _annualized_eps(history[history["symbol"] == symbol])
            mean_pe = sum(adj_closes[-self._lookback:] / eps) / self._lookback
            means[symbol] = mean_pe

        # Calculate current P/E ratio and deviation from the mean
        pe_ratios: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue
            adj_close = float(closes[symbol][-1])
            eps = _annualized_eps(history[history["symbol"] == symbol])
            current_pe = adj_close / eps
            pe_ratios[symbol] = (current_pe - means[symbol]) / means[symbol]

        # Rank symbols based on P/E deviation from the mean
        ranked_symbols = sorted(pe_ratios, key=pe_ratios.get, reverse=True)

        picks: list[str] = []
        for symbol in ranked_symbols:
            if pe_ratios[symbol] < 0.5:  # Threshold for buying
                picks.append(symbol)
            if len(picks) >= self._window:
                break

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


def _annualized_eps(history: pl.DataFrame) -> float:
    earnings = history.select("adj_close")[-1]
    if not history.is_empty() and len(history) > 12 * 5:  # At least 5 years of data
        eps = (earnings - history.select("adj_close")[0]) / (12 * 5)
    else:
        eps = earnings / 52  # Annualize based on number of trading days

    return float(eps)