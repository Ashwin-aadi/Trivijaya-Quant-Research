from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy aims to exploit the economic mechanism where liquid stocks often outperform less liquid ones due to lower transaction costs and better price discovery. By equal-weighting a portfolio of top-liquid stocks, the strategy seeks to balance exposure across a diversified set while capturing the benefits of higher liquidity."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        liquidity_scores: dict[str, float] = {}

        for symbol in symbols:
            volume = history[symbol + "_volume"]
            open_price = history[symbol + "_open"]
            high = history[symbol + "_high"]
            low = history[symbol + "_low"]
            close = history[symbol + "adj_close"]

            turnover_ratio = (2 * volume) / (high - low)
            average_true_range = (close.diff().abs().mean())  # 30-day ATR
            price_impact = (high - low) / volume

            liquidity_score = (
                float(turnover_ratio.mean())
                + float(average_true_range.mean())
                + float(price_impact.mean())
            )
            liquidity_scores[symbol] = liquidity_score

        sorted_symbols = [
            s for _, s in sorted(liquidity_scores.items(), key=lambda item: -item[1])
        ][:30]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest