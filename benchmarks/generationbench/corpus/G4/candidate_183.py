from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy focuses on equal weighting among highly liquid stocks to capitalize "
        "on the liquidity premium and reduce idiosyncratic risk. By screening for liquidity "
        "metrics such as trading volume and bid-ask spread, we ensure that only welliquid "
        "stocks are included in the portfolio."
    )

    def __init__(self, window: int = 20, top_n: int = 200) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = {}
        for symbol in view.symbols:
            row = history.filter(pl.col("symbol") == symbol).select(
                "session_date", "close", "volume"
            ).sort("session_date")

            if row.is_empty() or row.height < self._window:
                continue

            latest_close = float(row.select("close").last().item())
            volume = [float(v) for v in row.select("volume").to_list()[0]]
            bid_ask_spread = 1 - (row.filter(pl.col("session_date") == stamp).select(
                "open", "close"
            ).select(
                (pl.col("open") / pl.col("close")).alias("bid_ask_spread")
            ).last().item())

            liquidity_score = (
                volume[-1] * bid_ask_spread
            )  # Simple heuristic for liquidity score

            liquidity_scores[symbol] = liquidity_score

        ranked_symbols = sorted(liquidity_scores, key=liquidity_scores.get)[: self._top_n]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest