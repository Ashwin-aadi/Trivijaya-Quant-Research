from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that prices will revert to the mean after a "
        "significant deviation from it. By identifying stocks that have deviated significantly "
        "from their recent means, we can exploit this tendency for potential profits."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean_close")))
            .select(["symbol", "mean_close"])
        )
        closes = view.closes(lookback=self._window)
        latest_closes = view.latest_close()

        z_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in mean_close.columns or symbol not in latest_closes.keys():
                continue
            z_score = (
                (latest_closes[symbol] - mean_close[mean_close["symbol"] == symbol][
                    "mean_close"].to_list()[0])
                / mean_close[
                    mean_close["symbol"] == symbol]["mean_close"].to_list()[0]
            )
            z_scores[symbol] = float(z_score)

        sorted_symbols = [
            s for _, s in sorted(
                z_scores.items(), key=lambda item: abs(item[1]), reverse=True
            )][:5]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

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