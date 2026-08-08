from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "The relative strength strategy identifies stocks that are outperforming the market "
        "as a whole. This is based on the idea that strong stocks will continue to outperform "
        "weaker ones in the future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        mean_close = history.select(
            pl.col("adj_close").mean().alias("universe_mean")
        ).to_dict(as_series=False)["universe_mean"][0]

        relative_strengths: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            mean_adj_close = history.filter(
                pl.col("symbol") == symbol
            ).select(pl.col("adj_close").mean()).to_dict(as_series=False)["adj_close"][0]

            strength_ratio = (mean_adj_close / mean_close - 1.0)

            relative_strengths.append((symbol, strength_ratio))

        sorted_strengths = sorted(relative_strengths, key=lambda x: x[1], reverse=True)
        top_n_symbols = [s for s, _ in sorted_strengths[:5]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_dict(as_series=False)["session_date"][0]
    assert isinstance(newest, date)
    return newest