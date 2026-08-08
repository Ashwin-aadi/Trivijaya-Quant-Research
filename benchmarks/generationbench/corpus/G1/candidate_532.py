from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion occurs when a stock price returns to its historical average. "
        "This strategy aims to identify stocks that have deviated significantly from their "
        "average and are likely to revert to the mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        reversion_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            adj_closes = history.select(pl.col("adj_close").filter(pl.col("symbol") == symbol))
            mean_adj_close = adj_closes.with_columns(
                (pl.col("adj_close").mean().alias("mean"))
            ).select("mean").first()["mean"]
            latest_close = view.latest_close()[symbol]
            z_score = (latest_close - mean_adj_close) / history.select(
                pl.col("adj_close").mean()
            ).collect().item()

            if abs(z_score) > 1.0:  # Consider stronger signals
                reversion_signals[symbol] = z_score

        if not reversion_signals:
            return Signal(information_available_at=stamp, weights={})

        sorted_signals = sorted(reversion_signals.items(), key=lambda x: abs(x[1]), reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_signals[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest