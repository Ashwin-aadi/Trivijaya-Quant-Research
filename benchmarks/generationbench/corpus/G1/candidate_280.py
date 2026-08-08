from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts towards its recent average. This strategy identifies stocks that have "
        "deviated significantly from their trailing mean and bets on a return to the mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        trailing_mean: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in trailing_mean:
                mean_value = (
                    history.filter(pl.col("symbol") == symbol)
                    .select(
                        (pl.col("adj_close").mean().alias("mean"))
                    )
                    .to_dict(True)[0]["mean"]
                )
                trailing_mean[symbol] = float(mean_value)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in trailing_mean:
                continue
            latest_close = history.filter(pl.col("symbol") == symbol).select(
                pl.col("adj_close").last().alias("latest")
            ).to_dict(True)[0]["latest"]
            mean_diff = abs(float(latest_close) - trailing_mean[symbol])
            if mean_diff > 5.0:  # Assuming a threshold for deviation
                picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_dict(True)[0]["session_date"]
    assert isinstance(newest, date)
    return newest