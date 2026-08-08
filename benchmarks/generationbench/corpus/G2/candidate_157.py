from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that assets which have moved significantly away from their "
        "historical average price will revert to it. By identifying such assets and going long, "
        "one can profit from the mean-reverting behavior of financial markets."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        filtered_history = history.select(["session_date", *symbols])

        means = (
            filtered_history.group_by("session_date")
                             .agg([pl.col(symbol).mean().alias(f"mean_{symbol}") for symbol in symbols])
        )
        latest_means = means.sort("session_date", descending=False).tail(1)

        deviations = (
            history.select(["session_date"] + symbols)
                  .with_columns(
                      [(pl.col(symbol) - pl.col(f"mean_{symbol}")).alias(f"deviation_{symbol}") for symbol in symbols]
                  )
        )

        mean_deviations = (latest_means.join(deviations, on="session_date", how="left")
                           .select([*symbols, *[f"deviation_{symbol}" for symbol in symbols]])
                           .to_dict(as_series=False))

        picks: list[str] = []
        for symbol in symbols:
            if mean_deviations[symbol][-1] < -3 * pl.col(f"mean_{symbol}").std().over("session_date"):
                picks.append(symbol)

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