from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of strong market sentiment and "
        "can lead to sustained price action. By identifying stocks that have a significant volume "
        "on their recent directionally-moving days, we can potentially profit from the continuation "
        "of such trends."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        grouped = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").max().alias("max_volume"),
                (pl.col("close") - pl.col("open")).abs().mean().alias("avg_abs_move"),
                pl.col("volume").sum().alias("total_volume"),
                pl.col("adj_close").shift(-1).diff().filter(pl.col("adj_close") != 0).count()
                .alias("directional_days")
            )
        )

        # Filter to keep only the most active symbols
        grouped = grouped.filter(
            (pl.col("max_volume") > pl.col("total_volume").quantile(0.75))
            & (pl.col("avg_abs_move") > 1)
        )

        if grouped.height < 3:
            return Signal(information_available_at=stamp, weights={})

        # Find symbols with at least 2 directional days
        directional_symbols = (
            grouped.filter(pl.col("directional_days") >= 2)
            .select(
                pl.col("symbol"),
                (pl.col("total_volume") / self._window).alias("avg_daily_vol")
            )
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in directional_symbols["symbol"].to_list():
                continue
            avg_daily_vol = float(directional_symbols.filter(pl.col("symbol") == symbol)["avg_daily_vol"])
            picks.append(symbol)

        weights = {s: 1.0 / len(picks) for s in picks}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest