from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends while scaling positions based on historical volatility. "
        "High volatility periods reduce position sizes to limit risk, while low volatility allows "
        "for larger bets on established trends."
    )

    def __init__(self, window: int = 20, lookback_volatility: int = 5) -> None:
        self._window = window
        self._lookback_volatility = lookback_volatility

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_close_data = [symbol for symbol in view.symbols if symbol in history.columns]

        volatility_df: pl.DataFrame = (
            view.closes(lookback=self._lookback_volatility)
            .select(pl.col("session_date"))
            .with_column(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .group_by("symbol", maintain_order=True)
            .agg(
                (
                    pl.col("return").std().over(pl.all()).alias("volatility"),
                    (pl.col("adj_close")[-1] / pl.col("adj_close")[0]).alias("trend_factor"),
                )
            )
        )

        volatility_scaled_weights: dict[str, float] = {}
        for symbol in symbols_with_close_data:
            volatility = history.select(pl.col(symbol).std()).item()
            trend_factor = volatility_df.filter(pl.col("symbol") == symbol)["trend_factor"].item()

            weight = (1.0 / len(symbols_with_close_data)) * max(0.1, 1 - 0.9 * (volatility / self._lookback_volatility))
            if not np.isnan(weight):
                volatility_scaled_weights[symbol] = weight * trend_factor

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in volatility_scaled_weights.items() if not np.isnan(w)},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest