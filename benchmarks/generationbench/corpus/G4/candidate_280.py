from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their historical "
        "price levels and expects reversion to the mean. By buying undervalued stocks and selling "
        "overvalued ones, it aims to benefit from price corrections in financial markets."
    )

    def __init__(self, lookback_years: int = 10, percentile_threshold: float = 0.1) -> None:
        self._lookback_years = lookback_years
        self._percentile_threshold = percentile_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = view.closes(lookback=self._lookback_years * 252)

        # Calculate historical price distributions for each symbol
        historical_data = (
            history.select(
                pl.col("symbol"), "session_date", pl.col("adj_close").alias("close")
            )
            .with_columns(
                (pl.col("close") - pl.col("close").mean()).alias("deviation"),
                ((pl.col("close") / pl.col("close").shift(1)) - 1.0).alias("return"),
            )
            .group_by("symbol", "session_date")
            .agg(pl.col("close").quantile(self._percentile_threshold).alias(f"q{self._percentile_threshold}"))
        )

        # Filter the data for each symbol to get the most recent close
        latest_closes = closes.select(
            [pl.col(symbol) for symbol in symbols]
        ).select(
            [
                (pl.col(symbol) - pl.col(f"q{self._percentile_threshold}{symbol}")) > 0.0
                .alias(f"is_below_{self._percentile_threshold * 100}_th_percentile")
                for symbol in symbols
            ]
        )

        # Identify symbols to buy and sell based on the percentile threshold
        buys = latest_closes.select(
            [
                pl.col(symbol)
                .filter(pl.col(f"is_below_{self._percentile_threshold * 100}_th_percentile"))
                .alias(symbol)
                for symbol in symbols if f"is_below_{self._percentile_threshold * 100}_th_percentile" in latest_closes.columns
            ]
        ).to_dict(False)

        sells = latest_closes.select(
            [
                pl.col(symbol).filter(pl.col(f"is_below_{self._percentile_threshold * 100}_th_percentile") == False)
                .alias(symbol)
                for symbol in symbols if f"is_below_{self._percentile_threshold * 100}_th_percentile" not in latest_closes.columns
            ]
        ).to_dict(False)

        # Prepare the weights dictionary
        weights: dict[str, float] = {}
        buy_count = len(buys)
        sell_count = len(sells)

        if buy_count + sell_count > 50:
            for symbol in buys.keys():
                weights[symbol] = 1.0 / (buy_count * 2)
            for symbol in sells.keys():
                weights[symbol] = -1.0 / (sell_count * 2)
        else:
            for symbol in symbols:
                if symbol in buys:
                    weights[symbol] = 1.0 / buy_count
                elif symbol in sells:
                    weights[symbol] = -1.0 / sell_count

        return Signal(
            information_available_at=stamp,
            weights={k: float(v) for k, v in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest