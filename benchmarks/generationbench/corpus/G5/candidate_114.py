from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks relative to the NIFTY 100 index, "
        "assuming that these stocks are likely to continue their strong performance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            print(f"History too short for {self.__class__.__name__}. Returning no signal.")
            return Signal(information_available_at=stamp, weights={})

        nifty100_closes = view.closes(lookback=self._window).select(
            pl.col("^NIFTY 100").alias("nifty100")
        )

        if nifty100_closes.height < self._window:
            print(f"NIFTY 100 history too short for {self.__class__.__name__}. Returning no signal.")
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = view.history().with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        valid_symbols = [s for s in view.symbols if s in closes.columns]

        symbol_returns = returns.select(
            pl.all().exclude("session_date"),
            pl.col("symbol").alias("symbol"),
        ).with_columns(
            (pl.col(f"{s}_return") for s in valid_symbols)
        )

        # Calculate mean and std of daily returns
        mean_returns = symbol_returns.groupby("symbol").agg(
            (pl.col(f"{s}_return").mean().alias(f"mean_return_{s}") for s in valid_symbols)
        )
        std_returns = symbol_returns.groupby("symbol").agg(
            (pl.col(f"{s}_return").std().alias(f"std_return_{s}") for s in valid_symbols)
        )

        # Calculate relative strength
        mean_returns_std_adjusted = mean_returns.join(std_returns, on="symbol")
        relative_strength = (
            mean_returns_std_adjusted.with_columns(
                (pl.col(f"mean_return_{s}") / pl.col(f"std_return_{s}").fill_null(0.01)).alias(f"rs_{s}")
                for s in valid_symbols
            )
        ).sort(pl.col("rs_nifty100"), descending=True).head(self._window)

        selected_symbols = [
            row["symbol"] for _, row in relative_strength.iter_rows()
            if "rs_" in row and row[f"rs_{row['symbol']}"] > 0
        ]

        weight = 1.0 / len(selected_symbols) if selected_symbols else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest