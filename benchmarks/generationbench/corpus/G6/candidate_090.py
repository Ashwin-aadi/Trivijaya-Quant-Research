from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RSIRelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks based on their relative strength compared to the NIFTY 50 index "
        "using a 14-day RSI. Stocks with an RSI below 70 and increasing volume are favored as potential long candidates."
    )

    def __init__(self, window: int = 14, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_50_closes = [symbol for symbol in view.symbols if symbol.startswith("NIFTY")]
        if not nifty_50_closes:
            return Signal(information_available_at=stamp, weights={})

        nifty_50_close_df = history.filter(pl.col("symbol").is_in(nifty_50_closes))
        symbols_with_data = [s for s in view.symbols if s not in nifty_50_closes]

        if len(symbols_with_data) < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        rsi_history = _compute_rsi(history.filter(pl.col("symbol").is_in(symbols_with_data)), self._window)
        nifty_50_close_series = nifty_50_close_df["adj_close"]
        symbol_rsis = {s: float(r.to_list()[-1]) for s, r in zip(symbols_with_data, rsi_history)}

        picks: list[str] = []
        for symbol in symbols_with_data:
            if symbol not in symbol_rsis:
                continue
            if symbol_rsis[symbol] < 70 and _volume_increasing(history.filter(pl.col("symbol") == symbol)):
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _compute_rsi(df: pl.DataFrame, window: int) -> list[pl.Series]:
    returns = (df["adj_close"] / df["adj_close"].shift(1) - 1.0).alias("r")
    df_with_returns = df.with_columns(returns)
    r = df_with_returns.sort("session_date").group_by("symbol").agg(
        pl.col("r").rank(method="ordinal", descending=True).alias(f"rank_{window}")
    )
    return [df["symbol"].to_list(), (100.0 - 100.0 / (1 + r[f"rank_{window}"]))]


def _volume_increasing(df: pl.DataFrame) -> bool:
    volume = df.select(pl.col("volume").sum()).to_series().to_list()[0]
    return volume > 0


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest