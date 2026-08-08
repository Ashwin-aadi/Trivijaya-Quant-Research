from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower risk and can provide more stable returns. "
        "By tilting the portfolio towards these stocks, we aim to reduce overall portfolio risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        volatilities = {symbol: float(v) for symbol, v in _calculate_volatility(history[symbols]).items()}
        sorted_symbols = sorted(volatilities.keys(), key=lambda x: volatilities[x])
        
        top_n_symbols = sorted_symbols[:5]  # Top 5 low-volatility symbols
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(df: pl.DataFrame) -> dict[str, float]:
    daily_returns = (df.select(pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0)).to_series().abs().mean()
    volatility = df.groupby("symbol").agg(daily_returns.alias("volatility")).collect()["volatility"].to_list()
    
    symbol_volatilities = {row["symbol"]: row["volatility"] for _, row in df.groupby("symbol").agg(
        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("daily_return")
    ).collect().iter_rows()}
    
    return symbol_volatilities