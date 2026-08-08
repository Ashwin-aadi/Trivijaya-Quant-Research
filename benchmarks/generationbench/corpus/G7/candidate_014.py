from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "This strategy leverages volatility-scaled trend-following to identify medium-term "
        "trends. By correlating daily close and volume with historical volatility, it aims to "
        "capture trading activity that is more likely to indicate sustained price movements."
    )

    def __init__(self, window_close_volume: int = 30, window_volatility: int = 100, top_n: int = 5) -> None:
        self._window_close_volume = window_close_volume
        self._window_volatility = window_volatility
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_close_volume + self._window_volatility - 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_close_vol_df = view.closes(lookback=self._window_close_volume)
        volatility_df = _calculate_volatility(view.history(), self._window_volatility)
        
        signals: dict[str, float] = {}
        
        for symbol in view.symbols:
            if symbol not in latest_close_vol_df.columns or symbol not in volatility_df.columns:
                continue
            
            close = [float(v) for v in latest_close_vol_df[symbol].drop_nulls().to_list()]
            volume = [float(v) for v in history.select(pl.col(symbol)).to_series().to_list()]
            
            if len(close) < self._window_close_volume or len(volume) < self._window_close_volume:
                continue
            
            corr = _compute_corr(close, volume)
            vol = float(volatility_df[volatility_df["symbol"] == symbol]["vol"].item())
            
            if vol > 0.0:
                scaled_corr = corr / vol
                if abs(scaled_corr) >= 0.3:  # Threshold for significant trend
                    signals[symbol] = 1.0 / self._top_n

        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        picks = [symbol for symbol, _ in sorted_signals[:self._top_n]]
        
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(history: pl.DataFrame, window: int) -> pl.DataFrame:
    volatility_df = history.group_by("symbol").agg(
        (pl.col("adj_close") - pl.col("adj_close").shift(1)).pow(2).mean().alias("vol_sq"),
        (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean().alias("vol_abs")
    ).with_columns(
        ((pl.col("vol_sq") + 1e-8).sqrt() * 100 / window).alias("vol")
    )
    
    return volatility_df


def _compute_corr(close: list[float], volume: list[float]) -> float:
    n = len(close)
    mean_close = sum(close) / n
    mean_volume = sum(volume) / n
    
    cov = sum((close[i] - mean_close) * (volume[i] - mean_volume) for i in range(n)) / n
    std_close = (sum((close[i] - mean_close) ** 2 for i in range(n)) / n) ** 0.5
    std_volume = (sum((volume[i] - mean_volume) ** 2 for i in range(n)) / n) ** 0.5
    
    if std_close * std_volume == 0:
        return 0.0
    
    corr = cov / (std_close * std_volume)
    
    return corr