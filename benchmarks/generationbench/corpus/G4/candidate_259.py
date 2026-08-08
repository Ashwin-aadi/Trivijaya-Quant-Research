from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionAndRangeCompression(Strategy):
    rationale = (
        "This strategy exploits the theme of dispersion or range compression in the Indian equity market. "
        "It identifies stocks with historically high volatility (dispersion) and those experiencing significant range compression. "
        "The strategy aims to capitalize on these patterns by dynamically adjusting positions based on predefined thresholds."
    )

    def __init__(self, lookback: int = 20, dispersion_threshold: float = 1.5, bollinger_std_multiplier: float = 2.0) -> None:
        self._lookback = lookback
        self._dispersion_threshold = dispersion_threshold
        self._bollinger_std_multiplier = bollinger_std_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        atras = []
        lower_bollinger_bands = []

        for symbol in symbols:
            close_series = history[symbol].select("close")
            atr = _atr(close_series)
            bollinger_band = _bollinger_band(close_series, self._bollinger_std_multiplier)
            if len(atr) >= 1 and len(bollinger_band) >= 2:  # Ensure sufficient data points
                atras.append(atr[-1])
                lower_bollinger_bands.append(bollinger_band["lower"].item())

        dispersion_scores = {symbol: atr / pl.col("historic_atr").mean().item() for symbol, atr in zip(symbols, atras)}
        range_compression_scores = {
            symbol: (history[history["symbol"] == symbol]["close"].max().item() - history[history["symbol"] == symbol]["low"].max().item()) / lower_bollinger_band
            for symbol, lower_bollinger_band in zip(symbols, lower_bollinger_bands)
        }

        dispersion_sorted = sorted(dispersion_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        range_compression_sorted = sorted(range_compression_scores.items(), key=lambda x: abs(x[1]))[:20]

        weights = {}
        for symbol, _ in dispersion_sorted:
            if dispersion_scores[symbol] > self._dispersion_threshold:
                weight = 0.05 / len(dispersion_sorted)
                weights[symbol] = weight
        for symbol, _ in range_compression_sorted:
            weight = 0.03 / len(range_compression_sorted)
            weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest


def _atr(close_series: pl.Series) -> pl.Series:
    high_low_diff = close_series.shift(-1) - close_series
    true_range_1 = close_series - close_series.shift(1)
    true_range_2 = close_series - close_series.shift(-1)
    atr = (high_low_diff.abs().max(true_range_1).max(true_range_2)).alias("atr")
    return close_series.frame.join(atr, on="session_date", how="inner")


def _bollinger_band(close_series: pl.Series, std_multiplier: float) -> pl.DataFrame:
    sma = close_series.mean()
    std_dev = (close_series - sma).std().item()
    lower_band = sma - std_multiplier * std_dev
    return pl.DataFrame({"symbol": [close_series.name], "lower": [lower_band]})