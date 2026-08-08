from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthIndex(Strategy):
    rationale = (
        "The Relative Strength Index (RSI) measures the strength of a stock relative to "
        "the Nifty 50 index. Stocks with a higher RSI indicate strong relative performance "
        "and are more likely to continue trending positively."
    )

    def __init__(self, window: int = 14, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        nifty_50_history = view.history(lookback=self._window + 1)
        if nifty_50_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_50_close = float(nifty_50_history.filter(pl.col("symbol") == "NIFTY 50").select("adj_close").to_series().item())
        stock_history = view.history(lookback=self._window)

        rsi_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in ["NIFTY 50"] + stock_history.columns:
                continue
            if "NIFTY 50" == symbol:
                continue

            ohlc = (
                stock_history.filter(pl.col("symbol") == symbol)
                .select(["session_date", "open", "high", "low", "close"])
                .sort(by="session_date")
            )

            gains, losses = [], []
            for i in range(1, self._window):
                diff = ohlc["close"].to_list()[i] - ohlc["close"].to_list()[i-1]
                if diff > 0:
                    gains.append(diff)
                else:
                    losses.append(-diff)

            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            rs = avg_gain / (avg_loss + 1e-8)
            rsi_score = 100 - (100 / (1 + rs))
            rsi_scores[symbol] = rsi_score

        relative_strengths = [
            (symbol, rsi_scores[symbol] - rsi_scores["NIFTY 50"]) for symbol in rsi_scores
        ]
        sorted_relative_strengths = sorted(relative_strengths, key=lambda x: x[1], reverse=True)

        picks = [symbol for symbol, _ in sorted_relative_strengths[:self._top_n]]
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