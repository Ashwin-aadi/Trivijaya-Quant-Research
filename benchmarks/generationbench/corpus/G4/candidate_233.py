from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy leverages the breakout continuation pattern in Indian equity markets. "
        "It identifies potential breakout points, confirms them using Bollinger Bands, and then looks for"
        "continuation signals such as increased volume or confirmation trends. This behavior persists due to "
        "psychological factors and market participants' expectations following the breakout."
    )

    def __init__(self, window: int = 30, bb_width: float = 2.0) -> None:
        self._window = window
        self._bb_width = bb_width

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history(lookback=self._window)

            # Calculate Bollinger Bands
            mean_close = history.select(pl.col("adj_close").mean().alias("mean"))
            std_close = history.select(pl.col("adj_close").std().alias("std"))
            upper_bb = (history["adj_close"] + self._bb_width * std_close["std"]).to_series()
            lower_bb = (history["adj_close"] - self._bb_width * std_close["std"]).to_series()

            # Find breakout points
            breakout_points = history.select(
                pl.col("session_date"),
                pl.col("symbol"),
                upper_bb,
                lower_bb,
                (pl.col("adj_close") > upper_bb).alias("up_breakout"),
                (pl.col("adj_close") < lower_bb).alias("down_breakout")
            )

            # Filter out symbols that did not break out
            breakout_points = (
                breakout_points.filter((pl.col("up_breakout")) | (pl.col("down_breakout")))
                    .select("symbol", "session_date")
            )
            
            if not breakout_points.is_empty():
                recent_breach = history.join(breakout_points, on="session_date", how="inner").select(
                    pl.col("adj_close"), pl.col("volume")
                ).drop_nulls()
                
                # Volume analysis
                volume_ratio = (recent_breach["volume"] / view.closes().filter(pl.col("symbol") == symbol).tail(1)["volume"].item()).to_series()
                
                if recent_breach.shape[0] > 0 and max(volume_ratio) >= 1.2:
                    picks.append(symbol)

        picks = sorted(picks, key=lambda x: _breakout_score(view.history(lookback=self._window), x), reverse=True)[:30]
        
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


def _breakout_score(history: pl.DataFrame, symbol: str) -> float:
    latest_close = history.select(pl.col("adj_close").last().alias("latest"))
    breakout_point = history.filter(pl.col("symbol") == symbol).select(
        (pl.col("adj_close") > pl.col("adj_close").shift(1)).alias("up_breakout"),
        (pl.col("adj_close") < pl.col("adj_close").shift(1)).alias("down_breakout")
    )
    
    if not breakout_point.is_empty():
        recent_breach = history.join(breakout_point, on="session_date", how="inner")
        
        # Calculate the magnitude of the price movement
        last_close = float(latest_close["latest"])
        prev_close = recent_breach.select(pl.col("adj_close").shift(1)).to_series().item()
        if "up_breakout" in breakout_point.columns and max(breakout_point.filter(pl.col("up_breakout")).select("adj_close")) > prev_close:
            movement_score = (last_close - prev_close) / prev_close
        elif "down_breakout" in breakout_point.columns and min(breakout_point.filter(pl.col("down_breakout")).select("adj_close")) < prev_close:
            movement_score = (prev_close - last_close) / prev_close
        else:
            return 0.0

        # Volume analysis
        volume_ratio = float(
            recent_breach.select((pl.col("volume") / pl.col("volume").shift(1)).alias("volume_ratio")).to_series().max()
        )
        
        score = (movement_score + volume_ratio) / 2
    else:
        return 0.0

    return score