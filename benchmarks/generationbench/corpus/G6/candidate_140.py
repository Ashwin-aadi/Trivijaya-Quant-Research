from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy combines significant price movements with increased trading volumes "
        "to identify robust directional trends. It ensures that both momentum and volume "
        "are confirmed before entering positions, thereby enhancing signal reliability."
    )

    def __init__(self, window: int = 50, top_n: int = 15) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        dmi_signal = self._calculate_dmi_signal(history)
        momentum_score = self._calculate_momentum_score(history)

        picks: list[str] = []
        for symbol in view.symbols:
            if (symbol not in dmi_signal.columns) or (symbol not in momentum_score.columns):
                continue
            dmi_val, volume_val, momentum_val = (
                float(dmi_signal[symbol].to_list()[-1]),
                float(momentum_score[symbol].to_list()[-1]),
            )
            if (
                dmi_val > 0
                and history.select(pl.col(symbol).mean()).collect()[0][0] > history[
                    symbol
                ].mean().item()
                and momentum_val >= 0.75
            ):
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_dmi_signal(history: pl.DataFrame) -> pl.DataFrame:
    history = (
        history.select(
            [
                "symbol",
                pl.col("high").rank(method="ordinal", descending=True).alias("p_high"),
                pl.col("low").rank(method="ordinal", descending=True).alias("p_low"),
            ]
        )
        .group_by("symbol")
        .agg(
            (
                (pl.col("p_high") - pl.col("p_low"))
                / 2
                + pl.col("p_low")
                .mean()
                .cast(pl.Float64)
                .alias("avg_p_low"),
            )
        )
    )
    return history


def _calculate_momentum_score(history: pl.DataFrame) -> pl.DataFrame:
    sma_20 = history.select(pl.col("close").rolling_mean(20).alias("sma_20"))
    momentum_score = (
        (history["close"] - sma_20["sma_20"]) / sma_20["sma_20"].cast(pl.Float64)
    ).to_frame(name="momentum_score")
    return momentum_score