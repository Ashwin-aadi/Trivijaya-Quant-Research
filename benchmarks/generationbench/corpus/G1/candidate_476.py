from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can "
        "potentially lead to sustained price movements. By identifying such moves, we aim "
        "to capitalize on the momentum."
    )

    def __init__(self, window: int = 20, min_volume_factor: float = 1.5) -> None:
        self._window = window
        self._min_volume_factor = min_volume_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed_moves: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_prices = [float(v) for v in history[f"{symbol}_open"].to_list()]
            close_prices = [float(v) for v in history[f"{symbol}_close"].to_list()]
            volumes = [float(v) for v in history[f"{symbol}_volume"].to_list()]

            if len(open_prices) < self._window:
                continue

            # Calculate daily returns
            returns = [(close - open_) / open_ for open_, close in zip(open_prices, close_prices)]
            # Check for a significant price move and corresponding volume increase
            for i in range(1, len(returns)):
                if abs(returns[i]) > 0.05 and volumes[i] > volumes[i-1] * self._min_volume_factor:
                    volume_confirmed_moves[symbol] = returns[i]

        # Sort by the magnitude of the move to prioritize stronger moves
        sorted_symbols = sorted(volume_confirmed_moves.items(), key=lambda x: abs(x[1]), reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_symbols[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest