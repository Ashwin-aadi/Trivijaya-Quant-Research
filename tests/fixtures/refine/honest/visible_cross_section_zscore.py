"""Buy the names whose latest one-session move was weakest against that session's cross-section."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible, stdev, top_n


class VisibleCrossSectionZscore(Strategy):
    """Standardises the last visible session's move across the names trading that session."""

    rationale = (
        "Short-horizon reversal is one of the better-documented cross-sectional regularities: a "
        "name that fell sharply relative to its peers on one session tends to recover part of "
        "that move over the next few. Standardising within the session removes the market-wide "
        "component, so what is ranked is the name-specific part. The effect is small, decays "
        "within days, and turns over the whole book daily, so costs will very likely eat it."
    )

    def __init__(self, holdings: int = 10) -> None:
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=2)
        if closes.height < 2:
            return Signal(information_available_at=stamp, weights={})

        moves: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            pair = closes[symbol].to_list()
            if pair[0] is None or pair[1] is None or float(pair[0]) <= 0:
                continue
            moves[symbol] = float(pair[1]) / float(pair[0]) - 1.0
        if len(moves) < 3:
            return Signal(information_available_at=stamp, weights={})

        # The centre and the dispersion are both drawn from this one visible session's
        # cross-section. Pooling either across dates would push the level of later periods into
        # the score of an earlier one, which is the leak this construction is often mistaken for.
        values = list(moves.values())
        centre = sum(values) / len(values)
        dispersion = stdev(values)
        if dispersion <= 0:
            return Signal(information_available_at=stamp, weights={})

        scores = {symbol: (move - centre) / dispersion for symbol, move in moves.items()}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )
