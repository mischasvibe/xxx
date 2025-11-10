from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..indicators.bollinger_rsi import compute_reversion_indicators
from .base import Strategy, StrategyResult


@dataclass
class MeanReversionStrategy(Strategy):
    name: str = "mean_reversion"
    rsi_lower: float = 30
    rsi_upper: float = 70

    def generate_signal(self, data: pd.DataFrame) -> Optional[StrategyResult]:
        enriched = compute_reversion_indicators(data)
        latest = enriched.iloc[-1]

        if latest["rsi_14"] < self.rsi_lower and latest["close"] < latest["bb_lower"]:
            reason = "RSI oversold with price below lower Bollinger Band"
            return StrategyResult("buy", 0.65, reason)
        if latest["rsi_14"] > self.rsi_upper and latest["close"] > latest["bb_upper"]:
            reason = "RSI overbought with price above upper Bollinger Band"
            return StrategyResult("sell", 0.65, reason)
        return None
