from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StrategyResult:
    signal: str
    confidence: float
    reason: str


class Strategy:
    name: str = "base"

    def generate_signal(self, data: pd.DataFrame) -> Optional[StrategyResult]:  # pragma: no cover - interface
        raise NotImplementedError
