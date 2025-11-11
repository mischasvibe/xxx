from __future__ import annotations

import pandas as pd

from .ema_rsi_volume import rsi


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


def compute_reversion_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    bands = bollinger_bands(data["close"], 20, 2)
    data = data.join(bands)
    data["rsi_14"] = rsi(data["close"], 14)
    return data
