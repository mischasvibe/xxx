from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=period).mean()
    avg_loss = down.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def compute_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema_20"] = ema(data["close"], 20)
    data["ema_50"] = ema(data["close"], 50)
    data["rsi_14"] = rsi(data["close"], 14)
    data["obv"] = obv(data["close"], data["volume"])
    data["volume_ma"] = data["volume"].rolling(window=20).mean()
    return data
