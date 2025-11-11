from __future__ import annotations

import pandas as pd


def vwap(df: pd.DataFrame) -> pd.Series:
    price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_volume = df["volume"].cumsum()
    cumulative_pv = (price * df["volume"]).cumsum()
    return cumulative_pv / cumulative_volume


def asian_session_range(df: pd.DataFrame) -> pd.DataFrame:
    session = df.between_time("00:00", "06:00")
    if session.empty:
        return pd.DataFrame(index=df.index, columns=["range_high", "range_low"]).fillna(method="ffill")

    range_high = session["high"].resample("1D").max()
    range_low = session["low"].resample("1D").min()

    range_high = range_high.reindex(df.index, method="ffill")
    range_low = range_low.reindex(df.index, method="ffill")

    return pd.DataFrame({"range_high": range_high, "range_low": range_low})


def compute_breakout_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["vwap"] = vwap(data)
    ranges = asian_session_range(data)
    data = data.join(ranges)
    data["volume_ma"] = data["volume"].rolling(window=30).mean()
    return data
