from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - optional dependency
    Prophet = None

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProphetForecaster:
    changepoint_prior_scale: float = 0.1
    seasonality_mode: str = "multiplicative"
    model: Optional[Prophet] = None

    def fit(self, df: pd.DataFrame) -> None:
        if Prophet is None:
            raise ImportError("prophet library is required for forecasting. Install with `pip install prophet`."
            )
        prepared = pd.DataFrame({"ds": df.index, "y": df["close"].values})
        model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
        )
        model.fit(prepared)
        self.model = model

    def forecast(self, periods: int = 24) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Forecaster must be fit before calling forecast().")
        future = self.model.make_future_dataframe(periods=periods, freq="H")
        forecast = self.model.predict(future)
        return forecast.tail(periods)
