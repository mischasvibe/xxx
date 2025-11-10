from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import backtrader as bt
import pandas as pd

from ..strategies.base import Strategy
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PandasData(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("volume", -1),
        ("openinterest", -1),
    )


class SignalStrategy(bt.Strategy):
    params = dict(strategies=None, risk_per_trade=0.01, historical_df=None)

    def __init__(self):
        self.strategies: List[Strategy] = self.params.strategies or []
        self.historical_df: pd.DataFrame = self.params.historical_df

    def next(self):
        current_dt = self.data.datetime.datetime(0)
        current_history = self.historical_df.loc[: current_dt].copy()
        if len(current_history) < 50:
            return
        for strat in self.strategies:
            try:
                result = strat.generate_signal(current_history)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("Strategy %s failed: %s", strat.name, exc)
                continue
            if result is None:
                continue
            size = (self.broker.getvalue() * self.params.risk_per_trade) / self.data.close[0]
            if result.signal == "buy":
                self.buy(size=size)
            elif result.signal == "sell":
                self.sell(size=size)
            logger.info(
                "Executed %s signal with confidence %.2f: %s",
                strat.name,
                result.confidence,
                result.reason,
            )


@dataclass
class BacktestRunner:
    cash: float = 10_000.0
    commission: float = 0.00075
    risk_per_trade: float = 0.01
    strategies: Optional[List[Strategy]] = None

    def run(self, data: pd.DataFrame) -> Dict[str, float]:
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=self.commission)

        feed = PandasData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(
            SignalStrategy,
            strategies=self.strategies or [],
            risk_per_trade=self.risk_per_trade,
            historical_df=data,
        )
        logger.info("Starting backtest with cash %.2f", self.cash)
        cerebro.run()
        final_value = cerebro.broker.getvalue()
        pnl = final_value - self.cash
        logger.info("Backtest completed. Final value: %.2f", final_value)
        return {"final_value": final_value, "pnl": pnl, "return_pct": pnl / self.cash}
