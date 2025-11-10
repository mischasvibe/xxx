from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional

import pandas as pd

from .backtest.backtest_runner import BacktestRunner
from .ml_engine.feature_engineering import build_feature_matrix, create_labels, select_feature_columns
from .ml_engine.model_training import MLSignalModel
from .sentiment.finbert_analysis import FinBERTSentimentAnalyzer
from .sentiment.vader_analysis import VaderSentimentAnalyzer
from .strategies.base import Strategy, StrategyResult
from .strategies.breakout import BreakoutStrategy
from .strategies.orderflow import OrderflowStrategy
from .strategies.reversion import MeanReversionStrategy
from .strategies.trend_following import TrendFollowingStrategy
from .utils.config import DEFAULT_CONFIG, TradingBotConfig
from .utils.data_loader import DataLoader
from .utils.logger import get_logger

logger = get_logger(__name__)


class TradingBot:
    def __init__(self, config: TradingBotConfig = DEFAULT_CONFIG):
        self.config = config
        self.data_loader = DataLoader(config.data)
        self.strategies: List[Strategy] = [
            TrendFollowingStrategy(),
            BreakoutStrategy(),
            MeanReversionStrategy(),
            OrderflowStrategy(),
        ]
        self.ml_model = MLSignalModel(params=config.model.params)
        self.sentiment_finbert: Optional[FinBERTSentimentAnalyzer] = None
        self.sentiment_vader: Optional[VaderSentimentAnalyzer] = None

    def load_data(self) -> pd.DataFrame:
        return self.data_loader.load_historical_data()

    def train_model(self, data: pd.DataFrame) -> None:
        features = build_feature_matrix(data)
        labels = create_labels(features)
        X = select_feature_columns(features)
        y = labels.loc[X.index]
        logger.info("Training ML model with %d samples", len(X))
        self.ml_model.fit(X, y)

    def sentiment_filter(self, text: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if self.config.sentiment.enable_finbert:
            self.sentiment_finbert = self.sentiment_finbert or FinBERTSentimentAnalyzer()
            scores.update(self.sentiment_finbert.analyze(text))
        if self.config.sentiment.enable_vader:
            self.sentiment_vader = self.sentiment_vader or VaderSentimentAnalyzer()
            scores.update(self.sentiment_vader.analyze(text))
        return scores

    def evaluate_strategies(self, data: pd.DataFrame) -> Dict[str, Optional[StrategyResult]]:
        results: Dict[str, Optional[StrategyResult]] = {}
        for strat in self.strategies:
            try:
                result = strat.generate_signal(data)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("Strategy %s failed during evaluation: %s", strat.name, exc)
                result = None
            results[strat.name] = result
        return results

    def backtest(self, data: pd.DataFrame) -> Dict[str, float]:
        runner = BacktestRunner(
            cash=self.config.backtest.initial_cash,
            commission=self.config.backtest.commission,
            risk_per_trade=self.config.backtest.risk_per_trade,
            strategies=self.strategies,
        )
        return runner.run(data)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self.config)


def main() -> None:
    bot = TradingBot()
    data = bot.load_data()
    bot.train_model(data)
    sentiment_scores = bot.sentiment_filter("Bitcoin surges as institutional demand grows.")
    logger.info("Sentiment scores: %s", sentiment_scores)
    strategy_results = bot.evaluate_strategies(data)
    logger.info("Strategy snapshot: %s", strategy_results)
    metrics = bot.backtest(data.tail(500))
    logger.info("Backtest metrics: %s", metrics)


if __name__ == "__main__":
    main()
