from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from .model_training import MLSignalModel


@dataclass
class HyperparameterOptimizer:
    n_trials: int = 20

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, object], float]:
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": 42,
            }
            model = MLSignalModel(params=params)
            estimator = model.model or model.fit_model_instance()
            return float(cross_val_score(estimator, X, y, cv=3, scoring="f1_macro").mean())

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params, float(study.best_value)


# Monkey patch helper for optuna to obtain estimator without fitting - keeps interface simple
setattr(
    MLSignalModel,
    "fit_model_instance",
    lambda self: __import__("sklearn").ensemble.RandomForestClassifier(**self.params),
)
