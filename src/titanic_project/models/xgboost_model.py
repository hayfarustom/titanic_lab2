import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from .base_model import BaseModel


class XGBoostModel(ClassifierMixin, BaseEstimator, BaseModel):
    """XGBoost model for Titanic survival prediction."""

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)