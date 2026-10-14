import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(ClassifierMixin, BaseEstimator, BaseModel):
    """Random Forest model for Titanic survival prediction."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)