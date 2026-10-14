import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel


class LogisticRegressionModel(ClassifierMixin, BaseEstimator, BaseModel):
    """Logistic Regression model for Titanic survival prediction."""

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)