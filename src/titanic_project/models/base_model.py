from abc import ABC, abstractmethod
import pandas as pd


class BaseModel(ABC):
    """Base interface for machine learning models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """Make predictions using the trained model."""
        pass