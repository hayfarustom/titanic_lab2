from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    """Abstract base class for preprocessing."""

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Take a DataFrame and return a processed DataFrame."""