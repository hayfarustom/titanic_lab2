from abc import ABC, abstractmethod
import pandas as pd


class BaseFeaturesComputer(ABC):

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with engineered features."""