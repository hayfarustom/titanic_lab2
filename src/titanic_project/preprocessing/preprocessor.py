import pandas as pd
from .base_preprocessor import BasePreprocessor


class TitanicPreprocessor(BasePreprocessor):

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Cabin" in df.columns:
            df = df.drop(columns=["Cabin"])

        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].fillna("S")

        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

        if "Age" in df.columns:
            df["Age"] = df["Age"].fillna(
                df.groupby(["Sex", "Pclass"])["Age"].transform("median")
            )

        return df