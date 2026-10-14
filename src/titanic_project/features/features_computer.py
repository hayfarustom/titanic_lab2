import pandas as pd
from .base_features_computer import BaseFeaturesComputer


class TitanicFeaturesComputer(BaseFeaturesComputer):

    def family_size_group(self, number):
        if number == 1:
            return "Alone"
        elif number > 1 and number < 5:
            return "Small"
        else:
            return "Large"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Extract title from passenger name
        df["Title"] = (
            df["Name"]
            .str.split(", ", expand=True)[1]
            .str.split(".", expand=True)[0]
        )

        df["Title"] = df["Title"].replace(
            [
                "Lady",
                "the Countess",
                "Capt",
                "Col",
                "Don",
                "Dr",
                "Major",
                "Rev",
                "Sir",
                "Jonkheer",
                "Dona",
            ],
            "Rare",
        )

        df["Title"] = df["Title"].replace("Mlle", "Miss")
        df["Title"] = df["Title"].replace("Ms", "Miss")
        df["Title"] = df["Title"].replace("Mme", "Mrs")

        # Create family size
        df["Family_size"] = df["SibSp"] + df["Parch"] + 1
        df["Family_size"] = df["Family_size"].apply(
            self.family_size_group
        )

        # Drop unnecessary columns
        df = df.drop(columns=["Name", "Parch", "SibSp", "Ticket"])

        # Convert age to integer
        df["Age"] = df["Age"].astype("int64")

        return df