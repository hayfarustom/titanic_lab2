import argparse
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(description="Create Titanic features")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def family_size_group(number):
    if number == 1:
        return "Alone"
    elif number > 1 and number < 5:
        return "Small"
    else:
        return "Large"


def featurize(df):
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
    df["Family_size"] = df["Family_size"].apply(family_size_group)

    # Drop columns already replaced by engineered features
    df = df.drop(columns=["Name", "Parch", "SibSp", "Ticket"])

    # Match notebook type conversion
    df["Age"] = df["Age"].astype("int64")

    return df


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)
    df = featurize(df)
    df.to_csv(args.output, index=False)

    print(f"Saved features to {args.output}")


if __name__ == "__main__":
    main()