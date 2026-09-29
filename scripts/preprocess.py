import argparse
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(description="Preprocess Titanic data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def preprocess(df):
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


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)
    df = preprocess(df)
    df.to_csv(args.output, index=False)

    print(f"Saved processed data to {args.output}")


if __name__ == "__main__":
    main()