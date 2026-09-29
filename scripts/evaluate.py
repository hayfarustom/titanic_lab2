import argparse
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate Titanic survival model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    return parser


def main():
    args = build_parser().parse_args()

    # Load saved model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Load evaluation data
    df = pd.read_csv(args.input)

    y_true = df["Survived"].astype("int64")
    X = df.drop(columns=["Survived"])

    # Predict and calculate accuracy
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()