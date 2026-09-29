import argparse
import pickle
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict Titanic survival"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()

    # Load trained model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Load feature data
    df = pd.read_csv(args.input)

    # Keep PassengerId for the output
    passenger_ids = df["PassengerId"]

    # Remove PassengerId before prediction
    X = df.drop(columns=["PassengerId"])

    # Make predictions
    predictions = model.predict(X)

    # Save predictions
    output = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })

    output.to_csv(args.output, index=False)

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()