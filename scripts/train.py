import argparse
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from titanic_project.models.model_pipeline import create_model_pipeline


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train Titanic survival model"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--eval-output", required=True)

    return parser


def main():
    args = build_parser().parse_args()

    # Load feature-engineered data
    df = pd.read_csv(args.input)

    # Separate target and features
    y = df["Survived"].astype("int64")
    X = df.drop(columns=["Survived", "PassengerId"])

    # Split training and evaluation data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create and train the model
    pipeline = create_model_pipeline()
    pipeline.fit(X_train, y_train)

    # Save the trained model
    with open(args.model_output, "wb") as f:
        pickle.dump(pipeline, f)

    # Save evaluation data
    eval_df = X_test.copy()
    eval_df["Survived"] = y_test.values
    eval_df.to_csv(args.eval_output, index=False)

    print(f"Saved model to {args.model_output}")
    print(f"Saved evaluation data to {args.eval_output}")


if __name__ == "__main__":
    main()