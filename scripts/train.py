import argparse
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train Titanic survival model"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to feature-engineered training CSV",
    )

    parser.add_argument(
        "--model-output",
        required=True,
        help="Path where the trained model will be saved",
    )

    parser.add_argument(
        "--eval-output",
        required=True,
        help="Path where the evaluation dataset will be saved",
    )

    return parser


def main():
    args = build_parser().parse_args()

    # Load feature-engineered data
    df = pd.read_csv(args.input)

    # Target
    y = df["Survived"].astype("int64")

    # Remove target and ID from model features
    X = df.drop(columns=["Survived", "PassengerId"])

    # Split into training and evaluation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Columns used in the original notebook
    numeric_features = ["Age", "Fare"]
    onehot_features = [
        "Embarked",
        "Pclass",
        "Title",
        "Family_size",
    ]
    ordinal_features = ["Sex"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                MinMaxScaler(),
                numeric_features,
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                onehot_features,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_features,
            ),
        ]
    )

    # Model used in the notebook
    model = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Save trained model
    with open(args.model_output, "wb") as f:
        pickle.dump(pipeline, f)

    # Save evaluation dataset
    eval_df = X_test.copy()
    eval_df["Survived"] = y_test.values
    eval_df.to_csv(args.eval_output, index=False)

    print(f"Saved model to {args.model_output}")
    print(f"Saved evaluation data to {args.eval_output}")


if __name__ == "__main__":
    main()