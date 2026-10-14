from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from .logistic_regression_model import LogisticRegressionModel


def create_model_pipeline():
    numeric_features = ["Age", "Fare"]

    onehot_features = [
        "Embarked",
        "Pclass",
        "Title",
        "Family_size",
    ]

    ordinal_features = ["Sex"]

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

    model = LogisticRegressionModel()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline