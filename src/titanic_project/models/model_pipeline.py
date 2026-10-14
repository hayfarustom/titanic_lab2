from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel


def create_model_pipeline(model_type="logistic"):
    numeric_features = ["Age", "Fare"]

    onehot_features = [
        "Embarked",
        "Pclass",
        "Title",
        "Family_size",
    ]

    ordinal_features = ["Sex"]

    # Data preprocessing
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

    # Choose the machine learning model
    if model_type == "logistic":
        model = LogisticRegressionModel()

    elif model_type == "random_forest":
        model = RandomForestModel()

    elif model_type == "xgboost":
        model = XGBoostModel()

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    # Build the pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline