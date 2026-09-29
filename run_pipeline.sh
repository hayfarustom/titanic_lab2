#!/bin/bash

set -e

echo "Step 1: Preprocessing training data"
uv run python scripts/preprocess.py \
  --input data/titanic/train.csv \
  --output data/processed/train_processed.csv

echo "Step 2: Feature engineering training data"
uv run python scripts/featurize.py \
  --input data/processed/train_processed.csv \
  --output data/processed/train_features.csv

echo "Step 3: Training model"
uv run python scripts/train.py \
  --input data/processed/train_features.csv \
  --model-output models/titanic_model.pkl \
  --eval-output data/processed/eval.csv

echo "Step 4: Evaluating model"
uv run python scripts/evaluate.py \
  --model models/titanic_model.pkl \
  --input data/processed/eval.csv

echo "Step 5: Preprocessing test data"
uv run python scripts/preprocess.py \
  --input data/titanic/test.csv \
  --output data/processed/test_processed.csv

echo "Step 6: Feature engineering test data"
uv run python scripts/featurize.py \
  --input data/processed/test_processed.csv \
  --output data/processed/test_features.csv

echo "Step 7: Making predictions"
uv run python scripts/predict.py \
  --model models/titanic_model.pkl \
  --input data/processed/test_features.csv \
  --output data/processed/predictions.csv

echo "Pipeline completed successfully!"