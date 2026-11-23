"""Train a Titanic survival model, track it with MLflow, save model.joblib.

Run:  python src/train.py
"""
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
MODEL_PATH = ROOT / "model.joblib"

NUMERIC_FEATURES = ["pclass", "age", "fare"]
CATEGORICAL_FEATURES = ["sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "survived"

EXPERIMENT_NAME = "facultet-mlops-cicd"
# Hyperparameters are kept in one dict so we can log them to MLflow as-is.
PARAMS = {"n_estimators": 100, "max_depth": 5, "random_state": 42}


def load_data():
    """Read the CSV files and split them into X (features) and y (target)."""
    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    return train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]


def build_pipeline() -> Pipeline:
    """Preprocessing + model in ONE object, so the API applies identical steps."""
    preprocessing = ColumnTransformer(
        [
            # Fill missing numbers (e.g. age) with the median of the TRAIN data.
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            # Turn "male"/"female" into 0/1 columns; unseen values are ignored.
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [("preprocessing", preprocessing), ("model", RandomForestClassifier(**PARAMS))]
    )


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    """Score the model on data it has never seen."""
    predictions = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }


def main() -> None:
    X_train, y_train, X_test, y_test = load_data()
    pipeline = build_pipeline()

    # Local MLflow tracking: runs are stored in a SQLite file, ./mlflow.db
    # (see README to open the UI). No server or cloud account needed.
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)  # fit ONLY on training data
        metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_params(PARAMS)
        mlflow.log_metrics(metrics)
        # cloudpickle format: fine here because we trained (and trust) this model.
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

    joblib.dump(pipeline, MODEL_PATH)

    print("=== Evaluation on held-out test set ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 score: {metrics['f1']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
