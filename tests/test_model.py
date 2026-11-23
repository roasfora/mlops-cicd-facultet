"""Model tests: the saved model must exist, predict, and be good enough."""
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["pclass", "sex", "age", "fare"]
MIN_ACCURACY = 0.75  # educational quality gate: CI fails if the model is worse


@pytest.fixture(scope="module")
def model():
    return joblib.load(ROOT / "model.joblib")


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(ROOT / "data" / "test.csv")


def test_model_supports_predictions(model):
    assert hasattr(model, "predict")


def test_number_of_predictions_matches_test_rows(model, test_df):
    assert len(model.predict(test_df[FEATURES])) == len(test_df)


def test_predictions_contain_only_valid_classes(model, test_df):
    assert set(model.predict(test_df[FEATURES])) <= {0, 1}


def test_accuracy_meets_minimum_threshold(model, test_df):
    accuracy = accuracy_score(test_df["survived"], model.predict(test_df[FEATURES]))
    assert accuracy >= MIN_ACCURACY, f"accuracy {accuracy:.3f} < {MIN_ACCURACY}"
