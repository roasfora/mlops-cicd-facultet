"""Data tests: check the DATA before trusting any model trained on it."""
from pathlib import Path

import pandas as pd
import pytest

TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"
REQUIRED_COLUMNS = ["pclass", "sex", "age", "fare", "survived"]


@pytest.fixture(scope="module")  # loaded once, shared by all tests below
def train():
    return pd.read_csv(TRAIN_PATH)


def test_required_columns_exist(train):
    missing = [c for c in REQUIRED_COLUMNS if c not in train.columns]
    assert not missing, f"Missing columns: {missing}"


def test_training_dataset_is_not_empty(train):
    assert len(train) > 0


def test_no_missing_target_values(train):
    assert train["survived"].notna().all()


def test_target_contains_only_0_and_1(train):
    assert set(train["survived"].unique()) <= {0, 1}


def test_feature_data_types_are_valid(train):
    assert pd.api.types.is_integer_dtype(train["pclass"])
    assert pd.api.types.is_float_dtype(train["age"])
    assert pd.api.types.is_float_dtype(train["fare"])
    assert train["sex"].isin(["male", "female"]).all()
