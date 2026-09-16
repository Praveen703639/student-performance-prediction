import pandas as pd

from src.preprocessing.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_dataset,
    split_features_target,
)


def test_dataset_loads():
    """Verify that the dataset loads successfully."""

    df = load_dataset()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 649
    assert TARGET_COLUMN in df.columns


def test_expected_features_exist():
    """Verify that all expected input features are present."""

    df = load_dataset()

    expected_features = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    for feature in expected_features:
        assert feature in df.columns


def test_target_has_valid_range():
    """Verify that final grades are within the expected range."""

    df = load_dataset()

    assert df[TARGET_COLUMN].min() >= 0
    assert df[TARGET_COLUMN].max() <= 20


def test_no_missing_values():
    """Verify that the dataset contains no missing values."""

    df = load_dataset()

    assert df.isnull().sum().sum() == 0


def test_feature_target_split():
    """Verify that features and target are separated correctly."""

    df = load_dataset()

    X, y = split_features_target(df)

    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]

    assert X.shape[1] == 30
    assert TARGET_COLUMN not in X.columns