from pathlib import Path
from typing import Tuple

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/raw/student-mat.csv")
TARGET_COLUMN = "G3"

NUMERICAL_FEATURES = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]

CATEGORICAL_FEATURES = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]


def load_dataset() -> pd.DataFrame:
    """Load the raw student performance dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features from the prediction target."""
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline for numeric and categorical data."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def prepare_data():
    """Load, split, and create the preprocessing pipeline."""

    df = load_dataset()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    preprocessor = create_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    print("===== DATA PREPARATION =====")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Training features before encoding: {X_train.shape[1]}")
    print(f"Testing features before encoding:  {X_test.shape[1]}")

    print("\n===== TARGET =====")
    print(f"Training target mean: {y_train.mean():.2f}")
    print(f"Testing target mean:  {y_test.mean():.2f}")

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print("\n===== AFTER PREPROCESSING =====")
    print(f"Training matrix shape: {X_train_transformed.shape}")
    print(f"Testing matrix shape:  {X_test_transformed.shape}")

    print("\nPreprocessing completed successfully.")