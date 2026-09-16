from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from src.preprocessing.preprocess import prepare_data


MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports/results")


def build_models():
    """Create the regression models used in the experiment."""

    return {
        "dummy": DummyRegressor(strategy="mean"),

        "linear_regression": LinearRegression(),

        "ridge": Ridge(alpha=1.0),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")

    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    models = build_models()

    results = []

    print("\n===== MODEL TRAINING =====")

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        print(f"\nTraining: {model_name}")

        pipeline.fit(X_train, y_train)

        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)

        print(f"Training R²: {train_score:.4f}")
        print(f"Testing R²:  {test_score:.4f}")

        model_path = MODEL_DIR / f"{model_name}.joblib"

        joblib.dump(pipeline, model_path)

        print(f"Saved model: {model_path}")

        results.append(
            {
                "model": model_name,
                "train_r2": train_score,
                "test_r2": test_score,
            }
        )

    results_df = pd.DataFrame(results)

    results_path = RESULTS_DIR / "training_results.csv"

    results_df.to_csv(results_path, index=False)

    print("\n===== TRAINING COMPLETE =====")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()