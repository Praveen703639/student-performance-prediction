from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.preprocessing.preprocess import prepare_data


MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports/results")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")

    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = {
        "ridge": (
            Ridge(),
            {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),

        "random_forest": (
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [0.7, 1.0],
            },
        ),

        "gradient_boosting": (
            GradientBoostingRegressor(
                random_state=42,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [2, 4],
            },
        ),
    }

    tuning_results = []

    print("\n===== HYPERPARAMETER TUNING =====")

    for model_name, (model, parameter_grid) in models.items():

        print(f"\nTuning: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X_train, y_train)

        best_rmse = -search.best_score_

        print(f"Best CV RMSE: {best_rmse:.4f}")
        print("Best parameters:")
        print(search.best_params_)

        model_path = MODEL_DIR / f"{model_name}_tuned.joblib"

        joblib.dump(search.best_estimator_, model_path)

        print(f"Saved tuned model: {model_path}")

        tuning_results.append(
            {
                "model": model_name,
                "best_cv_rmse": best_rmse,
                "best_params": str(search.best_params_),
            }
        )

    results_df = pd.DataFrame(tuning_results)

    output_path = RESULTS_DIR / "tuning_results.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n===== TUNING COMPLETE =====")
    print(results_df.to_string(index=False))

    print(f"\nSaved tuning results: {output_path}")


if __name__ == "__main__":
    main()