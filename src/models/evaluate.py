from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from src.preprocessing.preprocess import prepare_data


MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports/results")


MODEL_NAMES = [
    "dummy",
    "linear_regression",
    "ridge",
    "random_forest",
    "gradient_boosting",
]


def evaluate_model(model, X_test, y_test):
    """Calculate regression metrics on the test set."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2, predictions


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")

    X_train, X_test, y_train, y_test, _ = prepare_data()

    results = []

    print("\n===== TEST SET EVALUATION =====")

    for model_name in MODEL_NAMES:

        model_path = MODEL_DIR / f"{model_name}.joblib"

        if not model_path.exists():
            print(f"Skipping {model_name}: model not found.")
            continue

        model = joblib.load(model_path)

        mae, rmse, r2, predictions = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print(f"\nModel: {model_name}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        results.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    results_df = pd.DataFrame(results)

    test_results_path = RESULTS_DIR / "evaluation_results.csv"
    results_df.to_csv(test_results_path, index=False)

    print("\n===== TEST RESULTS SUMMARY =====")
    print(results_df.to_string(index=False))

    print("\n===== 5-FOLD CROSS-VALIDATION =====")

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_results = []

    for model_name in MODEL_NAMES:

        model_path = MODEL_DIR / f"{model_name}.joblib"

        if not model_path.exists():
            continue

        model = joblib.load(model_path)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        rmse_scores = -scores

        mean_rmse = rmse_scores.mean()
        std_rmse = rmse_scores.std()

        print(
            f"{model_name}: "
            f"RMSE = {mean_rmse:.4f} "
            f"+/- {std_rmse:.4f}"
        )

        cv_results.append(
            {
                "model": model_name,
                "CV_RMSE_mean": mean_rmse,
                "CV_RMSE_std": std_rmse,
            }
        )

    cv_results_df = pd.DataFrame(cv_results)

    cv_results_path = RESULTS_DIR / "cross_validation_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)

    print("\n===== CROSS-VALIDATION SUMMARY =====")
    print(cv_results_df.to_string(index=False))

    print("\n===== EVALUATION COMPLETE =====")
    print(f"Saved test results: {test_results_path}")
    print(f"Saved CV results:   {cv_results_path}")


if __name__ == "__main__":
    main()