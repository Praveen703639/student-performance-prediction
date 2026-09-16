from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing.preprocess import prepare_data


MODEL_DIR = Path("models")
RESULTS_DIR = Path("reports/results")

MODEL_NAMES = [
    "ridge",
    "random_forest",
    "gradient_boosting",
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")

    X_train, X_test, y_train, y_test, _ = prepare_data()

    results = []

    print("\n===== TUNED MODEL TEST EVALUATION =====")

    for model_name in MODEL_NAMES:

        model_path = MODEL_DIR / f"{model_name}_tuned.joblib"

        if not model_path.exists():
            print(f"Skipping {model_name}: model not found.")
            continue

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

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

    output_path = RESULTS_DIR / "tuned_evaluation_results.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n===== TUNED MODEL RESULTS =====")
    print(results_df.to_string(index=False))

    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()