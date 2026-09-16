from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing.preprocess import prepare_data


MODEL_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")

MODEL_NAMES = [
    "dummy",
    "linear_regression",
    "ridge",
    "random_forest",
    "gradient_boosting",
]


def evaluate_predictions(y_true, predictions):
    """Calculate regression metrics."""

    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    return mae, rmse, r2


def create_actual_vs_predicted_plot(
    y_true,
    predictions,
    model_name,
):
    """Create an actual vs predicted scatter plot."""

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=y_true,
        y=predictions,
        s=60,
    )

    min_value = min(y_true.min(), predictions.min())
    max_value = max(y_true.max(), predictions.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Actual Final Grade (G3)")
    plt.ylabel("Predicted Final Grade (G3)")
    plt.title(f"Actual vs Predicted - {model_name}")

    plt.tight_layout()

    output_path = FIGURES_DIR / f"{model_name}_actual_vs_predicted.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def create_residual_plot(
    y_true,
    predictions,
    model_name,
):
    """Create a residual analysis plot."""

    residuals = y_true - predictions

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=predictions,
        y=residuals,
        s=60,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Predicted Final Grade (G3)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"Residual Analysis - {model_name}")

    plt.tight_layout()

    output_path = FIGURES_DIR / f"{model_name}_residuals.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def create_model_comparison_plot(results_df):
    """Create a comparison of model performance."""

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=results_df,
        x="model",
        y="RMSE",
    )

    plt.xlabel("Model")
    plt.ylabel("Test RMSE")
    plt.title("Model Comparison - Test RMSE")

    plt.xticks(rotation=25)

    plt.tight_layout()

    output_path = FIGURES_DIR / "model_comparison_rmse.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")

    X_train, X_test, y_train, y_test, _ = prepare_data()

    all_predictions = {}

    evaluation_results = []

    print("\n===== PREDICTION ANALYSIS =====")

    for model_name in MODEL_NAMES:

        model_path = MODEL_DIR / f"{model_name}.joblib"

        if not model_path.exists():
            print(f"Skipping {model_name}: model not found.")
            continue

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        mae, rmse, r2 = evaluate_predictions(
            y_test,
            predictions,
        )

        all_predictions[model_name] = predictions

        evaluation_results.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(f"\n{model_name}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        create_actual_vs_predicted_plot(
            y_test,
            predictions,
            model_name,
        )

        create_residual_plot(
            y_test,
            predictions,
            model_name,
        )

    results_df = pd.DataFrame(evaluation_results)

    create_model_comparison_plot(results_df)

    prediction_df = pd.DataFrame(
        {
            "actual_G3": y_test.reset_index(drop=True),
        }
    )

    for model_name, predictions in all_predictions.items():
        prediction_df[f"{model_name}_prediction"] = predictions

    predictions_path = RESULTS_DIR / "test_predictions.csv"

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\n===== PREDICTION ANALYSIS COMPLETE =====")

    print("\nPrediction sample:")
    print(prediction_df.head(10).to_string(index=False))

    print(f"\nSaved predictions: {predictions_path}")


if __name__ == "__main__":
    main()