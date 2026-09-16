from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


MODEL_PATH = Path("models/random_forest_tuned.joblib")
FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tuned Random Forest model...")

    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    random_forest = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = random_forest.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    output_path = RESULTS_DIR / "feature_importance.csv"

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print("\n===== TOP 15 FEATURES =====")
    print(
        importance_df.head(15).to_string(index=False)
    )

    top_features = importance_df.head(15).sort_values(
        "importance"
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances - Tuned Random Forest")

    plt.tight_layout()

    figure_path = FIGURES_DIR / "feature_importance.png"

    plt.savefig(
        figure_path,
        dpi=300,
    )

    plt.close()

    print(f"\nSaved feature importance data: {output_path}")
    print(f"Saved feature importance plot: {figure_path}")


if __name__ == "__main__":
    main()