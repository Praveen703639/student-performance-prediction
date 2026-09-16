import subprocess
import sys

STEPS = [
    ("Exploratory Data Analysis", [sys.executable, "-m", "src.visualization.eda"]),
    ("Model Training", [sys.executable, "-m", "src.models.train"]),
    ("Baseline Evaluation", [sys.executable, "-m", "src.models.evaluate"]),
    ("Hyperparameter Tuning", [sys.executable, "-m", "src.models.tune"]),
    ("Tuned Model Evaluation", [sys.executable, "-m", "src.models.evaluate_tuned"]),
    ("Feature Importance", [sys.executable, "-m", "src.models.feature_importance"]),
]

def run_step(name, command):
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nPipeline failed during: {name}")
        sys.exit(result.returncode)

def main():
    print("\nStudent Performance Prediction - ML Pipeline")

    for name, command in STEPS:
        run_step(name, command)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
