from pathlib import Path

from ucimlrepo import fetch_ucirepo


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI Student Performance dataset...")

    dataset = fetch_ucirepo(id=320)

    math_data = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if "G3" in math_data.columns:
        print("G3 found in feature dataframe.")
    else:
        print("G3 is stored separately as target data.")

    math_data["G3"] = targets["G3"].values

    output_path = output_dir / "student-mat.csv"
    math_data.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path}")
    print(f"Rows: {math_data.shape[0]}")
    print(f"Columns: {math_data.shape[1]}")


if __name__ == "__main__":
    main()