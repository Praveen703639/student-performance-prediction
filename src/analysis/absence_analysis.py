from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/student-mat.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    summary = (
        df.groupby("G3")["absences"]
        .agg(["count", "mean", "median"])
        .round(2)
    )

    print("===== ABSENCES BY FINAL GRADE =====")
    print(summary.to_string())


if __name__ == "__main__":
    main()