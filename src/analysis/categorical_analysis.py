from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/student-mat.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    columns = [
        "sex",
        "school",
        "address",
        "higher",
        "internet",
        "schoolsup",
        "famsup",
        "activities",
        "romantic",
    ]

    for column in columns:
        print(f"\n===== {column.upper()} =====")

        summary = (
            df.groupby(column, observed=True)["G3"]
            .agg(["count", "mean", "median"])
            .round(2)
        )

        print(summary.to_string())


if __name__ == "__main__":
    main()