from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/raw/student-mat.csv")
OUTPUT_DIR = Path("reports/figures")


def load_data() -> pd.DataFrame:
    """Load the raw student performance dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


def create_target_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of final grades."""
    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="G3",
        order=sorted(df["G3"].unique()),
    )

    plt.title("Distribution of Final Student Grades (G3)")
    plt.xlabel("Final Grade (G3)")
    plt.ylabel("Number of Students")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "target_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_numeric_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create a correlation heatmap for numeric variables."""
    numeric_df = df.select_dtypes(include=["number"])

    correlation = numeric_df.corr()

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )

    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "numeric_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_absences_vs_grade(df: pd.DataFrame) -> None:
    """Visualize the relationship between absences and final grade."""
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="G3",
        y="absences",
    )

    plt.title("Absences by Final Grade")
    plt.xlabel("Final Grade (G3)")
    plt.ylabel("Number of Absences")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "absences_vs_grade.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_studytime_vs_grade(df: pd.DataFrame) -> None:
    """Visualize final grades across study-time categories."""
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="studytime",
        y="G3",
    )

    plt.title("Final Grade by Weekly Study Time")
    plt.xlabel("Study Time Category")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "studytime_vs_grade.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    """Print useful dataset statistics."""
    print("\n===== DATASET SUMMARY =====")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n===== TARGET SUMMARY =====")
    print(df["G3"].describe())

    print("\n===== TARGET COUNTS =====")
    print(df["G3"].value_counts().sort_index())

    print("\n===== NUMERICAL FEATURES =====")
    print(df.select_dtypes(include=["number"]).columns.tolist())

    print("\n===== CATEGORICAL FEATURES =====")
    print(df.select_dtypes(include=["object", "string"]).columns.tolist())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print_summary(df)

    create_target_distribution(df)
    create_numeric_correlation_heatmap(df)
    create_absences_vs_grade(df)
    create_studytime_vs_grade(df)

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()