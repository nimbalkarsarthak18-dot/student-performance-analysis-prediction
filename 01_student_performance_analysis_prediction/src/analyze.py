"""Basic exploratory data analysis for the downloaded dataset."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "student_performance.csv"
FIG_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python -m src.download_data")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print("\nShape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types and non-null counts:")
    df.info()
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nNumeric summary:\n", df.describe(include="all").transpose())

    if "G3" not in df:
        raise ValueError("Target column 'G3' is missing.")

    plt.figure(figsize=(9, 5))
    sns.histplot(df["G3"], bins=20, kde=True)
    plt.title("Distribution of Final Grade (G3)")
    plt.xlabel("Final grade")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "g3_distribution.png", dpi=150)
    plt.close()

    numeric = df.select_dtypes(include="number")
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap — Numeric Features")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "numeric_correlation_heatmap.png", dpi=150)
    plt.close()

    df.describe(include="all").transpose().to_csv(METRICS_DIR / "descriptive_summary.csv")
    print(f"\nEDA outputs saved under: {ROOT / 'results'}")


if __name__ == "__main__":
    main()
