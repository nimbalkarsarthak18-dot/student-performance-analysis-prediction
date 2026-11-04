"""Download the UCI Student Performance dataset into data/raw/."""
from pathlib import Path
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=320)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if "G3" not in targets.columns:
        raise ValueError("Expected target column 'G3' was not found.")

    combined = features.copy()
    combined["G3"] = targets["G3"].values

    # The UCI dataset has separate mathematics and Portuguese subject files.
    # ucimlrepo exposes the selected dataset data; save the returned table.
    output = RAW_DIR / "student_performance.csv"
    combined.to_csv(output, index=False)

    print(f"Saved {len(combined)} rows and {len(combined.columns)} columns to: {output}")
    print("Review the UCI dataset page and cite the dataset before publishing results.")


if __name__ == "__main__":
    main()
