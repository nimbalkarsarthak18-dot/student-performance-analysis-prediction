"""Train and compare regression models with a leakage-aware target setup."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "student_performance.csv"
RESULTS_DIR = ROOT / "results" / "metrics"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["early", "midyear"], default="early")
    parser.add_argument("--subject", choices=["math", "portuguese"], default="math",
                        help="Subject label for experiment tracking. Verify the input CSV matches this subject.")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python -m src.download_data")

    df = pd.read_csv(DATA_PATH)
    if "G3" not in df:
        raise ValueError("Target column G3 is missing.")

    y = pd.to_numeric(df["G3"], errors="raise")
    X = df.drop(columns=["G3"]).copy()

    if args.setting == "early":
        X = X.drop(columns=[c for c in ["G1", "G2"] if c in X.columns])

    # Remove rows with missing target; feature imputers are fitted inside each pipeline.
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Dummy_mean_baseline": DummyRegressor(strategy="mean"),
        "Linear_Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "Decision_Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5),
        "Random_Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient_Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Hist_Gradient_Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(),
    }

    rows = []
    fitted = {}
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append({
            "subject_label": args.subject,
            "setting": args.setting,
            "model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "R2": r2_score(y_test, predictions),
        })
        fitted[name] = pipeline

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows).sort_values("RMSE")
    output = RESULTS_DIR / f"model_comparison_{args.subject}_{args.setting}.csv"
    results.to_csv(output, index=False)

    best_name = results.iloc[0]["model"]
    model_path = MODELS_DIR / f"best_{args.subject}_{args.setting}.joblib"
    joblib.dump(fitted[best_name], model_path)

    metadata = {
        "subject_label": args.subject,
        "setting": args.setting,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "selected_by": "lowest held-out test RMSE (exploratory baseline only)",
        "best_model": best_name,
        "warning": "For final research claims, select/tune using cross-validation on training data, then evaluate the test set once.",
    }
    (RESULTS_DIR / f"run_metadata_{args.subject}_{args.setting}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(results.to_string(index=False))
    print(f"\nMetrics: {output}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
