# Student Performance Analysis & Prediction

A reproducible machine learning project that explores student academic performance and builds regression models to estimate final grades from information available at prediction time.

> **Status:** Initial project scaffold. Dataset exploration and model experiments must be run locally. No model scores or findings are claimed until experiments have been executed.

## Project overview

This project uses the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva) to investigate how student, family, lifestyle and academic attributes relate to final grades.

The target is `G3` (final grade). The repository is designed around two separate prediction settings:

1. **Early prediction:** excludes `G1` and `G2`, which are grades from earlier assessment periods.
2. **Mid-year prediction:** includes `G1` and `G2`, when those grades would genuinely be available.

Keeping these settings separate helps avoid accidentally using information unavailable at the intended prediction time. Results from the two settings should not be compared as if they solve exactly the same task.

## Research questions

- How is the final grade (`G3`) distributed?
- Which variables have the strongest associations with `G3`?
- How do linear and nonlinear regression models compare?
- How does access to earlier period grades change predictive performance?
- Which errors are common, and what are the limitations of these predictions?

## Dataset

- **Source:** UCI Machine Learning Repository — Student Performance
- **UCI dataset ID:** 320
- **Data:** Mathematics and Portuguese language student records
- **Target:** `G3`
- **License:** CC BY 4.0 (see the UCI dataset page for attribution and terms)

The UCI dataset includes two subject files. The downloader in `src/download_data.py` downloads both and writes local CSV files under `data/raw/`.

The dataset contains sensitive contextual attributes. Use it for educational analysis, not to label, rank, punish, or make high-stakes decisions about individual students. Associations do not establish causation.

## Repository structure

```text
01_student_performance_analysis_prediction/
├── data/
│   ├── raw/                 # Downloaded source CSVs (not committed)
│   └── processed/           # Prepared data (not committed)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/
│   ├── figures/             # Generated plots
│   └── metrics/             # Generated evaluation tables
├── models/                  # Saved models (not committed by default)
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── analyze.py
│   └── train.py
├── tests/
│   └── test_data_loading.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-analysis-prediction.git
cd student-performance-analysis-prediction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python -m src.download_data
```

The downloader uses `ucimlrepo` and stores the two subject datasets in `data/raw/`.

### 5. Explore the data

```bash
python -m src.analyze
```

Generated charts and summary files are written to `results/`.

### 6. Train and compare models

```bash
python -m src.train --subject math --setting early
```

Other options:

```bash
python -m src.train --subject portuguese --setting early
python -m src.train --subject math --setting midyear
python -m src.train --subject portuguese --setting midyear
```

The training script uses a held-out test set and a preprocessing pipeline. It reports MAE, RMSE and R². Model selection and tuning should be performed using training data/cross-validation only; do not repeatedly tune against the test set.

## Methodology

### Target and prediction settings

- `G3`: final grade, target variable.
- Early setting: removes `G1` and `G2`.
- Mid-year setting: includes `G1` and `G2`.

The early setting is more appropriate for investigating prediction before those assessment grades exist. The mid-year setting answers a different question: prediction after earlier grades are known.

### Models

The starter training script compares:

- DummyRegressor (baseline)
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- HistGradientBoosting Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regression

Categorical variables are one-hot encoded; numerical variables are passed through a preprocessing pipeline. Scaling is applied where appropriate.

### Evaluation metrics

- **MAE:** average absolute prediction error, in grade points.
- **RMSE:** penalizes larger errors more strongly, in grade points.
- **R²:** proportion of target variance explained relative to a mean baseline; it can be negative on test data.

Always interpret metrics in the context of the dataset and prediction setting. A high score does not mean the model is suitable for real educational decisions.

## Reproducibility

- Set a random seed in the training script.
- Keep the downloaded raw data unchanged.
- Record package versions and experiment settings.
- Save generated metrics and plots.
- Report the exact subject and prediction setting with every result.
- Do not report results until the scripts have been run successfully.

## Limitations and responsible use

- This is a relatively small, context-specific dataset and may not generalize to other schools, countries or cohorts.
- Some features may encode socioeconomic or personal circumstances.
- Correlation does not imply causation.
- Predictions contain uncertainty and may be inaccurate for individual students.
- Do not use this project for admissions, grading, discipline, funding, or other high-impact decisions.
- Check subgroup errors and potential bias before making any claims about fairness.
- Respect the dataset license and cite the original dataset authors.

## Future improvements

- Add cross-validation and nested hyperparameter tuning.
- Compare error distributions and subgroup performance.
- Add permutation importance and SHAP explanations with careful interpretation.
- Add prediction intervals or calibrated uncertainty estimates.
- Build a small educational demonstration app with clear limitations.
- Add automated tests and continuous integration.

## Citation

Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance*. In Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008), 5–12.

Dataset: [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).

## Author

**YOUR NAME**

- GitHub: https://github.com/YOUR-USERNAME
- LinkedIn: YOUR-LINKEDIN-URL
- Email: YOUR-PUBLIC-EMAIL

Replace the placeholders above before publishing.
