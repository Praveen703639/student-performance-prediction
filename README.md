# Student Performance Prediction

<p align="center">
  <b>End-to-End Machine Learning Regression Pipeline</b><br>
  Predicting final student performance from demographic, academic, family, social, and lifestyle attributes.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Pytest-Tests-0A9EDC?logo=pytest&logoColor=white" alt="Pytest">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

---

## Project Overview

This project builds a reproducible machine learning pipeline to predict a student's **final grade (`G3`)** using information available about the student such as study habits, previous failures, absences, parental education, social activities, and other demographic/lifestyle attributes.

The project is deliberately designed as a **regression problem without using `G1` and `G2`**. This makes the experiment focused on estimating final performance from background and earlier-stage attributes rather than relying on intermediate grades that are strongly related to the final grade.

> **Portfolio focus:** This repository demonstrates the complete ML workflow — data preparation, exploratory analysis, preprocessing, baseline modeling, evaluation, cross-validation, hyperparameter tuning, feature analysis, reproducibility, and automated testing.

---

## Key Results

All metrics below are from the project's actual held-out test experiment.

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Ridge | 2.0261 | 2.7753 | 0.2102 |
| **Tuned Random Forest** | **1.9985** | **2.7557** | **0.2213** |
| Gradient Boosting | 2.0774 | 2.8035 | 0.1940 |

The tuned Random Forest produced the lowest measured MAE and RMSE among the tuned models in this experiment.

### Cross-Validation

The tuned Random Forest achieved a **5-fold cross-validation RMSE of 2.6476** during hyperparameter search.

These numbers are experiment-specific and should not be interpreted as a guarantee of performance on a different population or dataset split.

---

## Dataset

**UCI Machine Learning Repository — Student Performance**

- Dataset ID: `320`
- 649 student records
- 30 model input features
- Target: `G3`
- Two Portuguese schools
- No missing values in the downloaded dataset
- No duplicate records detected

### Target

`G3` represents the student's final grade.

### Why exclude `G1` and `G2`?

`G1` and `G2` are intermediate grades and are strongly correlated with `G3`. Including them would make the prediction task substantially different. This project therefore excludes both variables to study prediction from the remaining student attributes.

**Dataset:** https://archive.ics.uci.edu/dataset/320/student%2Bperformance

**Citation:** Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5TG7T

---

## Problem Definition

### Objective

Given a student's available demographic, academic, family, social, and lifestyle information:

> **Estimate the student's final grade (`G3`).**

### Machine Learning Type

**Supervised Learning → Regression**

Why regression instead of classification?

Because `G3` is a numerical grade and the project predicts its value rather than assigning the student to a discrete class.

---

## Machine Learning Pipeline

```text
                         UCI Student Dataset
                                  │
                                  ▼
                         Data Acquisition
                                  │
                                  ▼
                         Data Validation
                                  │
                                  ▼
                         Feature / Target Split
                                  │
                                  ▼
                           Train / Test Split
                              80% / 20%
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
          Numerical Features                 Categorical Features
                 │                                 │
        Median Imputation                  Most-Frequent Imputation
                 │                                 │
          StandardScaler                    One-Hot Encoding
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                         Model Training
                                  │
              ┌───────────┬───────┼────────┬────────────┐
              ▼           ▼       ▼        ▼            ▼
           Dummy       Linear    Ridge   Random Forest  Gradient
         Regressor   Regression          Regressor     Boosting
                                  │
                                  ▼
                         Model Evaluation
                    MAE • RMSE • R² • CV RMSE
                                  │
                                  ▼
                       Hyperparameter Tuning
                             GridSearchCV
                                  │
                                  ▼
                        Tuned Model Evaluation
                                  │
                                  ▼
                         Feature Importance
                                  │
                                  ▼
                      Reports + Saved Artifacts
```

---

## Features

### Numerical / Ordinal

`age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`

### Categorical

`school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`

### Target

`G3`

---

## Preprocessing Strategy

The project uses a scikit-learn `ColumnTransformer` containing separate preprocessing pipelines for numerical and categorical features.

### Numerical Pipeline

```text
Numerical Features
      ↓
Median Imputation
      ↓
StandardScaler
```

### Categorical Pipeline

```text
Categorical Features
      ↓
Most-Frequent Imputation
      ↓
OneHotEncoder(handle_unknown="ignore")
```

### Why use a Pipeline?

The preprocessing steps and model are combined into a single estimator. This is particularly important during cross-validation because transformations are fitted within the appropriate training folds instead of leaking validation information into the preprocessing stage.

### Transformation Result

- Training samples: `519`
- Test samples: `130`
- Original input features: `30`
- Transformed model features: `56`

---

## Models Evaluated

| Model | Purpose |
|---|---|
| Dummy Regressor | Establishes a simple baseline using the training mean |
| Linear Regression | Tests a linear relationship between inputs and target |
| Ridge Regression | Linear model with L2 regularization |
| Random Forest Regressor | Captures nonlinear relationships and feature interactions using an ensemble of trees |
| Gradient Boosting Regressor | Sequential tree-based ensemble that learns from previous prediction errors |

---

## Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between the predicted and actual grade.

**Lower is better.**

### RMSE — Root Mean Squared Error

Measures the square root of the mean squared prediction error. Larger errors receive greater penalty than with MAE.

**Lower is better.**

### R² — Coefficient of Determination

Measures how much target variance is explained relative to a mean-prediction baseline.

**Higher is better.**

---

## Baseline Experiment

Before tuning, five models were compared on the held-out test set:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Dummy Regressor | 2.3946 | 3.1726 | -0.0322 |
| Linear Regression | 2.1564 | 2.8618 | 0.1602 |
| Ridge | 2.1518 | 2.8580 | 0.1624 |
| Random Forest | 2.0505 | 2.8128 | 0.1887 |
| Gradient Boosting | 2.0744 | 2.7702 | 0.2130 |

The baseline experiment established a reference point before hyperparameter tuning.

---

## Cross-Validation

The project uses **5-fold cross-validation** on the training data.

```text
Training Data
─────────────────────────────────────
Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5
  V        T        T        T        T
  T        V        T        T        T
  T        T        V        T        T
  T        T        T        V        T
  T        T        T        T        V
```

This allows every training sample to participate in validation once while the model is trained on the remaining folds.

Baseline CV RMSE results:

| Model | Mean CV RMSE | Std. Dev. |
|---|---:|---:|
| Dummy Regressor | 3.2326 | 0.3914 |
| Linear Regression | 2.7897 | 0.3034 |
| Ridge | 2.7869 | 0.3034 |
| Random Forest | 2.7238 | 0.3066 |
| Gradient Boosting | 2.8876 | 0.2768 |

---

## Hyperparameter Tuning

`GridSearchCV` was used to systematically evaluate predefined parameter combinations using 5-fold cross-validation.

### Ridge

```text
alpha = [0.01, 0.1, 1, 10, 100]
```

### Random Forest

```text
n_estimators   = [200, 300]
max_depth      = [None, 5, 10]
min_samples_leaf = [1, 2, 4]
max_features   = [0.7, 1.0]
```

### Gradient Boosting

```text
n_estimators   = [100, 200]
learning_rate  = [0.03, 0.05, 0.1]
max_depth      = [2, 3]
min_samples_leaf = [2, 4]
```

### Selected Random Forest Configuration

The measured best Random Forest configuration during the search was:

```text
n_estimators      = 200
max_depth         = None
max_features      = 0.7
min_samples_leaf  = 4
```

Its best cross-validation RMSE during the search was approximately **2.6476**.

---

## Tuned Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 2.0261 | 2.7753 | 0.2102 |
| **Random Forest** | **1.9985** | **2.7557** | **0.2213** |
| Gradient Boosting | 2.0774 | 2.8035 | 0.1940 |

The tuned Random Forest had the strongest measured held-out metrics among these three tuned candidates in this experiment.

---

## Feature Importance

The tuned Random Forest's higher-importance features included:

| Feature | Importance |
|---|---:|
| `failures` | 0.2628 |
| `absences` | 0.0693 |
| `Fedu` | 0.0438 |
| `Dalc` | 0.0381 |
| `school` indicators | ~0.0376 |
| `health` | 0.0351 |
| `Walc` | 0.0335 |
| `Medu` | 0.0293 |
| `studytime` | 0.0283 |

> **Important:** Feature importance represents model reliance, not causality. A high importance value does not prove that changing that feature would cause a student's grade to change.

---

## Exploratory Data Analysis

The EDA stage investigates the structure of the data before and alongside modeling.

Analyses include:

- Final grade distribution
- Numeric feature correlations
- Study time vs final grade
- Absences vs final grade
- Categorical group comparisons
- Failure history analysis
- Model comparison visualizations
- Feature-importance visualization

Some observed correlations with `G3` were:

| Feature | Correlation with G3 |
|---|---:|
| `studytime` | 0.2498 |
| `Medu` | 0.2402 |
| `Fedu` | 0.2118 |
| `failures` | -0.3933 |
| `Dalc` | -0.2047 |
| `Walc` | -0.1766 |
| `absences` | -0.0914 |

Correlation and group averages describe associations in this dataset and should not be interpreted as causal relationships.

---

## Repository Structure

```text
student-performance-prediction/
│
├── data/
│   ├── raw/                 # Raw dataset (not committed)
│   └── processed/           # Processed data artifacts
│
├── models/                  # Generated model artifacts (ignored by Git)
│
├── notebooks/               # Optional exploratory notebooks
│
├── reports/
│   ├── figures/             # Generated visualizations
│   └── results/             # Generated CSV experiment results
│
├── src/
│   ├── analysis/            # Additional dataset analysis
│   ├── data/                # Dataset acquisition
│   ├── features/            # Feature engineering package
│   ├── models/              # Training, evaluation, tuning
│   ├── preprocessing/       # Data loading and preprocessing
│   └── visualization/       # EDA and model plots
│
├── tests/                   # Automated tests
│
├── .gitignore
├── requirements.txt
├── run_pipeline.py          # One-command pipeline runner
└── README.md
```

---

## Running Locally

### 1. Open the repository

```powershell
cd D:\AI-ML-Projects\student-performance-prediction
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Prepare the dataset

The raw dataset is intentionally not committed to Git. Use the project's data acquisition module to obtain it:

```powershell
python -m src.data.download_data
```

### 5. Run the complete pipeline

```powershell
python run_pipeline.py
```

### 6. Run tests

```powershell
pytest -v
```

---

## Pipeline Outputs

After execution, the project produces:

```text
models/
└── *.joblib

reports/
├── figures/
│   └── *.png
│
└── results/
    ├── evaluation_results.csv
    ├── cross_validation_results.csv
    ├── tuning_results.csv
    ├── tuned_evaluation_results.csv
    └── feature_importance.csv
```

Generated model binaries, report CSVs, figures, raw data, and the local virtual environment are excluded through `.gitignore`.

---

## Testing

The preprocessing test suite verifies:

- Dataset loads correctly
- Expected features exist
- Target values are within the expected range
- Dataset contains no missing values
- Feature/target separation has the expected shape

Latest local test run:

```text
5 passed
```

---

## Engineering Practices Demonstrated

- Reproducible train/test split using `random_state=42`
- Modular Python package structure
- Scikit-learn `Pipeline`
- Scikit-learn `ColumnTransformer`
- Separate preprocessing for numerical/categorical data
- Cross-validation
- Grid-search hyperparameter tuning
- Multiple model comparison
- Automated testing with Pytest
- Generated experiment reports
- Git/GitHub version control
- `.gitignore` management for generated and environment files

---

## Limitations

- The dataset represents students from two Portuguese schools and may not generalize to other populations.
- The available features do not capture every factor affecting student performance.
- The final metrics come from a single held-out test split.
- Feature importance should not be interpreted as causal evidence.
- Excluding `G1` and `G2` intentionally makes the task harder and changes the prediction setting.

---

## Future Improvements

- Compare additional regression algorithms
- Perform systematic feature selection
- Explore ensemble/stacking approaches
- Add prediction intervals
- Perform deeper error analysis across student groups
- Add model explainability with SHAP
- Build an interactive prediction interface
- Add GitHub Actions CI
- Add experiment tracking

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Joblib` · `Pytest` · `Jupyter`

---

## What This Project Demonstrates

This repository is designed as an ML engineering portfolio project rather than a single training script. The emphasis is on understanding the complete lifecycle of a supervised learning experiment:

```text
Problem Definition
      ↓
Data Understanding
      ↓
EDA
      ↓
Preprocessing
      ↓
Baseline Models
      ↓
Evaluation
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Final Evaluation
      ↓
Feature Analysis
      ↓
Testing & Reproducibility
```

---

## Author

**Praveen**

GitHub: https://github.com/Praveen703639

This project was built as part of an AI/ML portfolio focused on practical machine learning and software engineering fundamentals.
