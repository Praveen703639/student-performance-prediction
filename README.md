<!-- ═══════════════════════════════════════════════════════════════════════ -->
<!-- STUDENT PERFORMANCE PREDICTION — ML ENGINEERING SHOWCASE -->
<!-- ═══════════════════════════════════════════════════════════════════════ -->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&height=250&text=STUDENT%20PERFORMANCE&fontSize=46&fontColor=00F3FF&stroke=00F3FF&strokeWidth=2&desc=MACHINE%20LEARNING%20%7C%20REGRESSION%20%7C%20END-TO-END%20PIPELINE&descAlignY=68&descSize=15&theme=matrix" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=19&pause=900&color=00F3FF&center=true&vCenter=true&width=900&lines=End-to-End+Machine+Learning;Regression+%7C+Model+Comparison+%7C+Hyperparameter+Tuning;EDA+%7C+Cross-Validation+%7C+Feature+Analysis;Scikit-Learn+%7C+Pandas+%7C+Pytest;Built+as+a+Portfolio-Grade+ML+Engineering+Project" alt="Typing Animation"/>

<br>

[![Python](https://img.shields.io/badge/PYTHON-3.x-0E1128?style=for-the-badge&logo=python&logoColor=00F3FF)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/SCIKIT--LEARN-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/PANDAS-DATA%20ANALYSIS-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Pytest](https://img.shields.io/badge/PYTEST-TESTED-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)
[![Status](https://img.shields.io/badge/STATUS-COMPLETED-00F3FF?style=for-the-badge)](#)

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🧠 `PROJECT.BOOT()`

```python
class StudentPerformanceML:
    problem = "Predict final student grade (G3)"
    learning_type = "Supervised Learning / Regression"
    dataset = "UCI Student Performance"
    input_features = 30
    target = "G3"

    pipeline = [
        "Data Acquisition",
        "Validation & EDA",
        "Preprocessing",
        "Model Comparison",
        "Cross-Validation",
        "Hyperparameter Tuning",
        "Feature Analysis",
        "Automated Testing",
    ]
```

> **A reproducible machine learning pipeline for estimating final student performance from demographic, academic, family, social, and lifestyle attributes — intentionally excluding intermediate grades `G1` and `G2`.**

<div align="center">

### ⚡ `QUICK.FACTS`

| `649` | `30` | `56` | `5` | `5-FOLD` |
|:---:|:---:|:---:|:---:|:---:|
| Student Records | Input Features | Transformed Features | Models Compared | Cross-Validation |

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 📊 `RESULTS.DASHBOARD`

All metrics below come from the project's actual held-out test experiment.

<div align="center">

| MODEL | MAE ↓ | RMSE ↓ | R² ↑ |
|:---|---:|---:|---:|
| Ridge Regression | 2.0261 | 2.7753 | 0.2102 |
| **Tuned Random Forest** | **1.9985** | **2.7557** | **0.2213** |
| Gradient Boosting | 2.0774 | 2.8035 | 0.1940 |

<br>

**5-FOLD TUNED RANDOM FOREST CV RMSE: `2.6476`**

</div>

> These results are specific to this experiment and should not be treated as guaranteed performance on another population or dataset split.

### 🔬 Baseline → Tuned

```text
                    TEST RMSE                 TEST R²

Dummy               3.1726                    -0.0322
Linear Regression   2.8618                     0.1602
Ridge               2.8580                     0.1624
Random Forest       2.8128                     0.1887
Gradient Boosting   2.7702                     0.2130

                         ↓
                  HYPERPARAMETER
                     SEARCH
                         ↓

Tuned Ridge         2.7753                     0.2102
Tuned Random Forest 2.7557                     0.2213
Tuned Gradient      2.8035                     0.1940
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🧬 `ML.PIPELINE`

```text
                         ┌─────────────────────────┐
                         │   UCI STUDENT DATASET   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   DATA ACQUISITION      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ VALIDATION + EDA        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ FEATURE / TARGET SPLIT  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    TRAIN / TEST 80/20   │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌───────────────────┐               ┌───────────────────┐
          │ NUMERICAL FEATURES│               │ CATEGORICAL       │
          │                   │               │ FEATURES           │
          │ Median Imputation │               │ Most-Frequent      │
          │ StandardScaler    │               │ One-Hot Encoding   │
          └─────────┬─────────┘               └─────────┬─────────┘
                    └─────────────────┬─────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │   MODEL COMPARISON      │
                         │ Dummy • Linear • Ridge  │
                         │ RF • Gradient Boosting  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ 5-FOLD CROSS-VALIDATION │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    GRIDSEARCHCV         │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ TUNED MODEL EVALUATION  │
                         └────────────┬────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  Feature Importance       Reports / Artifacts
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🎓 `PROBLEM.DEFINITION`

### Objective

Given a student's available demographic, academic, family, social, and lifestyle information:

> **Estimate the student's final grade (`G3`).**

### Why Regression?

`G3` is a numerical final grade, so the project predicts a continuous value rather than assigning the student to a discrete class.

### Why exclude `G1` and `G2`?

`G1` and `G2` are intermediate grades and are strongly correlated with `G3`. Excluding them creates a prediction setting based on the remaining student attributes rather than directly relying on previous grades.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🗂️ `DATASET`

**UCI Machine Learning Repository — Student Performance**

| Property | Value |
|---|---|
| Dataset ID | `320` |
| Records | `649` |
| Model Inputs | `30` |
| Target | `G3` |
| Schools | Two Portuguese schools |
| Missing Values | None detected |
| Duplicate Records | None detected |

**Dataset:** [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student%2Bperformance)

**Citation:** Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🧩 `FEATURES`

### Numerical / Ordinal

```text
age • Medu • Fedu • traveltime • studytime • failures
famrel • freetime • goout • Dalc • Walc • health • absences
```

### Categorical

```text
school • sex • address • famsize • Pstatus • Mjob • Fjob
reason • guardian • schoolsup • famsup • paid • activities
nursery • higher • internet • romantic
```

### Target

```text
G3 → Final student grade
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## ⚙️ `PREPROCESSING`

The project uses a scikit-learn `ColumnTransformer` with separate pipelines for numerical and categorical data.

```text
NUMERICAL                         CATEGORICAL
───────────                       ───────────
      │                                 │
      ▼                                 ▼
Median Imputation              Most-Frequent Imputation
      │                                 │
      ▼                                 ▼
StandardScaler                   OneHotEncoder
                                      │
                                      ▼
                           handle_unknown = "ignore"
      │                                 │
      └──────────────┬──────────────────┘
                     ▼
               56 FEATURES
```

### Transformation

- Training samples: `519`
- Test samples: `130`
- Original input features: `30`
- Transformed features: `56`

### Why use a Pipeline?

Preprocessing and modeling are combined into a single estimator. During cross-validation, transformations are fitted inside the relevant training folds, helping prevent preprocessing leakage from validation data.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🤖 `MODELS`

| Model | Role in Experiment |
|---|---|
| Dummy Regressor | Simple mean-prediction baseline |
| Linear Regression | Tests a linear relationship |
| Ridge Regression | Linear model with L2 regularization |
| Random Forest | Nonlinear ensemble of decision trees |
| Gradient Boosting | Sequential tree-based ensemble |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 📐 `EVALUATION`

### MAE — Mean Absolute Error

Average absolute difference between predicted and actual grades. **Lower is better.**

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly. **Lower is better.**

### R² — Coefficient of Determination

Measures explained variance relative to a mean-prediction baseline. **Higher is better.**

### Cross-Validation

A 5-fold split is used on the training data:

```text
          F1     F2     F3     F4     F5
Fold 1    VAL    TR     TR     TR     TR
Fold 2    TR     VAL    TR     TR     TR
Fold 3    TR     TR     VAL    TR     TR
Fold 4    TR     TR     TR     VAL    TR
Fold 5    TR     TR     TR     TR     VAL
```

Baseline CV RMSE:

| Model | Mean CV RMSE | Std. Dev. |
|---|---:|---:|
| Dummy Regressor | 3.2326 | 0.3914 |
| Linear Regression | 2.7897 | 0.3034 |
| Ridge | 2.7869 | 0.3034 |
| Random Forest | 2.7238 | 0.3066 |
| Gradient Boosting | 2.8876 | 0.2768 |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🎛️ `HYPERPARAMETER.TUNING`

`GridSearchCV` systematically evaluates predefined parameter combinations using 5-fold cross-validation.

### Ridge

```text
alpha = [0.01, 0.1, 1, 10, 100]
```

### Random Forest

```text
n_estimators      = [200, 300]
max_depth         = [None, 5, 10]
min_samples_leaf  = [1, 2, 4]
max_features      = [0.7, 1.0]
```

### Gradient Boosting

```text
n_estimators      = [100, 200]
learning_rate     = [0.03, 0.05, 0.1]
max_depth         = [2, 3]
min_samples_leaf  = [2, 4]
```

### Selected Random Forest Configuration

```text
n_estimators      = 200
max_depth         = None
max_features      = 0.7
min_samples_leaf  = 4
```

Best measured CV RMSE during the search: **2.6476**.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🔎 `FEATURE.IMPORTANCE`

Top features from the tuned Random Forest:

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

> **Important:** Feature importance represents model reliance, not causality. A high importance value does not prove that changing a feature would cause a student's grade to change.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 📈 `EDA`

The exploratory analysis examines the dataset before and alongside modeling:

- Final grade distribution
- Numeric feature correlations
- Study time vs final grade
- Absences vs final grade
- Categorical group comparisons
- Failure history analysis
- Model comparison
- Feature importance

Selected numeric correlations with `G3`:

| Feature | Correlation |
|---|---:|
| `studytime` | 0.2498 |
| `Medu` | 0.2402 |
| `Fedu` | 0.2118 |
| `failures` | -0.3933 |
| `Dalc` | -0.2047 |
| `Walc` | -0.1766 |
| `absences` | -0.0914 |

Correlations and group averages describe associations in this dataset; they are not causal evidence.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🧪 `ENGINEERING.PRACTICES`

```text
✓ Modular Python package structure
✓ Reproducible train/test split (random_state=42)
✓ ColumnTransformer + Pipeline
✓ Numerical + categorical preprocessing
✓ Multiple model comparison
✓ 5-fold cross-validation
✓ Grid-search hyperparameter tuning
✓ Held-out test evaluation
✓ Feature importance analysis
✓ Automated tests with Pytest
✓ Generated experiment reports
✓ Git / GitHub version control
✓ Clean .gitignore for generated artifacts
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 📁 `CODEBASE.TOUR`

```text
student-performance-prediction/
│
├── data/
│   ├── raw/                 → Raw dataset (not committed)
│   └── processed/           → Processed data artifacts
│
├── models/                  → Generated model artifacts
│
├── notebooks/               → Optional exploratory notebooks
│
├── reports/
│   ├── figures/             → Generated visualizations
│   └── results/             → Experiment CSV results
│
├── src/
│   ├── analysis/            → Dataset analysis modules
│   ├── data/                → Dataset acquisition
│   ├── features/            → Feature engineering package
│   ├── models/              → Training / evaluation / tuning
│   ├── preprocessing/       → Loading / splitting / preprocessing
│   └── visualization/       → EDA / model visualizations
│
├── tests/                   → Automated tests
│
├── requirements.txt
├── run_pipeline.py          → One-command pipeline runner
├── .gitignore
└── README.md
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🚀 `RUN.LOCALLY`

### 1. Clone

```powershell
git clone https://github.com/Praveen703639/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Create / activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Download / prepare dataset

```powershell
python -m src.data.download_data
```

### 5. Run everything

```powershell
python run_pipeline.py
```

### 6. Run tests

```powershell
pytest -v
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 📦 `OUTPUTS`

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

Generated binaries, figures, report CSVs, raw data, caches, and the local virtual environment are excluded through `.gitignore`.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🧪 `TESTING`

The preprocessing test suite verifies:

- Dataset loads correctly
- Expected features exist
- Target values are within the expected range
- Dataset contains no missing values
- Feature / target separation has the expected shape

Latest local run:

```text
5 passed
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## ⚠️ `LIMITATIONS`

- The dataset represents students from two Portuguese schools and may not generalize to other populations.
- Available features do not capture every factor affecting student performance.
- Final metrics come from a single held-out test split.
- Feature importance is not causal evidence.
- Excluding `G1` and `G2` intentionally changes the prediction setting and makes the task harder.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🔮 `FUTURE.IMPROVEMENTS`

- Compare additional regression algorithms
- Try alternative feature engineering strategies
- Evaluate additional cross-validation strategies
- Add experiment tracking
- Add CI-based automated testing
- Add a lightweight prediction API / interface
- Add model interpretability with SHAP or similar methods

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🛠️ `TECH.ARSENAL`

<div align="center">

<img src="https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PANDAS-150458?style=for-the-badge&logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/SCIKIT--LEARN-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/MATPLOTLIB-11557C?style=for-the-badge" />
<img src="https://img.shields.io/badge/SEABORN-4C72B0?style=for-the-badge" />
<img src="https://img.shields.io/badge/PYTEST-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" />
<img src="https://img.shields.io/badge/GIT-F05032?style=for-the-badge&logo=git&logoColor=white" />

<br><br>

`Machine Learning` · `Regression` · `EDA` · `Preprocessing` · `Cross-Validation` · `Hyperparameter Tuning` · `Feature Analysis`

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 🎯 `WHAT.THIS.PROJECT.DEMONSTRATES`

```text
DATA                    MODELING                 ENGINEERING
────                    ────────                 ───────────
UCI Dataset             Regression               Modular Structure
Data Validation         Model Comparison         Reproducibility
EDA                     Cross-Validation         Automated Tests
Preprocessing           GridSearchCV             Git / GitHub
Feature Analysis        Evaluation               Experiment Reports
```

This repository is intended to demonstrate not only a trained model, but the **complete workflow used to build, evaluate, tune, test, and document a machine learning system.**

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

## 👨‍💻 `AUTHOR`

<div align="center">

**Praveen703639**

AI / ML • Software Engineering • Unreal Engine C++

[![GitHub](https://img.shields.io/badge/GITHUB-Praveen703639-0A0A0A?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Praveen703639)

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<div align="center">

### `BUILD → MEASURE → ANALYZE → IMPROVE`

**Machine Learning Portfolio Project • 2026**

</div>
