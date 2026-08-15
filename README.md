# Heart Failure Prediction

An end-to-end tabular machine learning project for predicting `DEATH_EVENT` among patients with heart failure.

The project focuses on building a reproducible XGBoost workflow and evaluating how preprocessing, class-imbalance strategies, threshold selection, feature engineering, and feature availability affect model performance.

> This project is intended for educational and portfolio purposes. It is not a clinically validated or deployable medical decision-support system.

## Project Objective

The objective is to classify whether a patient experienced a death event during the recorded follow-up period.

The main workflow uses XGBoost with:

- Pipeline-based preprocessing
- Five-fold stratified cross-validation
- Hyperparameter tuning with `GridSearchCV`
- `scale_pos_weight` evaluation
- Out-of-fold threshold optimization
- Test-set evaluation
- Feature-importance analysis
- A feature-availability audit for `time`

## Dataset

This project uses the [Heart Failure Clinical Records dataset](https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords) from the UCI Machine Learning Repository. The dataset is also described in the associated [research paper](https://doi.org/10.1186/s12911-020-1023-5).

Dataset summary:

- 299 observations
- 12 predictor features
- 203 survived cases
- 96 death-event cases
- Approximately 32.1% positive-class observations
- No missing values according to the official dataset documentation
- Target column: `DEATH_EVENT`

Feature groups:

| Group | Features |
|---|---|
| Demographic | `age`, `sex` |
| Clinical and laboratory | `ejection_fraction`, `serum_creatinine`, `serum_sodium`, `creatinine_phosphokinase`, `platelets` |
| Medical history | `anaemia`, `diabetes`, `high_blood_pressure`, `smoking` |
| Follow-up variable | `time` |

The `time` feature represents follow-up duration in days. It may not be available in a true baseline prediction scenario, so its contribution is evaluated separately from the other clinical features.

## Project Structure

```text
heart-failure-prediction/
├── data/
│   ├── data_description.txt
│   └── heart_failure_clinical_records_dataset.csv
├── models/
│   ├── metrics_test.csv
│   ├── metrics_test_smote.csv
│   ├── scale_pos_weight_results.csv
│   ├── time_feature_audit.csv
│   ├── model.joblib
│   ├── model_smote.joblib
│   ├── preprocessor.joblib
│   └── xgb_without_time_audit.joblib
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── feature_engineering.ipynb
│   ├── modeling.ipynb
│   └── modeling_smote.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

### Notebooks

- `eda.ipynb`: explores distributions, class balance, correlations, and potential outliers.
- `preprocessing.ipynb`: documents the preprocessing workflow and transformed data.
- `feature_engineering.ipynb`: explores engineered features such as eGFR and ejection-fraction groups.
- `modeling.ipynb`: contains the main model comparison, tuning, threshold optimization, final evaluation, and `time` feature audit.
- `modeling_smote.ipynb`: evaluates a SMOTE-based training pipeline.

### Model Artifacts

- `model.joblib`: main XGBoost pipeline using the `time` feature.
- `preprocessor.joblib`: preprocessing component from the main pipeline.
- `metrics_test.csv`: main model evaluation metrics.
- `model_smote.joblib`: model trained with the SMOTE workflow.
- `metrics_test_smote.csv`: SMOTE evaluation metrics.
- `scale_pos_weight_results.csv`: OOF results for candidate class weights.
- `time_feature_audit.csv`: comparison between XGBoost with and without `time`.
- `xgb_without_time_audit.joblib`: separate audit model without `time`.

## Methodology

1. Exploratory data analysis
2. Stratified train/test split
3. Preprocessing inside scikit-learn pipelines
4. Five-fold stratified cross-validation
5. Baseline model comparison
6. Hyperparameter tuning with `GridSearchCV`
7. `scale_pos_weight` experiment
8. Out-of-fold threshold optimization
9. Test-set evaluation
10. Feature-importance analysis and `time` audit
11. Model and metrics export

Numerical transformations are fitted inside pipelines. This keeps preprocessing parameters isolated to the training folds and avoids fitting transformations on the test set.

## Main Model Performance

The main model is XGBoost with the `time` feature and the selected `scale_pos_weight` of `2.0`.

| Metric | Value |
|---|---:|
| CV AUC | 0.922 |
| Test AUC | 0.886 |
| Optimized threshold | 0.47 |
| Test F1 | 0.743 |
| Test precision | 0.813 |
| Test recall | 0.684 |
| OOF recall | 0.805 |
| OOF F2 | 0.791 |
| Selected `scale_pos_weight` | 2.0 |

The optimized threshold was selected using out-of-fold predictions from the training data rather than directly from the test set. The test set was used only for final evaluation.

## Experiment Results

| Experiment | Result | Decision |
|---|---|---|
| Threshold optimization | Improved OOF recall targeting; test recall remained 0.684 | Integrated into `main` |
| `scale_pos_weight` | Selected value 2.0; only modest OOF improvement | Integrated into `main` |
| SMOTE | Test AUC 0.822, below the main model | Not selected |
| EGFR and EF-group features | Test AUC 0.866, below the main model | Not selected |
| XGBoost preprocessing | Log-only matched the old preprocessing; raw features performed worse | Not selected |
| `time` feature audit | With `time`: test AUC 0.886; without `time`: 0.788 | Integrated as an audit |

### `time` Feature Audit

| Variant | Test AUC | Optimized F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| XGBoost + `time` | 0.886 | 0.743 | 0.813 | 0.684 |
| XGBoost without `time` | 0.788 | 0.655 | 0.500 | 0.947 |

The model without `time` achieves much higher recall after threshold optimization, but produces substantially more false positives. It is therefore more appropriate for high-sensitivity screening than precise classification.

The model with `time` provides better overall discrimination. However, because `time` represents follow-up duration, the no-time model is a more realistic baseline scenario when follow-up information is unavailable at prediction time.

The threshold optimization, `scale_pos_weight`, SMOTE, and `time` feature audit are represented in `main`. The EGFR/EF-group and XGBoost preprocessing branches remain historical experiments and were not selected for the main workflow.

## Limitations and Responsible Use

- The dataset contains only 299 patients.
- The test set contains 60 observations and 19 positive cases.
- Metrics may have high variance because of the small sample size.
- Results come from one fixed stratified holdout split.
- No external validation dataset is available.
- The `time` feature may create a temporal availability issue for baseline prediction.
- The model has not been validated for clinical use.
- Threshold selection reflects a portfolio experiment, not a clinical decision policy.

## Reproducibility and Usage

The project uses Python 3.12 or newer and the dependencies declared in `pyproject.toml` and `uv.lock`.

Install the environment with:

```bash
uv sync
```

Start Jupyter with:

```bash
uv run jupyter notebook
```

Recommended notebook order:

```text
eda.ipynb
preprocessing.ipynb
feature_engineering.ipynb
modeling.ipynb
modeling_smote.ipynb
```

## Portfolio Highlights

This project demonstrates:

- An end-to-end tabular machine learning workflow
- Leakage-aware preprocessing pipelines
- Cross-validation and hyperparameter tuning
- Class-imbalance handling
- Out-of-fold threshold optimization
- Ablation studies across preprocessing and feature sets
- Feature availability and temporal-validity auditing
- Model interpretability through feature importance
- Reproducible model and metric exports
- Honest reporting of performance trade-offs

## Dataset Attribution

The dataset is provided by the UCI Machine Learning Repository under the dataset's stated license. Please refer to the [official dataset page](https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords) for attribution and licensing details.
