# ECE 579 Term Project — Claim Severity Prediction

Predicts insurance claim severity (dollar cost) on the Allstate Claims Severity dataset (Kaggle), comparing a Ridge regression baseline against a tuned XGBoost gradient-boosting model.

## Setup

1. Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

2. Install dependencies:

pip install -r requirements.txt

3. Download `train.csv` from the [Allstate Claims Severity Kaggle competition](https://www.kaggle.com/competitions/allstate-claims-severity/data) and place it at `data/train.csv`. (Not included in this repo per Kaggle's competition rules — see the data provenance note in the project report.)

## Running the pipeline


**Expected runtime:** ~17 minutes total on a normal laptop CPU. The XGBoost hyperparameter search is the single slowest step (~6 minutes); the remaining ~11 minutes is split across cross-validation, hyperparameter search for Ridge and the Decision Tree, the per-segment breakdown, and the error analysis, each of which re-fits the tuned models across 5 folds.

## What it does

1. Loads and splits the data (70% train / 15% validation / 15% test, fixed seed = 42).
2. One-hot encodes the 116 categorical features (fit on training data only).
3. Evaluates four models — mean predictor, Ridge, Decision Tree, XGBoost — on a single validation split, then again with 5-fold cross-validation (mean ± std).
4. Tunes Ridge’s alpha via grid search, the Decision Tree’s max_depth via grid search, and XGBoost’s max_depth/learning_rate/n_estimators/subsample via randomized search, all selected on   cross-validated RMSE.
5. Reports a claim-size segment (low/medium/high severity) breakdown and an error analysis on the worst-predicted claims.

See `pipeline_progress.md` for a running log of results and findings at each stage.

## Reproducing the headline results table

Running `python src/pipeline.py` top to bottom reproduces every number in the project report's Preliminary Results table (Section 6), printed to the console in order.