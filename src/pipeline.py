from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import time
script_start = time.time()

df = pd.read_csv("data/train.csv")
X = df.drop(columns=["id", "loss"])
y = df["loss"]

print(X.shape, y.shape)   
print(X.head())

#step2

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

#Step 3
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cat_cols = [c for c in X.columns if c.startswith("cat")]
cont_cols = [c for c in X.columns if c.startswith("cont")]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",
)

X_train_encoded = preprocessor.fit_transform(X_train)
print("Encoded shape:", X_train_encoded.shape)

print("\n--- Hyperparameter search: Ridge alpha ---")
ridge_pipe = Pipeline([("prep", preprocessor), ("model", Ridge(random_state=RANDOM_SEED))])
ridge_param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}

ridge_search = GridSearchCV(
    ridge_pipe, ridge_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
    n_jobs=-1,
)
ridge_search.fit(X_train, y_train)

print(f"Best alpha: {ridge_search.best_params_['model__alpha']}")
print(f"Best CV RMSE: {-ridge_search.best_score_:.2f}")

#Step 4
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(name, model):
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    print(f"{name:35s} RMSE: {rmse:10.2f}   MAE: {mae:10.2f}")
    return rmse, mae
#step 5
from sklearn.dummy import DummyRegressor

evaluate("Trivial reference (mean predictor)", DummyRegressor(strategy="mean"))
#step 6
from sklearn.linear_model import Ridge

evaluate("Baseline: Ridge regression", Ridge(alpha=1.0, random_state=42))
#step 7
import xgboost as xgb

evaluate(
    "Proposed: Gradient Boosting (XGBoost)",
    xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
)

def cv_evaluate(name, model, n_splits=5):
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_validate(
        pipe, X_train, y_train, cv=kf,
        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error"),
        n_jobs=-1,
    )
    rmse_scores = -scores["test_neg_root_mean_squared_error"]
    mae_scores = -scores["test_neg_mean_absolute_error"]
    print(f"{name:35s} RMSE: {rmse_scores.mean():8.2f} ± {rmse_scores.std():6.2f}   "
          f"MAE: {mae_scores.mean():8.2f} ± {mae_scores.std():6.2f}")
    return rmse_scores, mae_scores

print("\n--- Cross-validated (5-fold) results on training data ---")
cv_evaluate("Trivial reference (mean predictor)", DummyRegressor(strategy="mean"))
cv_evaluate("Baseline: Ridge regression", Ridge(alpha=1.0, random_state=RANDOM_SEED))
cv_evaluate("Proposed: Gradient Boosting (XGBoost)",
    xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1))

 

#step 9
print("\n--- Hyperparameter search: XGBoost ---")
xgb_pipe = Pipeline([("prep", preprocessor), ("model", xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1))])

xgb_param_dist = {
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__n_estimators": [100, 200, 300, 500],
    "model__subsample": [0.6, 0.8, 1.0],
}

xgb_search = RandomizedSearchCV(
    xgb_pipe, xgb_param_dist,
    n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
start = time.time()
xgb_search.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Search runtime: {elapsed/60:.1f} minutes")

print(f"Best params: {xgb_search.best_params_}")
print(f"Best CV RMSE (3-fold search): {-xgb_search.best_score_:.2f}")

#step 10
print("\n--- Confirming tuned XGBoost with 5-fold CV ---")
tuned_xgb = xgb.XGBRegressor(
    **{k.replace("model__", ""): v for k, v in xgb_search.best_params_.items()},
    random_state=RANDOM_SEED, n_jobs=-1,
)
cv_evaluate("Tuned XGBoost", tuned_xgb)

#step 17
print("\n--- Hyperparameter search: Decision Tree ---")
tree_pipe = Pipeline([("prep", preprocessor), ("model", DecisionTreeRegressor(random_state=RANDOM_SEED))])
tree_param_grid = {"model__max_depth": [5, 10, 15]}

tree_search = GridSearchCV(
    tree_pipe, tree_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
    n_jobs=-1,
)
tree_search.fit(X_train, y_train)

print(f"Best max_depth: {tree_search.best_params_['model__max_depth']}")
print(f"Best CV RMSE: {-tree_search.best_score_:.2f}")

#step 18
tuned_tree = DecisionTreeRegressor(
    max_depth=tree_search.best_params_["model__max_depth"],
    random_state=RANDOM_SEED,
)
cv_evaluate("Tuned Decision Tree", tuned_tree)

#step 11
def segment_evaluate(name, model, n_splits=5):
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    preds = cross_val_predict(pipe, X_train, y_train, cv=kf, n_jobs=-1)

    q1, q2 = y_train.quantile(0.33), y_train.quantile(0.66)
    buckets = pd.cut(y_train, bins=[-np.inf, q1, q2, np.inf], labels=["low", "medium", "high"])

    print(f"\n{name} — per-segment results (tertiles of claim severity):")
    for b in ["low", "medium", "high"]:
        mask = buckets == b
        rmse = np.sqrt(mean_squared_error(y_train[mask], preds[mask]))
        mae = mean_absolute_error(y_train[mask], preds[mask])
        print(f"  {b:8s} (n={mask.sum():6d})  RMSE: {rmse:10.2f}   MAE: {mae:10.2f}")

print("\n--- Per-segment (claim-size bucket) results ---")
segment_evaluate("Baseline: Ridge regression", Ridge(alpha=1.0, random_state=RANDOM_SEED))
segment_evaluate("Tuned XGBoost", tuned_xgb)
segment_evaluate("Tuned Decision Tree", tuned_tree)

#step 12
print("\n--- Error analysis: worst-predicted claims (Tuned XGBoost) ---")
tuned_xgb_preds = cross_val_predict(
    Pipeline([("prep", preprocessor), ("model", tuned_xgb)]),
    X_train, y_train,
    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
    n_jobs=-1,
)

errors = pd.DataFrame({
    "y_true": y_train.values,
    "y_pred": tuned_xgb_preds,
})
errors["signed_error"] = errors["y_true"] - errors["y_pred"]
errors["abs_error"] = errors["signed_error"].abs()

worst = errors.sort_values("abs_error", ascending=False).head(20)
print(worst[["y_true", "y_pred", "signed_error"]].to_string())

print(f"\nMean signed error, all {len(errors)} rows: {errors['signed_error'].mean():.2f}")
print(f"Of the 20 worst-predicted claims, {(worst['signed_error'] > 0).sum()} were under-predictions "
      f"(model guessed too low) vs. {(worst['signed_error'] < 0).sum()} over-predictions.")

#step 15
print("\n--- FINAL TEST-SET EVALUATION (touched once, at the very end) ---")

def final_test_evaluate(name, model):
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"{name:35s} RMSE: {rmse:10.2f}   MAE: {mae:10.2f}")
    return rmse, mae

final_test_evaluate("Trivial reference (mean predictor)", DummyRegressor(strategy="mean"))
final_test_evaluate("Baseline: Ridge regression (tuned)", Ridge(alpha=1.0, random_state=RANDOM_SEED))
final_test_evaluate("Proposed: Gradient Boosting (tuned XGBoost)", tuned_xgb)
final_test_evaluate("Tuned Decision Tree", tuned_tree)

print(f"\nTotal script runtime: {(time.time() - script_start)/60:.1f} minutes")

