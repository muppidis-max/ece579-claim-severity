import pandas as pd

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