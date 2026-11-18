"""
ml_pipeline.py
--------------
All the machine-learning logic lives here, so that BOTH train_model.py (terminal)
and app.py (Streamlit website) use exactly the same code.

Steps:
  1. load_data()          -> read the CSV file
  2. build_preprocessor() -> scale numbers + convert text columns to numbers
  3. get_models()         -> the algorithms we compare
  4. train_and_evaluate() -> train, test, return error metrics (MAE, RMSE, R2)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent          # folder where this file lives
DATA_PATH = BASE_DIR / "data" / "crop_data.csv"
TARGET = "yield_tonnes_per_ha"

NUMERIC = ["rainfall_mm", "temperature_c", "humidity_percent", "soil_ph", "nitrogen",
           "phosphorus", "potassium", "fertilizer_kg_per_ha", "pesticide_kg_per_ha"]
CATEGORICAL = ["crop", "season", "irrigation"]


def load_data(path=DATA_PATH):
    """Reads the CSV. If the file is missing (e.g. 'data' folder not uploaded to GitHub),
    the same dataset is regenerated automatically so the app never crashes."""
    if Path(path).exists():
        return pd.read_csv(path)
    from generate_dataset import make_dataframe
    return make_dataframe()


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def train_and_evaluate(df):
    """Train all models on 80% of data, test on the remaining 20%."""
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results, fitted = {}, {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": r2_score(y_test, pred),
            "cv_r2": cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean(),
            "pred": pred,
        }
        fitted[name] = pipe

    best = max(results, key=lambda n: results[n]["r2"])
    return fitted, results, best, (X_train, X_test, y_train, y_test)


def feature_importance(pipe):
    """Which inputs matter most? (tree-based models only)"""
    names = [n.split("__")[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = pipe.named_steps["model"].feature_importances_
    return pd.Series(imp, index=names).sort_values(ascending=False)
