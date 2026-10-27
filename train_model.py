
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
TARGET = "posted_rate"
ID = "load_id"

train = pd.read_csv(ROOT / "train-test.csv")
validation = pd.read_csv(ROOT / "validation.csv")
template = pd.read_csv(ROOT / "validation-predictions-template.csv")
december = pd.read_csv(ROOT / "december-chart-inputs.csv")

def features(df):
    x = df.drop(columns=[TARGET, ID], errors="ignore").copy()
    if "date" in x.columns:
        d = pd.to_datetime(x["date"], errors="coerce")
        x["year"] = d.dt.year
        x["month"] = d.dt.month
        x["day"] = d.dt.day
        x["day_of_week"] = d.dt.dayofweek
        x["day_of_year"] = d.dt.dayofyear
        x = x.drop(columns=["date"])
    return x

X = features(train)
y = pd.to_numeric(train[TARGET], errors="raise")

# Add missing feature columns and keep the training column order.
def align(x, columns):
    x = x.copy()
    for c in columns:
        if c not in x.columns:
            x[c] = np.nan
    return x[columns]

Xv = align(features(validation), X.columns)
Xd = align(features(december), X.columns)

cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num = [c for c in X.columns if c not in cat]

prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

model = Pipeline([
    ("prep", prep),
    ("model", ExtraTreesRegressor(
        n_estimators=350, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ))
])

model.fit(X, y)

pred = np.maximum(model.predict(Xv), 0.01)
pred_map = pd.DataFrame({
    ID: validation[ID].astype(str),
    "predicted_rate": pred
})

out = template[[ID]].copy()
out[ID] = out[ID].astype(str)
out = out.merge(pred_map, on=ID, how="left", validate="one_to_one")
out = out[["load_id", "predicted_rate"]]

if len(out) != 12000 or out["predicted_rate"].isna().any():
    raise ValueError("Missing predictions or incorrect row count")

out.to_csv(ROOT / "validation_predictions.csv", index=False)

december["predicted_rate"] = np.maximum(model.predict(Xd), 0.01)
december.to_csv(ROOT / "december_predictions.csv", index=False)

print("Created validation_predictions.csv")
print("Created december_predictions.csv")
