import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------
# LOAD TRAINING DATA
# ---------------------------------------------------------

df = pd.read_csv("features/ml_training_data.csv")


# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------

X = df[
    [
        "station",
        "year",
        "station_age",
        "fuel_type"
    ]
]

y = df["annual_fuel_litres"]


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

categorical_features = [
    "station",
    "fuel_type"
]

numeric_features = [
    "year",
    "station_age"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ---------------------------------------------------------
# RIDGE REGRESSION MODEL
# ---------------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            Ridge(alpha=1.0)
        )
    ]
)


# ---------------------------------------------------------
# LEAVE-ONE-OUT EVALUATION
# ---------------------------------------------------------

loo = LeaveOneOut()

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=loo
)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(
    y,
    predictions
)

# ---------------------------------------------------------
# SIMPLE MEDIAN BENCHMARK
# ---------------------------------------------------------

benchmark_predictions = []

for i in range(len(y)):
    remaining_values = np.delete(y.to_numpy(), i)
    benchmark_predictions.append(
        np.median(remaining_values)
    )

benchmark_mae = mean_absolute_error(
    y,
    benchmark_predictions
)


# ---------------------------------------------------------
# ML RELIABILITY CHECK
# ---------------------------------------------------------

ml_is_better = mae < benchmark_mae

print("\nML MODEL EVALUATION")
print("=" * 50)

for actual, predicted in zip(y, predictions):
    print(
        f"Actual: {actual:,.0f} L"
        f" | Predicted: {predicted:,.0f} L"
    )

print("=" * 50)
print(f"Mean Absolute Error: {mae:,.0f} L")
print(f"Median Benchmark MAE: {benchmark_mae:,.0f} L")

if ml_is_better:
    print("ML STATUS: RELIABLE ENOUGH FOR PROTOTYPE USE")
else:
    print("ML STATUS: NOT RELIABLE - USE BASELINE INSTEAD")

print(
    "\nIMPORTANT:"
    "\nOnly 3 historical observations are available."
    "\nThis evaluation is therefore a prototype assessment,"
    "\nnot a statistically reliable accuracy estimate."
)


# ---------------------------------------------------------
# TRAIN FINAL MODEL ON ALL AVAILABLE DATA
# ---------------------------------------------------------

model.fit(X, y)


# ---------------------------------------------------------
# SAVE TRAINED MODEL
# ---------------------------------------------------------

import os
import joblib

os.makedirs("models", exist_ok=True)

model_file = "models/fuel_prediction_model.pkl"

joblib.dump(
    model,
    model_file
)
# ---------------------------------------------------------
# SAVE ML RELIABILITY STATUS
# ---------------------------------------------------------

import json

status_file = "models/fuel_model_status.json"

status = {
    "ml_reliable": bool(ml_is_better),
    "ml_mae_litres": float(mae),
    "benchmark_mae_litres": float(benchmark_mae),
    "training_observations": int(len(df)),
    "target": "annual_fuel_litres",
    "note": (
        "ML is disabled for operational predictions because "
        "it does not outperform the historical median benchmark."
        if not ml_is_better
        else
        "ML passed the prototype benchmark."
    )
}

with open(status_file, "w") as f:
    json.dump(status, f, indent=4)

print(f"ML status saved to: {status_file}")


print(f"\nTrained model saved to: {model_file}")