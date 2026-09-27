import pandas as pd
import os


# ---------------------------------------------------------
# LOAD CLEANED DATA
# ---------------------------------------------------------

fuel = pd.read_csv("processed_data/fuel_clean.csv")
station = pd.read_csv("processed_data/station_clean.csv")


# ---------------------------------------------------------
# KEEP ONLY OBSERVED FUEL SUPPLY/TRANSFER RECORDS
# ---------------------------------------------------------

fuel = fuel[
    (fuel["data_status"].str.upper() == "OBSERVED")
    & (fuel["quantity_litres"].notna())
].copy()


# ---------------------------------------------------------
# REMOVE STORAGE-CAPACITY RECORDS
# ---------------------------------------------------------

fuel = fuel[
    ~fuel["notes"].fillna("").str.contains(
        "capacity",
        case=False,
        na=False
    )
].copy()


# ---------------------------------------------------------
# GET STATION COMMISSIONING YEARS
# ---------------------------------------------------------

commissioned = station[
    station["parameter"].str.lower() == "commissioned"
].copy()

commissioned["commissioned_year"] = pd.to_numeric(
    commissioned["value"],
    errors="coerce"
)

commissioned = commissioned[
    ["station_id", "commissioned_year"]
]


# ---------------------------------------------------------
# MERGE STATION INFORMATION
# ---------------------------------------------------------

fuel = fuel.merge(
    commissioned,
    on="station_id",
    how="left"
)


# ---------------------------------------------------------
# CREATE FEATURES
# ---------------------------------------------------------

fuel["year"] = pd.to_numeric(
    fuel["year"],
    errors="coerce"
)

fuel["station_age"] = (
    fuel["year"] - fuel["commissioned_year"]
)


# ---------------------------------------------------------
# CREATE FINAL ML DATASET
# ---------------------------------------------------------

training_data = fuel[
    [
        "station_id",
        "station",
        "year",
        "station_age",
        "fuel_type",
        "quantity_litres"
    ]
].copy()


training_data = training_data.rename(
    columns={
        "quantity_litres": "annual_fuel_litres"
    }
)


# ---------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------

os.makedirs("features", exist_ok=True)

output_file = "features/ml_training_data.csv"

training_data.to_csv(
    output_file,
    index=False
)


print("\nML TRAINING DATA CREATED")
print("=" * 50)
print(training_data.to_string(index=False))
print("\nSaved to:", output_file)
print("Rows:", len(training_data))
