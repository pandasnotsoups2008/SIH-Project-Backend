import pandas as pd
import os
import json
import joblib


# ============================================================
# ANTARCTICA DIGITAL TWIN
# RESOURCE PREDICTION ENGINE
# ============================================================

print("=" * 70)
print("              ANTARCTICA DIGITAL TWIN")
print("                 PREDICTION ENGINE")
print("=" * 70)


# ------------------------------------------------------------
# 1. CONFIGURABLE PLANNING ASSUMPTIONS
# ------------------------------------------------------------
# These are planning assumptions, NOT observed Antarctic data.
# They can later be replaced/optimized using ML.

FOOD_PER_PERSON_PER_DAY = 2.5       # kg
WATER_PER_PERSON_PER_DAY = 4.0      # litres

# Base fuel requirement per person per day.
BASE_FUEL_PER_PERSON_DAY = 8.0      # litres

# Environmental multiplier.
# Winter generally requires more energy/fuel for station operations.
SEASON_MULTIPLIERS = {
    "summer": 1.00,
    "winter": 1.20,
    "transition": 1.10
}

ML_STATUS_FILE = "models/fuel_model_status.json"
ML_MODEL_FILE = "models/fuel_prediction_model.pkl"
ml_model = None
ml_reliable = False

if os.path.exists(ML_STATUS_FILE):
    with open(ML_STATUS_FILE, "r") as f:
        ml_status = json.load(f)

    ml_reliable = ml_status.get("ml_reliable", False)

if ml_reliable and os.path.exists(ML_MODEL_FILE):
    ml_model = joblib.load(ML_MODEL_FILE)


# ------------------------------------------------------------
# 2. PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_resources(
    station,
    researchers,
    support_staff,
    duration_days,
    season
):

    total_personnel = researchers + support_staff

    season_key = season.lower()

    multiplier = SEASON_MULTIPLIERS.get(
        season_key,
        1.00
    )


    # --------------------------------------------------------
    # FOOD
    # --------------------------------------------------------

    food_required = (
        total_personnel
        * duration_days
        * FOOD_PER_PERSON_PER_DAY
    )


    # --------------------------------------------------------
    # WATER
    # --------------------------------------------------------

    water_required = (
        total_personnel
        * duration_days
        * WATER_PER_PERSON_PER_DAY
    )


    # --------------------------------------------------------
    # FUEL
    # ---------------------------------------------------------
    # FUEL PREDICTION
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # ML FUEL MODEL RELIABILITY CHECK
    # ---------------------------------------------------------

    ml_reliable = False

    if os.path.exists(ML_STATUS_FILE):
        with open(ML_STATUS_FILE, "r") as f:
            ml_status = json.load(f)

        ml_reliable = ml_status.get("ml_reliable", False)
        ml_model = None

if ml_reliable and os.path.exists(ML_MODEL_FILE):
    ml_model = joblib.load(ML_MODEL_FILE)
    # ---------------------------------------------------------
    # ML STATUS MESSAGE
    # ---------------------------------------------------------

    if ml_reliable:
        ml_message = "ML MODEL: ACTIVE"
    else:
        ml_message = "ML MODEL: DISABLED - INSUFFICIENT RELIABLE DATA"
    fuel_required = (
        total_personnel
        * duration_days
        * BASE_FUEL_PER_PERSON_DAY
        * multiplier
    )

    # Use ML only when the model has been validated as reliable.
    # Otherwise, keep the scientifically safer baseline.
    if ml_reliable and ml_model is not None:
        try:
            ml_input = pd.DataFrame([{
                "station": station,
                "fuel_type": "Fuel",
                "year": 2026,
                "station_age": 2026 - 2012
            }])

            ml_prediction = ml_model.predict(ml_input)[0]

            # ML model predicts annual fuel.
            # Convert to the requested expedition duration.
            fuel_required = ml_prediction * (duration_days / 365)

        except Exception:
            # Never allow an ML failure to break the Digital Twin.
            fuel_required = (
                total_personnel
                * duration_days
                * BASE_FUEL_PER_PERSON_DAY
                * multiplier
            )

  
    # --------------------------------------------------------
    # ENERGY
    # --------------------------------------------------------
    # Estimated daily electricity demand.
    # This is a planning assumption for the prototype.

    ENERGY_PER_PERSON_DAY = 25       # kWh

    energy_required = (
        total_personnel
        * duration_days
        * ENERGY_PER_PERSON_DAY
        * multiplier
    )


    # --------------------------------------------------------
    # SAFETY BUFFER
    # --------------------------------------------------------
    # Antarctic logistics benefit from reserve capacity.

    SAFETY_BUFFER = 0.10

    food_with_buffer = food_required * (1 + SAFETY_BUFFER)
    water_with_buffer = water_required * (1 + SAFETY_BUFFER)
    fuel_with_buffer = fuel_required * (1 + SAFETY_BUFFER)
    energy_with_buffer = energy_required * (1 + SAFETY_BUFFER)


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "station": station,
        "researchers": researchers,
        "support_staff": support_staff,
        "total_personnel": total_personnel,
        "duration_days": duration_days,
        "season": season,

        "food_kg": round(food_with_buffer, 2),
        "water_litres": round(water_with_buffer, 2),
        "fuel_litres": round(fuel_with_buffer, 2),
        "energy_kwh": round(energy_with_buffer, 2),
        "ml_message": ml_message
    }

    return result


# ------------------------------------------------------------
# 3. TEST THE DIGITAL TWIN
# ------------------------------------------------------------

if __name__ == "__main__":

    prediction = predict_resources(
        station="Bharati",
        researchers=30,
        support_staff=15,
        duration_days=180,
        season="Winter"
    )


    print("\n")
    print("=" * 70)
    print("             SIMULATION INPUT")
    print("=" * 70)

    print(f"Station:          {prediction['station']}")
    print(f"Researchers:      {prediction['researchers']}")
    print(f"Support staff:    {prediction['support_staff']}")
    print(f"Total personnel:  {prediction['total_personnel']}")
    print(f"Duration:         {prediction['duration_days']} days")
    print(f"Season:           {prediction['season']}")


    print("\n")
    print("=" * 70)
    print("             RESOURCE PREDICTION")
    print("=" * 70)
    print(f"ML Status: {prediction['ml_message']}")

    print(
        f"Food:             "
        f"{prediction['food_kg']:,.2f} kg"
    )

    print(
        f"Water:            "
        f"{prediction['water_litres']:,.2f} litres"
    )

    print(
        f"Fuel:             "
        f"{prediction['fuel_litres']:,.2f} litres"
    )

    print(
        f"Energy:           "
        f"{prediction['energy_kwh']:,.2f} kWh"
    )

    print("\n")
    print("=" * 70)
    print("              SIMULATION COMPLETE")
    print("=" * 70)