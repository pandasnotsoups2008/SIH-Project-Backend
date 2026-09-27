import pandas as pd

datasets = {
    "Energy": "energy.csv",
    "Equipment": "equipments.csv",
    "Expedition": "expedition.csv",
    "Food": "Food.csv",
    "Fuel": "fuel.csv",
    "Logistics": "Logistics.csv",
    "Station": "Project_Station.csv",
    "Water": "water.csv",
    "Weather": "Weather.csv"
}

print("=" * 70)
print("          ANTARCTICA DIGITAL TWIN")
print("             DATA INSPECTION")
print("=" * 70)

for name, filename in datasets.items():

    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)

    try:
        df = pd.read_csv(filename)

        print("\nCOLUMNS:")
        for i, column in enumerate(df.columns, 1):
            print(f"{i}. {column}")

        print("\nFIRST 5 ROWS:")
        print(df.head().to_string(index=False))

        print("\nDATA TYPES:")
        print(df.dtypes.to_string())

    except Exception as e:
        print(f"ERROR loading {filename}: {e}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)