import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/ner_rainfall_features.csv"
MODEL_FILE = "data/landslide_model.joblib"
INFO_FILE = "data/model_info.json"


# Only these will be used by the AI model
FEATURES = [
    "rainfall_mean_mm",
    "rainfall_max_mm",
    "rainfall_min_mm"
]


print("=" * 65)
print("          NER LANDSLIDE AI - MODEL TRAINING")
print("=" * 65)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading training data...")

data = pd.read_csv(INPUT_FILE)

print("Training samples:", len(data))


# ============================================================
# CHECK FEATURES
# ============================================================

for feature in FEATURES:

    if feature not in data.columns:

        print("\nERROR: Missing feature:")
        print(feature)

        print("\nAvailable columns:")
        print(data.columns.tolist())

        raise SystemExit


# ============================================================
# CREATE BASELINE RISK LABEL
# ============================================================

print("\nCreating baseline risk labels...")

def calculate_risk(rainfall):

    if rainfall < 50:
        return 0

    elif rainfall < 100:
        return 1

    else:
        return 2


data["risk"] = data["rainfall_mean_mm"].apply(
    calculate_risk
)


print("\nRisk distribution:")

print(
    data["risk"]
    .value_counts()
    .sort_index()
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = data[FEATURES]

y = data["risk"]


print("\nFeatures used by AI:")

for feature in FEATURES:
    print(" -", feature)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


print("Model training completed!")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "model": "Random Forest",

    "features": FEATURES,

    "risk_classes": {
        "0": "LOW",
        "1": "MEDIUM",
        "2": "HIGH"
    },

    "warning": (
        "This is a baseline model. "
        "Risk labels are currently derived from rainfall thresholds. "
        "Real landslide observations will be added later."
    )
}


with open(
    INFO_FILE,
    "w"
) as file:

    json.dump(
        model_info,
        file,
        indent=4
    )


# ============================================================
# FINISH
# ============================================================

print("\n")
print("=" * 65)
print("          MODEL CREATED SUCCESSFULLY!")
print("=" * 65)

print("\nModel saved:")
print(MODEL_FILE)

print("\nModel information saved:")
print(INFO_FILE)

print("\nAI features:")

print(FEATURES)

print("\nIMPORTANT:")
print("This is still a BASELINE model.")
print("Later we will add real landslide locations")
print("and terrain/environmental features.")

print("\n")
print("=" * 65)