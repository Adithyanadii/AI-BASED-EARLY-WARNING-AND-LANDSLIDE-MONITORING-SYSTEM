import rasterio
import joblib
import pandas as pd


# Files
RAINFALL_FILE = "data/ner_rainfall_20260907.tif"
MODEL_FILE = "data/landslide_model.joblib"


print("=" * 60)
print("       NER LOCATION-BASED LANDSLIDE RISK")
print("=" * 60)


# Load model
model = joblib.load(MODEL_FILE)


# Ask for location
latitude = float(input("\nEnter latitude: "))
longitude = float(input("Enter longitude: "))


# Open rainfall map
with rasterio.open(RAINFALL_FILE) as src:

    # Convert latitude/longitude to raster row and column
    row, col = src.index(
        longitude,
        latitude
    )

    # Check location
    if (
        row < 0
        or row >= src.height
        or col < 0
        or col >= src.width
    ):
        print("\nERROR: Location is outside the rainfall map.")
        raise SystemExit

    # Read rainfall value
    rainfall_raw = src.read(
        1,
        window=rasterio.windows.Window(
            col,
            row,
            1,
            1
        )
    )[0, 0]


# Convert NASA stored value to mm
rainfall = float(rainfall_raw) / 10.0


# Check invalid value
if rainfall < 0:
    print("\nERROR: No valid rainfall data at this location.")
    raise SystemExit


# Create model input
input_data = pd.DataFrame({
    "rainfall_mean_mm": [rainfall],
    "rainfall_max_mm": [rainfall],
    "rainfall_min_mm": [rainfall]
})


# AI prediction
prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]


risk_levels = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


risk = risk_levels[int(prediction)]

confidence = max(probabilities) * 100


# Display result
print("\n")
print("=" * 60)
print("                 RESULT")
print("=" * 60)

print("\nLocation:")
print("Latitude :", latitude)
print("Longitude:", longitude)

print("\nRainfall:")
print(round(rainfall, 2), "mm")

print("\nLandslide Risk:")
print(risk)

print("\nModel confidence:")
print(round(confidence, 2), "%")

print("\n")
print("=" * 60)