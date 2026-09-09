import joblib
import pandas as pd
import rasterio


MODEL_FILE = "data/landslide_model.joblib"
RAINFALL_FILE = "data/ner_rainfall_20260907.tif"


print("=" * 60)
print("          NER LANDSLIDE RISK PREDICTION")
print("=" * 60)


# Load AI model
model = joblib.load(MODEL_FILE)

# Open rainfall raster
src = rasterio.open(RAINFALL_FILE)


# Get location from user
print("\nEnter the location you want to check.")

latitude = float(input("Latitude: "))
longitude = float(input("Longitude: "))


# Convert latitude/longitude to raster row and column
row, col = src.index(longitude, latitude)


# Check whether location is inside the rainfall raster
if (
    row < 0
    or row >= src.height
    or col < 0
    or col >= src.width
):
    print("\nLocation is outside the rainfall data area.")
    src.close()
    exit()


# Read a 5 x 5 area around the location
half_size = 2

row_start = max(0, row - half_size)
row_end = min(src.height, row + half_size + 1)

col_start = max(0, col - half_size)
col_end = min(src.width, col + half_size + 1)


window = rasterio.windows.Window(
    col_start,
    row_start,
    col_end - col_start,
    row_end - row_start
)


rainfall_data = src.read(1, window=window)


# Remove invalid values
valid_rainfall = rainfall_data[
    rainfall_data >= 0
]


if len(valid_rainfall) == 0:
    print("\nNo valid rainfall data found at this location.")
    src.close()
    exit()


# NASA IMERG GIS rainfall is stored in 0.1 mm units
valid_rainfall = valid_rainfall / 10.0


# Calculate rainfall statistics
mean_rainfall = float(valid_rainfall.mean())
max_rainfall = float(valid_rainfall.max())
min_rainfall = float(valid_rainfall.min())


# Create AI input
input_data = pd.DataFrame({

    "rainfall_mean_mm": [
        mean_rainfall
    ],

    "rainfall_max_mm": [
        max_rainfall
    ],

    "rainfall_min_mm": [
        min_rainfall
    ]
})


# Make prediction
prediction = model.predict(input_data)[0]


# Convert prediction number to risk level
risk_levels = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}


risk = risk_levels[int(prediction)]


# Get model probability
probabilities = model.predict_proba(input_data)[0]

confidence = max(probabilities) * 100


# Display result
print("\n")
print("=" * 60)
print("                 PREDICTION RESULT")
print("=" * 60)

print("\nLocation:")
print("Latitude :", latitude)
print("Longitude:", longitude)

print("\nRainfall around location:")
print("Average :", round(mean_rainfall, 2), "mm")
print("Maximum :", round(max_rainfall, 2), "mm")
print("Minimum :", round(min_rainfall, 2), "mm")

print("\nLandslide Risk:", risk)

print(
    "Model Confidence:",
    round(confidence, 2),
    "%"
)

print("\n")
print("=" * 60)


src.close()