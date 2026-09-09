import os
import glob
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import geometry_mask


# ============================================================
# SETTINGS
# ============================================================

RAINFALL_FILE = "data/ner_rainfall_20260907.tif"
OUTPUT_FILE = "data/ner_rainfall_features.csv"

SAMPLES_PER_STATE = 250

# Current 7 NER states
NER_STATES = [
    "Assam",
    "Meghalaya",
    "Manipur",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura"
]


# ============================================================
# START
# ============================================================

print("=" * 65)
print("       NER LANDSLIDE AI - FEATURE PREPARATION")
print("=" * 65)


# ============================================================
# FIND STATE BOUNDARY FILE
# ============================================================

print("\nSearching for state boundary file...")

possible_files = [
    "data/state_NWIC.GeoJSON",
    "data/state_NWIC_GeoJSON.geojson",
    "data/state_NWIC.GeoJSON.geojson",
    "data/state_NWIC_GeoJSON.json"
]

boundary_file = None

for file in possible_files:
    if os.path.exists(file):
        boundary_file = file
        break


if boundary_file is None:

    candidates = glob.glob("data/*.geojson")
    candidates += glob.glob("data/*.GeoJSON")
    candidates += glob.glob("data/*.json")

    for file in candidates:

        name = os.path.basename(file).lower()

        if "state" in name and "nwic" in name:
            boundary_file = file
            break


if boundary_file is None:

    print("\nERROR: State boundary GeoJSON not found.")

    print("\nFiles inside data folder:")

    for file in os.listdir("data"):
        print(" -", file)

    raise SystemExit


print("Boundary file found:")
print(boundary_file)


# ============================================================
# LOAD STATE BOUNDARIES
# ============================================================

print("\nLoading state boundaries...")

states = gpd.read_file(boundary_file)

print("Boundary file loaded successfully!")


# ============================================================
# FIND STATE NAME COLUMN
# ============================================================

print("\nFinding state name column...")

state_column = None

# IMPORTANT:
# state_name must come BEFORE state
# because the NWIC file contains both columns.

for column in [
    "state_name",
    "state",
    "STATE",
    "ST_NAME",
    "name"
]:

    if column in states.columns:

        state_column = column
        break


if state_column is None:

    print("ERROR: Could not find state name column.")

    print("Columns available:")

    print(states.columns.tolist())

    raise SystemExit


print("Using state name column:", state_column)


# Clean names
states[state_column] = (
    states[state_column]
    .astype(str)
    .str.strip()
)


# ============================================================
# SELECT 7 NER STATES
# ============================================================

selected_states = states[
    states[state_column].isin(NER_STATES)
].copy()


print("\nNER states found:", len(selected_states))

print(
    selected_states[state_column].tolist()
)


if len(selected_states) != 7:

    print("\nWARNING!")

    print(
        "Expected 7 states but found",
        len(selected_states)
    )

    print("Found:")

    print(
        selected_states[state_column].tolist()
    )

    print("\nAvailable matching state names:")

    for name in states[state_column].unique():

        if any(
            x.lower() in str(name).lower()
            for x in [
                "assam",
                "meghalaya",
                "manipur",
                "mizoram",
                "nagaland",
                "sikkim",
                "tripura"
            ]
        ):

            print(" -", name)


# ============================================================
# LOAD NASA RAINFALL
# ============================================================

print("\nLoading NASA rainfall raster...")


with rasterio.open(RAINFALL_FILE) as src:

    rainfall_raw = src.read(1)

    raster_transform = src.transform

    raster_crs = src.crs

    raster_height = src.height

    raster_width = src.width

    raster_nodata = src.nodata


print(
    "NASA rainfall raster opened successfully!"
)

print(
    "Raster size:",
    raster_width,
    "x",
    raster_height
)

print(
    "NASA CRS:",
    raster_crs
)


# ============================================================
# CONVERT NASA RAINFALL TO MILLIMETERS
# ============================================================

# NASA Late IMERG GIS rainfall values are stored
# in 0.1 mm units.
#
# Therefore:
# raw value / 10 = rainfall in mm

rainfall = (
    rainfall_raw.astype(np.float32)
    / 10.0
)


# Handle NoData

if raster_nodata is not None:

    rainfall[
        rainfall_raw == raster_nodata
    ] = np.nan


# Remove invalid values

rainfall[
    ~np.isfinite(rainfall)
] = np.nan


rainfall[
    rainfall < 0
] = np.nan


print("\nRainfall data prepared.")


# ============================================================
# MATCH CRS
# ============================================================

if selected_states.crs != raster_crs:

    print("\nReprojecting state boundaries...")

    selected_states = (
        selected_states.to_crs(raster_crs)
    )


print("Boundary CRS ready.")


# ============================================================
# GENERATE TRAINING SAMPLES
# ============================================================

all_samples = []


print("\n")

print("=" * 65)
print("GENERATING TRAINING SAMPLES")
print("=" * 65)


# Use one random generator for reproducible sampling
rng = np.random.default_rng(42)


for index, state_row in selected_states.iterrows():

    state_name = str(
        state_row[state_column]
    )

    geometry = state_row.geometry


    print("\nProcessing:", state_name)


    # --------------------------------------------------------
    # CREATE MASK FOR STATE
    # --------------------------------------------------------

    mask = geometry_mask(

        [geometry],

        transform=raster_transform,

        invert=True,

        out_shape=(
            raster_height,
            raster_width
        )
    )


    # --------------------------------------------------------
    # FIND VALID RAINFALL PIXELS
    # --------------------------------------------------------

    valid_pixels = (
        mask
        &
        np.isfinite(rainfall)
    )


    rows, cols = np.where(
        valid_pixels
    )


    if len(rows) == 0:

        print(
            "WARNING: No valid rainfall pixels found."
        )

        continue


    print(
        "Valid pixels:",
        len(rows)
    )


    # --------------------------------------------------------
    # RANDOMLY SAMPLE PIXELS
    # --------------------------------------------------------

    sample_count = min(
        SAMPLES_PER_STATE,
        len(rows)
    )


    selected_indices = rng.choice(

        len(rows),

        size=sample_count,

        replace=False
    )


    selected_rows = rows[
        selected_indices
    ]

    selected_cols = cols[
        selected_indices
    ]


    state_samples = []


    # --------------------------------------------------------
    # PROCESS EACH SAMPLE
    # --------------------------------------------------------

    for row, col in zip(
        selected_rows,
        selected_cols
    ):


        # ----------------------------------------------------
        # 3 x 3 NEIGHBOURHOOD
        # ----------------------------------------------------

        row_start = max(
            0,
            row - 1
        )

        row_end = min(
            raster_height,
            row + 2
        )

        col_start = max(
            0,
            col - 1
        )

        col_end = min(
            raster_width,
            col + 2
        )


        neighbourhood = rainfall[
            row_start:row_end,
            col_start:col_end
        ]


        valid_values = neighbourhood[
            np.isfinite(neighbourhood)
        ]


        if len(valid_values) == 0:
            continue


        # ----------------------------------------------------
        # RAINFALL FEATURES
        # ----------------------------------------------------

        rainfall_mean = float(
            np.mean(valid_values)
        )

        rainfall_max = float(
            np.max(valid_values)
        )

        rainfall_min = float(
            np.min(valid_values)
        )


        # ----------------------------------------------------
        # PIXEL LOCATION
        # ----------------------------------------------------

        x, y = rasterio.transform.xy(

            raster_transform,

            row,

            col
        )


        # ----------------------------------------------------
        # SAVE SAMPLE
        # ----------------------------------------------------

        state_samples.append({

            "state": state_name,

            "longitude": float(x),

            "latitude": float(y),

            "rainfall_mean_mm":
                rainfall_mean,

            "rainfall_max_mm":
                rainfall_max,

            "rainfall_min_mm":
                rainfall_min
        })


    print(
        "Samples generated:",
        len(state_samples)
    )


    all_samples.extend(
        state_samples
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

print("\nCreating feature table...")


data = pd.DataFrame(
    all_samples
)


# ============================================================
# CHECK DATA
# ============================================================

if len(data) == 0:

    print(
        "\nERROR: No training samples were created."
    )

    print("\nPossible causes:")

    print(
        "1. State names do not match."
    )

    print(
        "2. Rainfall raster does not overlap NER."
    )

    print(
        "3. Rainfall raster contains only NoData."
    )

    raise SystemExit


# Remove invalid values

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


data = data.dropna(
    subset=[
        "rainfall_mean_mm",
        "rainfall_max_mm",
        "rainfall_min_mm"
    ]
)


# ============================================================
# SAVE CSV
# ============================================================

data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")

print("=" * 65)
print("FEATURE DATA CREATED SUCCESSFULLY!")
print("=" * 65)


print(
    "\nTotal training samples:",
    len(data)
)


print("\nSamples by state:")


print(
    data["state"]
    .value_counts()
    .sort_index()
)


print("\nFeature columns:")


print(
    data.columns.tolist()
)


print("\nRainfall statistics:")


print(
    data[
        [
            "rainfall_mean_mm",
            "rainfall_max_mm",
            "rainfall_min_mm"
        ]
    ].describe()
)


print("\nSaved as:")


print(
    OUTPUT_FILE
)


print("\n")


print("=" * 65)