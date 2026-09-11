import geopandas as gpd
import rasterio
import numpy as np
import pandas as pd
from shapely.geometry import Point

# -----------------------------------------
# FILES
# -----------------------------------------

BOUNDARY_FILE = "data/state_NWIC.GeoJSON"
RAINFALL_FILE = "data/ner_rainfall_20260907.tif"
OUTPUT_FILE = "data/sikkim_demo_monitoring_points.csv"

# -----------------------------------------
# LOAD SIKKIM BOUNDARY
# -----------------------------------------

gdf = gpd.read_file(BOUNDARY_FILE)

state_column = "state_name"

sikkim = gdf[
    gdf[state_column].astype(str).str.strip().str.lower() == "sikkim"
].copy()

if sikkim.empty:
    raise ValueError("Sikkim boundary was not found.")

print("Sikkim boundary loaded successfully.")

# -----------------------------------------
# OPEN RAINFALL RASTER
# -----------------------------------------

with rasterio.open(RAINFALL_FILE) as src:

    # Reproject boundary to rainfall CRS
    sikkim = sikkim.to_crs(src.crs)

    boundary = sikkim.geometry.union_all()

    minx, miny, maxx, maxy = boundary.bounds

    # -----------------------------------------
    # CREATE GRID
    # -----------------------------------------

    # Approximately 0.05 degree spacing
    # Creates many monitoring locations.
    step = 0.05

    points = []

    x_values = np.arange(minx, maxx, step)
    y_values = np.arange(miny, maxy, step)

    for x in x_values:
        for y in y_values:

            point = Point(x, y)

            if boundary.contains(point):

                # Convert point to raster row/column
                row, col = src.index(x, y)

                if (
                    0 <= row < src.height
                    and 0 <= col < src.width
                ):

                    rainfall_raw = src.read(
                        1,
                        window=((row, row + 1), (col, col + 1))
                    )[0, 0]

                    # Ignore invalid/no-data pixels
                    if rainfall_raw >= 0:

                        rainfall_mm = float(rainfall_raw) / 10.0

                        points.append({
                            "monitoring_id": len(points) + 1,
                            "state": "Sikkim",
                            "latitude": y,
                            "longitude": x,
                            "rainfall_mm": rainfall_mm
                        })

# -----------------------------------------
# SAVE DATA
# -----------------------------------------

df = pd.DataFrame(points)

if df.empty:
    raise ValueError("No monitoring points were generated.")

df.to_csv(OUTPUT_FILE, index=False)

print()
print("======================================")
print("SIKKIM DEMO DATA GENERATED")
print("======================================")
print(f"Monitoring points: {len(df)}")
print(f"Output: {OUTPUT_FILE}")
print()
print(df.head())