import geopandas as gpd
import rasterio
from rasterio.mask import mask

# Files
boundary_file = "data/state_NWIC.GeoJSON"
rainfall_file = "data/3B-DAY-L.GIS.IMERG.20260907.V07C.tif"
output_file = "data/ner_rainfall_20260907.tif"

# Load state boundaries
states = gpd.read_file(boundary_file)

print("State boundary loaded successfully!")
print("State name column: state_name")

# 7 NER states required for this dataset
ner_states = [
    "Assam",
    "Meghalaya",
    "Manipur",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura"
]

# Select the 7 states
ner = states[
    states["state_name"].astype(str).str.strip().isin(ner_states)
].copy()

print("NER states found:", len(ner))
print(ner["state_name"].tolist())

# Open NASA rainfall raster
with rasterio.open(rainfall_file) as src:

    print("NASA rainfall raster opened successfully!")
    print("NASA CRS:", src.crs)

    # Convert boundaries to NASA raster CRS
    ner = ner.to_crs(src.crs)

    # Convert geometries to GeoJSON format
    geometries = [
        geom.__geo_interface__
        for geom in ner.geometry
        if geom is not None and not geom.is_empty
    ]

    print("Geometries prepared:", len(geometries))

    # Extract rainfall for the 7 NER states
    clipped, transform = mask(
        src,
        geometries,
        crop=True
    )

    # Update metadata
    metadata = src.meta.copy()

    metadata.update({
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "transform": transform
    })

    # Save extracted rainfall
    with rasterio.open(output_file, "w", **metadata) as dst:
        dst.write(clipped)

print()
print("======================================")
print("NER RAINFALL EXTRACTED SUCCESSFULLY!")
print("======================================")
print("Output:", output_file)