import rasterio
import numpy as np

file_path = "data/ner_rainfall_20260907.tif"

with rasterio.open(file_path) as src:

    rainfall = src.read(1).astype(float)

    print("NER rainfall raster loaded successfully!")
    print("Shape:", rainfall.shape)
    print("CRS:", src.crs)
    print("NoData value:", src.nodata)

    # Remove NoData values
    if src.nodata is not None:
        valid = rainfall[rainfall != src.nodata]
    else:
        valid = rainfall[rainfall < 299999]

    # Remove any remaining invalid values
    valid = valid[np.isfinite(valid)]

    print("Valid pixels:", len(valid))
    print("Stored minimum:", valid.min())
    print("Stored maximum:", valid.max())
    print("Stored mean:", valid.mean())

    # NASA IMERG GIS precipitation is scaled by 10
    rainfall_mm = valid / 10.0

    print()
    print("Rainfall after scaling:")
    print("Minimum (mm):", rainfall_mm.min())
    print("Maximum (mm):", rainfall_mm.max())
    print("Mean (mm):", rainfall_mm.mean())