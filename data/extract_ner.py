import geopandas as gpd

file_path = "data/state_NWIC.GeoJSON"

gdf = gpd.read_file(file_path)

print("State boundary file loaded successfully!")
print("Number of features:", len(gdf))
print("Columns:")
print(gdf.columns.tolist())

print("\nState information:")
print(gdf[["state", "stcode"]].to_string(index=False))