import ee
import geopandas as gpd
import datetime
from shapely.geometry import Point
import numpy as np

# Initialize the Earth Engine connection with your project ID
try:
    ee.Initialize(project='gdelt-disasters-project')
except ee.EEException as e:
    print(f"Failed to initialize Earth Engine. Make sure your project ID is correct and the GEE API is enabled. Error: {e}")
    exit()

def get_drc_boundary():
    """
    Defines the bounding box for the Democratic Republic of Congo.
    [lon_min, lat_min, lon_max, lat_max]
    """
    return [12.0, -14.0, 32.0, 5.0]

def get_mineral_points(aoi, scale):
    """
    Processes a single area of interest (AOI) for mineral detection.
    """
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .filterBounds(aoi)

    if sentinel2.size().getInfo() == 0:
        return []

    def detect_minerals(image):
        b2 = image.select('B2')
        b4 = image.select('B4')
        b8 = image.select('B8')
        b11 = image.select('B11')

        iron_ratio = b4.divide(b2).rename('iron_ratio')
        clay_ratio = b11.divide(b8).rename('clay_ratio')
        mineral_score = iron_ratio.gt(1.5).Or(clay_ratio.gt(1.2)).rename('mineral_score')

        return image.addBands(iron_ratio).addBands(clay_ratio).addBands(mineral_score)

    processed_collection = sentinel2.map(detect_minerals)
    processed_image = processed_collection.median().clip(aoi)

    mineral_locations = processed_image.select('mineral_score').eq(1).selfMask().reduceToVectors(
        geometry=aoi,
        crs=processed_image.select('mineral_score').projection(),
        scale=scale,
        geometryType='centroid',
        eightConnected=False,
        maxPixels=1e9,
        bestEffort=True
    )

    geojson_data = mineral_locations.getInfo()

    mineral_points = []
    if 'features' in geojson_data:
        for feature in geojson_data['features']:
            coords = feature['geometry']['coordinates']
            lon, lat = coords
            mineral_points.append({
                "geometry": Point(lon, lat),
                "mineral_type": "Detected",
                "iron_ratio": None,
                "clay_ratio": None
            })
    
    return mineral_points

def save_geojson(points, output_file):
    """Saves the detected mineral points to a GeoJSON file."""
    if not points:
        print("No mineral points detected.")
        return

    gdf = gpd.GeoDataFrame(points, geometry=[p["geometry"] for p in points])
    gdf["mineral_type"] = [p["mineral_type"] for p in points]
    gdf.to_file(output_file, driver="GeoJSON")
    print(f"GeoJSON saved as {output_file}")

def main():
    """Main function to orchestrate the processing of the DRC."""
    
    drc_bbox = get_drc_boundary()
    all_mineral_points = []
    
    print(f"Starting mineral detection for the Democratic Republic of Congo...")

    lon_min, lat_min, lon_max, lat_max = drc_bbox
    cell_size = 2  # Sub-chunk size in degrees
    scale = 10000   # New scale for 10 km resolution
    
    lons = np.arange(lon_min, lon_max, cell_size)
    lats = np.arange(lat_min, lat_max, cell_size)
    
    for min_lon_np in lons:
        for min_lat_np in lats:
            min_lon = float(min_lon_np)
            min_lat = float(min_lat_np)
            max_lon = float(min_lon_np + cell_size)
            max_lat = float(min_lat_np + cell_size)
            
            aoi = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)
            
            print(f"  - Processing sub-region: [{min_lon}, {min_lat}, {max_lon}, {max_lat}] at {scale}m resolution")
            try:
                mineral_points = get_mineral_points(aoi, scale)
                all_mineral_points.extend(mineral_points)
            except ee.EEException as e:
                print(f"  - Skipping sub-region due to error: {e}")
                continue
    
    print("\nDRC processing complete. Saving results.")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mineral_points_DRC_{timestamp}.geojson"
    save_geojson(all_mineral_points, output_file)
    return output_file

if __name__ == "__main__":
    output_file = main()
    print(f"\nMineral extraction completed. Output saved as {output_file}")