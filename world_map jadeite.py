import ee
import geopandas as gpd
import datetime
from shapely.geometry import Point

# Initialize Earth Engine
try:
    ee.Initialize(project='gdelt-disasters-project')
except ee.EEException as e:
    print(f"Failed to initialize Earth Engine: {e}")
    exit()

def get_earth_boundary():
    """Global bounding box: [minLon, minLat, maxLon, maxLat]"""
    return [-180, -90, 180, 90]

def get_jadeite_points(aoi, scale=10000):
    """
    Detect jadeite proxy areas using Sentinel-2 imagery.
    Jadeite detection is complex; this uses a simple NIR/Red ratio threshold as a placeholder.
    """

    start_date = '2023-01-01'
    end_date = '2023-12-31'

    # Filter Sentinel-2 for date, cloud cover, and AOI
    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .filterBounds(aoi)

    if sentinel2.size().getInfo() == 0:
        print("No Sentinel-2 images found for the given area and date range.")
        return []

    def detect_jadeite(image):
        # Bands: B4=Red, B8=NIR
        b4 = image.select('B4')
        b8 = image.select('B8')
        nir_red_ratio = b8.divide(b4).rename('nir_red_ratio')

        # Threshold chosen as placeholder for jadeite spectral signature proxy
        jadeite_mask = nir_red_ratio.gt(1.2).rename('jadeite_mask')

        return image.addBands([nir_red_ratio, jadeite_mask])

    processed = sentinel2.map(detect_jadeite)
    median_img = processed.median().clip(aoi)

    jadeite_areas = median_img.select('jadeite_mask').selfMask()

    vectors = jadeite_areas.reduceToVectors(
        geometry=aoi,
        crs=median_img.select('jadeite_mask').projection(),
        scale=scale,
        geometryType='centroid',
        eightConnected=False,
        maxPixels=1e9,
        bestEffort=True
    )

    geojson = vectors.getInfo()

    points = []
    if 'features' in geojson:
        for feature in geojson['features']:
            lon, lat = feature['geometry']['coordinates']
            points.append({
                "geometry": Point(lon, lat),
                "mineral_type": "jadeite"
            })

    return points

def save_geojson(points, output_file):
    if not points:
        print("No jadeite points detected.")
        return

    gdf = gpd.GeoDataFrame(
        [{"mineral_type": p["mineral_type"], "geometry": p["geometry"]} for p in points],
        geometry="geometry"
    )

    gdf.to_file(output_file, driver="GeoJSON")
    print(f"Saved GeoJSON to {output_file}")

def main():
    print("Starting global jadeite proxy detection at 10km resolution...")
    earth_bbox = get_earth_boundary()
    aoi = ee.Geometry.BBox(*earth_bbox)

    points = get_jadeite_points(aoi, scale=10000)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"jadeite_points_global_{timestamp}.geojson"
    save_geojson(points, output_file)
    return output_file

if __name__ == "__main__":
    out_file = main()
    print(f"Processing complete. Output saved as {out_file}")