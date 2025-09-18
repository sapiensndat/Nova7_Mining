from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import math

# --- FLASK APPLICATION SETUP ---
app = Flask(__name__)
CORS(app)

# --- GOOGLE EARTH ENGINE INITIALIZATION ---
try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("✅ GEE connection successful.")
except ee.EEException:
    print("⏳ Authenticating to Google Earth Engine. Please follow the instructions.")
    ee.Authenticate()
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# --- API ENDPOINT DEFINITION ---
@app.route('/api/analyze', methods=['GET'])
def analyze_location():
    """
    API endpoint to analyze gold prospectivity for a given lat/lng.
    e.g., /api/analyze?lat=40.758&lng=-116.864
    """
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Invalid or missing 'lat' and 'lng' parameters."}), 400

    print(f"🛰️  Processing request for LAT: {lat}, LNG: {lng}...")

    try:
        # --- CORE GEE ANALYSIS LOGIC ---
        target_point = ee.Geometry.Point([lng, lat])
        aoi = target_point.buffer(15000)

        def mask_s2_clouds(image):
            qa = image.select('QA60')
            cloud_bit_mask, cirrus_bit_mask = 1 << 10, 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
            return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

        s2_image = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate('2022-01-01', '2025-08-29')
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))
                     .filterBounds(aoi)
                     .map(mask_s2_clouds)
                     .median()
                     .clip(aoi))

        iron_oxide_ratio = s2_image.select('B11').divide(s2_image.select('B8A'))
        clay_hydroxyl_ratio = s2_image.select('B11').divide(s2_image.select('B12'))
        alteration_composite = ee.Image.cat([
            clay_hydroxyl_ratio.rename('ClayHydroxyl'),
            iron_oxide_ratio.rename('IronOxide'),
            s2_image.select('B4').divide(s2_image.select('B2')).rename('RedBlueRatio')
        ])

        dem = ee.Image('USGS/SRTMGL1_003')
        hillshade = ee.Terrain.hillshade(dem.updateMask(dem.gt(0))).clip(aoi)

        # === FIX 1: USE THE OFFICIAL, GLOBAL USGS MINERAL DATABASE ===
        mrds = ee.FeatureCollection('USGS/MRDS/mrds_data')
        
        # === FIX 2: UPDATE THE PROPERTY NAME FOR FILTERING IN THE GLOBAL DATASET ===
        gold_deposits = mrds.filter(ee.Filter.stringContains('commodity_1', 'Gold'))
        
        def add_distance(feature):
            return feature.set('distance_from_target', target_point.distance(feature.geometry()))
        nearest_deposit = gold_deposits.map(add_distance).sort('distance_from_target').first()
        
        glim = ee.Image('users/arjenhaag/glim_all_litho').select('b1').clip(aoi)
        litho_code = glim.reduceRegion(reducer=ee.Reducer.mode(), geometry=target_point, scale=1000).get('b1')
        litho_dict = {
            1: 'Unconsolidated sediments', 2: 'Siliciclastic sedimentary rocks', 3: 'Carbonate sedimentary rocks',
            4: 'Evaporite sedimentary rocks', 5: 'Metamorphic - low grade', 6: 'Metamorphic - medium grade',
            7: 'Metamorphic - high grade', 8: 'Metamorphic - high pressure', 9: 'Plutonic - felsic',
            10: 'Plutonic - mafic', 11: 'Volcanic - felsic', 12: 'Volcanic - mafic',
            13: 'Volcanic - intermediate', 14: 'Pyroclastic rocks', 15: 'Water', 16: 'Ice', 17: 'No data'
        }

        # --- GATHER AND FORMAT RESULTS ---
        nearest_info_server = nearest_deposit.getInfo() if nearest_deposit else None
        litho_code_val = litho_code.getInfo()
        geology_description = litho_dict.get(litho_code_val, "Geology data not available") if litho_code_val is not None else "Geology data not available"

        proximity_result = {"found": False, "message": "No known gold deposits found nearby in the MRDS database."}
        if nearest_info_server:
            dist_meters = nearest_info_server['properties']['distance_from_target']
            coords = nearest_info_server['geometry']['coordinates']
            lat2, lon2, lat1, lon1 = map(math.radians, [coords[1], coords[0], lat, lng])
            dLon = lon2 - lon1
            y = math.sin(dLon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
            bearing_deg = (math.degrees(math.atan2(y, x)) + 360) % 360
            dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            direction = dirs[round(bearing_deg / 22.5) % 16]
            proximity_result = {
                "found": True,
                "name": nearest_info_server['properties']['site_name'],
                "distance_km": f"{(dist_meters / 1000):.2f}",
                "direction": direction
            }

        alteration_map_id = alteration_composite.getMapId({'min': 0.8, 'max': 2.5})
        hillshade_map_id = hillshade.getMapId({'min': 100, 'max': 250})
        true_color_map_id = s2_image.getMapId({'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3})

        response_data = {
            "status": "success",
            "inputCoordinates": {"lat": lat, "lng": lng},
            "results": {
                "geologicalSetting": {"description": geology_description, "source": "GLiM Global Lithology (Coarse Scale)"},
                "structuralControls": {"description": "Linear features in the hillshade may indicate faults or fractures."},
                "hydrothermalAlteration": {
                    "description": "Composite highlights mineral groups associated with gold.",
                    "legend": {"brightRedMagenta": "Clay/Mica Alteration", "brightGreenYellow": "Iron Oxide Alteration", "brightWhite": "Combination Target"}
                },
                "proximityToKnownDeposits": proximity_result,
                "notesAndLimitations": ["This is a first-pass remote sensing analysis.", "All targets require on-the-ground field verification."]
            },
            "mapLayers": {
                "alterationComposite": {"tileUrl": alteration_map_id['tile_fetcher'].url_format},
                "structuralHillshade": {"tileUrl": hillshade_map_id['tile_fetcher'].url_format},
                "trueColor": {"tileUrl": true_color_map_id['tile_fetcher'].url_format}
            }
        }
        print(f"✅ Successfully processed request for LAT: {lat}, LNG: {lng}")
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error processing request for LAT: {lat}, LNG: {lng}. Error: {e}")
        return jsonify({"status": "error", "message": f"An internal server error occurred: {e}"}), 500

# --- RUN THE FLASK SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)