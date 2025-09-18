import ee
from flask import Flask, request, jsonify, render_template, send_from_directory
from geopy.distance import geodesic
from flask_cors import CORS
import datetime
import sys
import os
import json
import time
import base64
import logging

# Configure logging for better visibility in Heroku logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# EARTH ENGINE INIT
# -------------------------------
try:
    ee_credentials_json_string = os.getenv('EE_CREDENTIALS')
    
    if not ee_credentials_json_string:
        raise ValueError("❌ EE_CREDENTIALS environment variable not set.")
    
    # Strip any leading/trailing whitespace or quotes that might be added by the shell
    clean_json_string = ee_credentials_json_string.strip().strip('"')

    ee_credentials_json = json.loads(clean_json_string)

    creds = ee.ServiceAccountCredentials(
        email=ee_credentials_json["client_email"],
        key_data=ee_credentials_json["private_key"]
    )
    ee.Initialize(creds, project=ee_credentials_json["project_id"])
    logging.info("✅ Earth Engine initialized successfully")

except json.JSONDecodeError as e:
    logging.critical(f"❌ Failed to decode EE_CREDENTIALS JSON: {e}")
    # Print the first 50 characters of the string for debugging, if it exists
    log_value = ee_credentials_json_string[:50] + '...' if 'ee_credentials_json_string' in locals() else 'None'
    logging.critical(f"❌ The value of EE_CREDENTIALS was: '{log_value}'")
    sys.exit(1)
except Exception as e:
    logging.critical(f"❌ Failed to initialize Earth Engine for an unknown reason: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

# -------------------------------
# GLOBAL CONSTANTS / ENHANCEMENTS
# -------------------------------
GOLD_PRICE_PER_G_T = 64
RECOVERY_FACTOR = 0.85
KNOWN_DEPOSITS = [
    {"name": "DRC_Kinross_Belt", "lat": -7.0, "lng": 27.0},
    {"name": "Tanzania_Gold_Belt", "lat": -5.0, "lng": 37.0}
]

# -------------------------------
# ALL UTILITY FUNCTIONS
# -------------------------------
def get_aoi(lat, lng, size_km=2.5):
    point = ee.Geometry.Point([lng, lat])
    return point.buffer(size_km * 500).bounds()

def compute_gold_proxies(aoi):
    """
    Computes gold proxies from Sentinel-2 data within a given area of interest.
    Includes robust error handling for API calls.
    """
    try:
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi)
              .filterDate('2022-01-01', '2025-01-01')
              .sort('CLOUDY_PIXEL_PERCENTAGE')
              .first())
        if not s2:
            logging.warning("❌ No Sentinel-2 image found for the given AOI.")
            return {}
        s2 = s2.clip(aoi)
        b2, b3, b4, b8, b11, b12 = [s2.select(b) for b in ["B2", "B3", "B4", "B8", "B11", "B12"]]
        iron_oxide = b4.divide(b2).add(b4.divide(b3)).rename("Iron_Oxide_Index")
        hydroxyl = b11.divide(b8).add(b12.divide(b8)).rename("Hydroxyl_Index")
        silica = b8.divide(b11).rename("Silica_Index")
        dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
        slope = ee.Terrain.slope(dem).rename("Slope")
        combined = iron_oxide.addBands([hydroxyl, silica, slope])
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        # Retry mechanism for getInfo() to handle transient errors
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # The .getInfo() call is where the network request and potential failure occurs.
                return stats.getInfo()
            except Exception as e:
                logging.warning(f"❌ Attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    logging.error("❌ Max retries reached. Returning empty stats.")
                    return {}
    except Exception as e:
        logging.error(f"❌ Earth Engine API call failed unexpectedly: {e}")
        return {}

def estimate_depth_quantity_grade_au(stats):
    """Estimate depth, tonnes, grade (g/t), contained Au ounces, uncertainty index"""
    if not stats:
        return {"Depth_m": None, "Estimated_Tonnes": None, "Grade_g_t": None, "Contained_Au_oz": None, "Uncertainty": None}

    iron = stats.get("Iron_Oxide_Index", 0)
    hydrox = stats.get("Hydroxyl_Index", 0)
    silica = stats.get("Silica_Index", 0)
    slope = stats.get("Slope", 0)

    depth = max(10, 100 - int(10 * slope + 5 * hydrox + 5 * iron))
    alteration_factor = (iron + hydrox + silica) / 3
    estimated_tonnes = max(1000, int(depth * alteration_factor * 1000))
    grade = round(alteration_factor * 5, 2)
    contained_au_oz = round(estimated_tonnes * grade / 31.103, 2)
    uncertainty = min(100, max(10, int(slope * 5)))

    return {
        "Depth_m": depth,
        "Estimated_Tonnes": estimated_tonnes,
        "Grade_g_t": grade,
        "Contained_Au_oz": contained_au_oz,
        "Uncertainty": uncertainty
    }

def estimate_depth_quantity_grade(stats):
    """Estimate depth, tonnes, grade (g/t), uncertainty index"""
    if not stats:
        return {"Depth_m": None, "Estimated_Tonnes": None, "Grade_g_t": None, "Uncertainty": None}

    iron = stats.get("Iron_Oxide_Index", 0)
    hydrox = stats.get("Hydroxyl_Index", 0)
    silica = stats.get("Silica_Index", 0)
    slope = stats.get("Slope", 0)

    depth = max(10, 100 - int(10 * slope + 5 * hydrox + 5 * iron))
    alteration_factor = (iron + hydrox + silica) / 3
    estimated_tonnes = max(1000, int(depth * alteration_factor * 1000))
    grade = round(alteration_factor * 5, 2)
    uncertainty = min(100, max(10, int(slope * 5)))

    return {"Depth_m": depth, "Estimated_Tonnes": estimated_tonnes, "Grade_g_t": grade, "Uncertainty": uncertainty}

def assess_gold_potential(stats):
    if not stats:
        return {"Gold_Potential": "No Data", "Confidence": 0, "Indicators": {}, "Depth_Estimate": None}
    iron = stats.get("Iron_Oxide_Index", 0)
    hydrox = stats.get("Hydroxyl_Index", 0)
    silica = stats.get("Silica_Index", 0)
    slope = stats.get("Slope", 0)
    score = 0
    if iron and iron > 1.5: score += 25
    if hydrox and hydrox > 1.2: score += 25
    if silica and silica > 0.8: score += 20
    if slope and slope > 8: score += 15
    confidence = min(100, score)
    depth_quantity = estimate_depth_quantity_grade_au(stats)
    return {
        "Gold_Potential": "High" if confidence >= 85 else "Moderate" if confidence >= 30 else "Low",
        "Confidence": confidence,
        "Indicators": {"Iron_Oxides": iron, "Hydroxyl_Alteration": hydrox, "Silica_Alteration": silica, "Slope": slope},
        "Depth_Estimate": depth_quantity
    }

def build_geology_report(interp):
    depth = interp.get("Depth_Estimate", {})
    depth_str = f"Estimated depth: {depth.get('Depth_m', 'Unknown')} m; Approx. ore: {depth.get('Estimated_Tonnes', 'Unknown')} tonnes"
    contained_au = depth.get("Contained_Au_oz", None)
    if contained_au:
        depth_str += f"; Contained Au: {contained_au} oz"

    if depth.get("Estimated_Tonnes") and depth.get("Grade_g_t"):
        recovery_value = depth["Estimated_Tonnes"] * depth["Grade_g_t"] * GOLD_PRICE_PER_G_T * RECOVERY_FACTOR
        recovery_value_str = f"${recovery_value:,.0f}"
    else:
        recovery_value_str = "N/A"

    gold_flag = interp.get("Gold_Potential", "Unknown")
    color_flag = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}.get(gold_flag, "⚪")

    return {
        "Geological_Setting": "Host rocks interpreted from satellite + DEM data. Likely metavolcanics / intrusives if alteration is strong.",
        "Structural_Controls": "Lineaments & slopes >8° suggest possible fault zones.",
        "Alteration_Zones": f"Hydroxyl: {interp['Indicators'].get('Hydroxyl_Alteration', 0):.2f}, Silica: {interp['Indicators'].get('Silica_Alteration', 0):.2f}",
        "Geochemical_Anomalies": "Requires field sampling (not available via remote sensing).",
        "Indicator_Minerals": "Possible pyrite, arsenopyrite (gold pathfinders).",
        "Remote_Sensing_Signatures": f"Iron oxide index: {interp['Indicators'].get('Iron_Oxides', 0):.2f}",
        "Geophysical_Data": "Magnetic/gravity data required (not from Sentinel-2).",
        "Surface_Evidence": "Outcrops, gossans, or quartz veins need field validation.",
        "Hydrothermal_Alteration": "Silica + hydroxyl patterns suggest hydrothermal activity.",
        "Proximity_to_Known": "Nearby search results included below (if any).",
        "Ore_Deposit_Model": "Epithermal/greenstone-hosted gold (remote sensing proxy).",
        "Sampling_Assays": "Not available remotely; ground work needed.",
        "Economic_Factors": f"Recovery-adjusted economic value: {recovery_value_str}",
        "Depth_Geometry": depth_str,
        "Gold_Potential_Flag": f"{color_flag} {gold_flag}",
        "Ground_Truthing": "Field survey and core drilling needed."
    }

def compute_composite_score(anomaly):
    depth = anomaly["result"].get("Depth_Estimate", {})
    tonnes = depth.get("Estimated_Tonnes", 0) or 0
    grade = depth.get("Grade_g_t", 0) or 0
    confidence = anomaly["result"].get("Confidence", 0)
    return tonnes * grade * GOLD_PRICE_PER_G_T * RECOVERY_FACTOR * (confidence / 100)

def compute_composite_score_au(anomaly):
    depth = anomaly["result"].get("Depth_Estimate", {})
    tonnes = depth.get("Estimated_Tonnes", 0) or 0
    grade = depth.get("Grade_g_t", 0) or 0
    confidence = anomaly["result"].get("Confidence", 0)
    return tonnes * grade * GOLD_PRICE_PER_G_T * RECOVERY_FACTOR * (confidence / 100)

def multi_mineral_mode(stats, metals=["Au", "Cu", "Co", "Ni", "Li"]):
    results = {}
    base = estimate_depth_quantity_grade_au(stats)
    for metal in metals:
        factor = 1.0
        results[metal] = {
            "Estimated_Tonnes": base["Estimated_Tonnes"] * factor,
            "Grade_g_t": base["Grade_g_t"] * factor,
            "Contained_oz": round(base["Estimated_Tonnes"] * base["Grade_g_t"] * factor / 31.103, 2)
        }
    return results

def temporal_persistence_filter(lat, lng, years=[2020, 2021, 2022, 2023, 2024]):
    stable_count = 0
    for year in years:
        aoi = get_aoi(lat, lng)
        stats = compute_gold_proxies(aoi)
        interp = assess_gold_potential(stats)
        if interp.get("Confidence", 0) >= 85:
            stable_count += 1
    persistence_score = stable_count / len(years) * 100
    return persistence_score

def find_nearby_with_geojson(lat, lng, step_km=10, max_distance_km=100, threshold=85):
    visited = set()
    results = []
    to_check = [(lat, lng, 0)]
    while to_check:
        curr_lat, curr_lng, dist = to_check.pop(0)
        key = (round(curr_lat, 4), round(curr_lng, 4))
        if key in visited or dist > max_distance_km:
            continue
        visited.add(key)
        aoi = get_aoi(curr_lat, curr_lng)
        stats = compute_gold_proxies(aoi)
        interp = assess_gold_potential(stats)

        interp["Depth_Estimate"] = estimate_depth_quantity_grade(stats)
        interp["Geophysical_Anomalies"] = {}
        interp["Geochemical_Assays"] = {}

        nearest = min(KNOWN_DEPOSITS, key=lambda x: geodesic((curr_lat, curr_lng), (x["lat"], x["lng"])).km)
        interp["Proximity_to_Known_Deposits"] = {
            "deposit": nearest["name"],
            "distance_km": round(geodesic((curr_lat, curr_lng), (nearest["lat"], nearest["lng"])).km, 2)
        }
        interp["GeoJSON"] = {"type": "Point", "coordinates": [curr_lng, curr_lat]}

        results.append({"lat": curr_lat, "lng": curr_lng, "distance_km": round(dist, 2), "result": interp})

        if interp["Confidence"] >= threshold:
            break

        offsets = [(step_km / 111, 0), (0, step_km / 111), (0, -step_km / 111), (-step_km / 111, 0),
                  (step_km / 111, step_km / 111), (step_km / 111, -step_km / 111),
                  (-step_km / 111, step_km / 111), (-step_km / 111, -step_km / 111)]
        for dlat, dlng in offsets:
            new_lat, new_lng = curr_lat + dlat, curr_lng + dlng
            new_dist = geodesic((lat, lng), (new_lat, new_lng)).km
            to_check.append((new_lat, new_lng, new_dist))
    return results

# -------------------------------
# WEB PAGES ROUTES
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/<page_name>')
@app.route('/<page_name>.html')
def render_page(page_name):
    try:
        return render_template(f'{page_name}.html')
    except Exception as e:
        logging.error(f"Error rendering page {page_name}.html: {e}")
        return "<h1>404 Not Found</h1><p>The requested URL was not found on the server.</p>", 404

@app.route('/map')
def show_map():
    return render_template('map.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# -------------------------------
# API ENDPOINT
# -------------------------------
@app.route('/api/analyze_gold', methods=['GET'])
def analyze_gold():
    try:
        lat = float(request.args.get("lat"))
        lng = float(request.args.get("lng"))
        stats = compute_gold_proxies(get_aoi(lat, lng))
        interp = assess_gold_potential(stats)
        depth = interp.get("Depth_Estimate", {})

        if depth.get("Estimated_Tonnes") and depth.get("Grade_g_t"):
            recovery_value = depth["Estimated_Tonnes"] * depth["Grade_g_t"] * GOLD_PRICE_PER_G_T * RECOVERY_FACTOR
        else:
            recovery_value = None

        response = {
            "location": {"lat": lat, "lng": lng},
            "gold_analysis": interp,
            "exploration_report": build_geology_report(interp),
            "Contained_Au_oz": depth.get("Contained_Au_oz"),
            "Recovery_Adjusted_Value_USD": recovery_value,
            "Gold_Potential_Flag": interp.get("Gold_Potential")
        }

        if interp["Confidence"] < 85:
            nearby = find_nearby_with_geojson(lat, lng, threshold=85)
            for n in nearby:
                n["composite_score"] = compute_composite_score_au(n)
            nearby_sorted = sorted(nearby, key=lambda x: x["composite_score"], reverse=True)
            response["nearby_anomalies"] = nearby_sorted
            if nearby_sorted:
                top = nearby_sorted[0]
                explanation = f"Top anomaly due to highest composite score ({top['composite_score']:.0f} USD) from tonnes {top['result']['Depth_Estimate']['Estimated_Tonnes']} t, grade {top['result']['Depth_Estimate']['Grade_g_t']} g/t, confidence {top['result']['Confidence']}%."
                response["top_economic_anomaly"] = {**top, "ranking_explanation": explanation}

        return jsonify(response)
    except Exception as e:
        logging.error(f"❌ API error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080))
