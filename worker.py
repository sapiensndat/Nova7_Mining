from worker import celery_app  # import the Celery app from worker.py
import os
import ee
from geopy.distance import geodesic
import datetime
import sys
import json
import time
import logging
from math import cos, radians

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create and configure the Celery application
# Heroku uses the REDIS_URL environment variable to provide the Redis connection string.
celery_app = Celery(
    __name__,
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
)

# Optional: Set a timezone and other configurations
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# -------------------------------
# EARTH ENGINE INIT
# -------------------------------
try:
    # Decode the base64-encoded credentials from the environment variable
    # This is a robust way to handle multi-line secrets on services like Heroku or Fly.io
    ee_credentials_json_string = os.getenv('EE_CREDENTIALS')

    if not ee_credentials_json_string:
        raise ValueError("❌ EE_CREDENTIALS environment variable not set.")

    clean_json_string = ee_credentials_json_string.strip().strip('"')
    ee_credentials_json = json.loads(clean_json_string)

    creds = ee.ServiceAccountCredentials(
        email=ee_credentials_json["client_email"],
        key_data=ee_credentials_json["private_key"]
    )
    ee.Initialize(creds, project=ee_credentials_json["project_id"])
    logging.info("✅ Earth Engine initialized successfully in worker")
except Exception as e:
    logging.critical(f"❌ Failed to initialize Earth Engine in worker: {e}")
    sys.exit(1)

# -------------------------------
# GLOBAL CONSTANTS / UTILITY FUNCTIONS
# -------------------------------
GOLD_PRICE_PER_G_T = 64
RECOVERY_FACTOR = 0.85
KNOWN_DEPOSITS = [
    {"name": "DRC_Kinross_Belt", "lat": -7.0, "lng": 27.0},
    {"name": "Tanzania_Gold_Belt", "lat": -5.0, "lng": 37.0}
]

def get_aoi(lat, lng, size_km=2.5):
    """Generates an Area of Interest (AOI) as a buffered EE Geometry."""
    point = ee.Geometry.Point([lng, lat])
    return point.buffer(size_km * 500).bounds()

def compute_gold_proxies(aoi):
    """
    Computes remote sensing indices from a Sentinel-2 image.
    This function handles common EE API failures with retries.
    """
    try:
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi)
              .filterDate('2022-01-01', '2025-01-01')
              .sort('CLOUDY_PIXEL_PERCENTAGE')
              .first())

        if not s2:
            return {}

        s2 = s2.clip(aoi)

        # Select relevant bands
        b2, b3, b4, b8, b11, b12 = [s2.select(b) for b in ["B2", "B3", "B4", "B8", "B11", "B12"]]

        # Compute geological indices
        iron_oxide = b4.divide(b2).add(b4.divide(b3)).rename("Iron_Oxide_Index")
        hydroxyl = b11.divide(b8).add(b12.divide(b8)).rename("Hydroxyl_Index")
        silica = b8.divide(b11).rename("Silica_Index")

        # Get DEM and compute slope
        dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
        slope = ee.Terrain.slope(dem).rename("Slope")

        combined = iron_oxide.addBands([hydroxyl, silica, slope])

        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )

        # Handle potential API timeouts with retries
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                return stats.getInfo()
            except Exception as e:
                logging.warning(f"❌ Attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)
                else:
                    logging.error("❌ Max retries reached. Returning empty stats.")
                    return {}
    except Exception as e:
        logging.error(f"❌ Earth Engine API call failed unexpectedly: {e}")
        return {}


def estimate_depth_quantity_grade_au(stats):
    """Estimates depth, quantity, grade, and contained gold from the computed indices."""
    if not stats:
        return {"Depth_m": None, "Estimated_Tonnes": None, "Grade_g_t": None, "Contained_Au_oz": None, "Uncertainty": None}

    iron = stats.get("Iron_Oxide_Index", 0)
    hydrox = stats.get("Hydroxyl_Index", 0)
    silica = stats.get("Silica_Index", 0)
    slope = stats.get("Slope", 0)

    # Simple heuristic-based estimation
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
    """A lighter version of the estimation function."""
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
    """
    Assesses the gold potential based on computed indices.
    Returns a flag and a confidence score.
    """
    if not stats:
        return {"Gold_Potential": "No Data", "Confidence": 0, "Indicators": {}, "Depth_Estimate": None}
    iron = stats.get("Iron_Oxide_Index", 0)
    hydrox = stats.get("Hydroxyl_Index", 0)
    silica = stats.get("Silica_Index", 0)
    slope = stats.get("Slope", 0)

    # Scoring based on heuristic thresholds
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
    """Builds a human-readable geological report from the analysis results."""
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

def compute_composite_score_au(anomaly):
    """Calculates a composite economic score for an anomaly."""
    depth = anomaly["result"].get("Depth_Estimate", {})
    tonnes = depth.get("Estimated_Tonnes", 0) or 0
    grade = depth.get("Grade_g_t", 0) or 0
    confidence = anomaly["result"].get("Confidence", 0)
    return tonnes * grade * GOLD_PRICE_PER_G_T * RECOVERY_FACTOR * (confidence / 100)

def find_nearby_with_geojson(lat, lng, step_km=10, max_distance_km=100, threshold=85):
    """
    Performs a breadth-first search for nearby gold anomalies.
    Returns a list of anomalies as GeoJSON Feature objects.
    """
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

        # Create a GeoJSON point for the result
        interp["GeoJSON"] = {"type": "Point", "coordinates": [curr_lng, curr_lat]}

        results.append({"lat": curr_lat, "lng": curr_lng, "distance_km": round(dist, 2), "result": interp})
        
        # Stop searching if a high-confidence anomaly is found
        if interp["Confidence"] >= threshold:
            break
            
        # Explore neighboring grid cells
        offsets = [(step_km / 111, 0), (0, step_km / 111), (0, -step_km / 111), (-step_km / 111, 0),
                  (step_km / 111, step_km / 111), (step_km / 111, -step_km / 111),
                  (-step_km / 111, step_km / 111), (-step_km / 111, -step_km / 111)]
        
        for dlat, dlng in offsets:
            new_lat = curr_lat + dlat
            new_lng = curr_lng + dlng / cos(radians(curr_lat)) # Account for longitude compression
            new_dist = geodesic((lat, lng), (new_lat, new_lng)).km
            if (round(new_lat, 4), round(new_lng, 4)) not in visited:
                to_check.append((new_lat, new_lng, new_dist))

    return results

@celery_app.task(bind=True)
def analyze_gold_in_background(self, lat, lng):
    """
    Celery task to perform the full gold analysis workflow.
    It takes a lat/lng, computes remote sensing data, assesses potential,
    and returns a structured payload.
    """
    logging.info(f"🚀 Starting gold analysis for Lat: {lat}, Lng: {lng}...")
    try:
        # 1. Get the Area of Interest
        aoi = get_aoi(lat, lng)

        # 2. Compute the geological proxies from satellite data
        stats = compute_gold_proxies(aoi)

        # 3. Assess the gold potential based on the proxies
        interpretation = assess_gold_potential(stats)
        
        # 4. Find nearby anomalies (in the real world, this would be a more complex service)
        nearby_results = find_nearby_with_geojson(lat, lng)
        
        # 5. Build a detailed geological report
        report = build_geology_report(interpretation)
        
        result_payload = {
            "input_location": {"lat": lat, "lng": lng},
            "analysis_result": interpretation,
            "nearby_anomalies": nearby_results,
            "geology_report": report
        }
        
        logging.info("✅ Analysis completed successfully.")
        return result_payload

    except Exception as e:
        # Update Celery task state to FAILURE if an error occurs
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        logging.error(f"❌ Gold analysis failed: {e}")
        return {"error": str(e)}
