# gold.py
# Flask API for GOLD mineralization exploration insights with advanced economic heuristics

import ee
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from geopy.distance import geodesic
import sys
import os
import json
import time
import logging
from math import cos, radians
import socket
from supabase import create_client, Client

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# EARTH ENGINE INIT
# -------------------------------
try:
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
    logging.info("✅ Earth Engine initialized successfully")

except json.JSONDecodeError as e:
    logging.critical(f"❌ Failed to decode EE_CREDENTIALS JSON: {e}")
    log_value = ee_credentials_json_string[:50] + '...' if 'ee_credentials_json_string' in locals() else 'None'
    logging.critical(f"❌ The value of EE_CREDENTIALS was: '{log_value}'")
    sys.exit(1)
except Exception as e:
    logging.critical(f"❌ Failed to initialize Earth Engine for an unknown reason: {e}")
    sys.exit(1)

# -------------------------------
# SUPABASE INIT
# -------------------------------
SUPABASE_URL = 'https://qudbehodnhevjxmdwvta.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF1ZGJlaG9kbmhldmp4bWR3dnRhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ1MjExODksImV4cCI6MjA3MDA5NzE4OX0.YnK8UKd76PBLh3L0zAv5fY0JFK6EWq-UBHLY8jXz55w'
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logging.info("✅ Supabase client initialized successfully")

# -------------------------------
# FLASK APP SETUP
# -------------------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://4a587dd510bb.ngrok-free.app"}})

# -------------------------------
# GLOBAL CONSTANTS / UTILITY FUNCTIONS
# -------------------------------
GOLD_PRICE_PER_G_T = 64  # USD per gram, placeholder
RECOVERY_FACTOR = 0.85   # Recovery fraction
KNOWN_DEPOSITS = [
    {"name": "DRC_Kinross_Belt", "lat": -7.0, "lng": 27.0, "type": "Orogenic"},
    {"name": "Tanzania_Gold_Belt", "lat": -5.0, "lng": 37.0, "type": "Orogenic"},
    {"name": "Witwatersrand", "lat": -26.2041, "lng": 28.0473, "type": "Paleoplacer"},
    {"name": "Carlin_Trend", "lat": 40.7600, "lng": -116.0100, "type": "Carlin-type"},
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

    depth = max(10, 100 - int(10 * slope + 5 * hydrox + 5 * iron))
    alteration_factor = (iron + hydrox + silica) / 3
    estimated_tonnes = max(1000, int(depth * alteration_factor * 1000))
    grade = round(alteration_factor * 5, 2)
    contained_au_oz = round(estimated_tonnes * grade / 31.1035, 2)
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
        "Indicators": {
            "Iron_Oxides": round(iron, 2) if iron else 0,
            "Hydroxyl_Alteration": round(hydrox, 2) if hydrox else 0,
            "Silica_Alteration": round(silica, 2) if silica else 0,
            "Slope": round(slope, 2) if slope else 0
        },
        "Depth_Estimate": depth_quantity
    }

def generate_exploration_report(gold_potential, confidence, depth_estimate, lang='en'):
    """Generate exploration report with detailed recommendations."""
    report = {
        'Gold_Potential_Flag': gold_potential,
        'Geological_Context': "Region shows orogenic gold signatures with quartz vein systems." if gold_potential == "High" else "Mixed geological signals, possible epithermal influence.",
        'Recommended_Methods': "Core drilling and geophysical surveys recommended." if gold_potential in ["High", "Moderate"] else "Surface sampling to confirm potential.",
        'How_to_Exploit': """1. Conduct initial 50x50m grid sampling.
2. Use geophysical surveys (IP, magnetics) to delineate structures.
3. Drill 200-500m boreholes targeting depth estimates.
4. Analyze core samples for Au grade and mineral associations.""",
        'Challenges': "Complex terrain may require advanced drilling techniques." if depth_estimate['Depth_m'] > 300 else "Standard exploration feasible with moderate overburden.",
        'Geological_Layers': "Quartz veins with sericite and chlorite alteration.",
        'Recommended_Grids': "50x50m grid for initial sampling, tightening to 25x25m in high-potential areas.",
        'Surface_Petrography': "Visible quartz outcrops, alteration halos with sericite and chlorite."
    }
    if lang == 'fr':
        translations = {
            'Gold_Potential_Flag': 'Potentiel d\'or',
            'Geological_Context': 'Contexte géologique',
            'Recommended_Methods': 'Méthodes recommandées',
            'How_to_Exploit': 'Comment exploiter',
            'Challenges': 'Défis',
            'Geological_Layers': 'Couches géologiques',
            'Recommended_Grids': 'Grilles recommandées',
            'Surface_Petrography': 'Pétrographie de surface'
        }
        report = {translations.get(k, k): v for k, v in report.items()}
        if gold_potential == "High":
            report[translations['Geological_Context']] = "La région montre des signatures d'or orogénique avec des systèmes de veines de quartz."
        elif gold_potential == "Moderate":
            report[translations['Geological_Context']] = "Signaux géologiques mixtes, possible influence épithermale."
        report[translations['Recommended_Methods']] = "Forage au diamant et levés géophysiques recommandés." if gold_potential in ["High", "Moderate"] else "Échantillonnage de surface pour confirmer le potentiel."
        report[translations['How_to_Exploit']] = """1. Effectuer un échantillonnage initial sur une grille de 50x50m.
2. Utiliser des levés géophysiques (IP, magnétiques) pour délimiter les structures.
3. Forer des trous de 200 à 500m ciblant les estimations de profondeur.
4. Analyser les carottes pour la teneur en Au et les associations minérales."""
        report[translations['Challenges']] = "Terrain complexe pouvant nécessiter des techniques de forage avancées." if depth_estimate['Depth_m'] > 300 else "Exploration standard réalisable avec un recouvrement modéré."
    return report

def compute_composite_score_au(anomaly):
    """Calculates a composite economic score for an anomaly."""
    depth = anomaly["result"].get("Depth_Estimate", {})
    tonnes = depth.get("Estimated_Tonnes", 0) or 0
    grade = depth.get("Grade_g_t", 0) or 0
    confidence = anomaly["result"].get("Confidence", 0)
    return tonnes * grade * GOLD_PRICE_PER_G_T * RECOVERY_FACTOR * (confidence / 100)

def find_nearby_with_geojson(lat, lng, step_km=10, max_distance_km=50, threshold=85):
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
        
        interp["Depth_Estimate"] = estimate_depth_quantity_grade_au(stats)
        interp["Geophysical_Anomalies"] = {}
        interp["Geochemical_Assays"] = {}

        nearest = min(KNOWN_DEPOSITS, key=lambda x: geodesic((curr_lat, curr_lng), (x["lat"], x["lng"])).km)
        interp["Proximity_to_Known_Deposits"] = {
            "deposit": nearest["name"],
            "distance_km": round(geodesic((curr_lat, curr_lng), (nearest["lat"], nearest["lng"])).km, 2)
        }

        interp["GeoJSON"] = {"type": "Point", "coordinates": [curr_lng, curr_lat]}
        interp["Geological_Layers"] = "Quartz veins with sericite and chlorite alteration."
        interp["Recommended_Grids"] = "40x40m grid for high-confidence zones."
        interp["Surface_Petrography"] = "Gossans and altered quartz outcrops."

        composite_score = compute_composite_score_au({"result": interp})
        results.append({
            "lat": curr_lat,
            "lng": curr_lng,
            "distance_km": round(dist, 2),
            "result": interp,
            "composite_score": round(composite_score, 2),
            "ranking_explanation": f"Ranked based on composite score reflecting tonnes ({interp['Depth_Estimate']['Estimated_Tonnes']}), grade ({interp['Depth_Estimate']['Grade_g_t']} g/t), and confidence ({interp['Confidence']}%)."
        })
        
        if interp["Confidence"] >= threshold:
            break
            
        offsets = [
            (step_km / 111, 0), (0, step_km / 111), (0, -step_km / 111), (-step_km / 111, 0),
            (step_km / 111, step_km / 111), (step_km / 111, -step_km / 111),
            (-step_km / 111, step_km / 111), (-step_km / 111, -step_km / 111)
        ]
        
        for dlat, dlng in offsets:
            new_lat = curr_lat + dlat
            new_lng = curr_lng + dlng / cos(radians(curr_lat))
            new_dist = geodesic((lat, lng), (new_lat, new_lng)).km
            if (round(new_lat, 4), round(new_lng, 4)) not in visited and new_dist <= max_distance_km:
                to_check.append((new_lat, new_lng, new_dist))

    return sorted(results, key=lambda x: x['composite_score'], reverse=True)[:5]  # Limit to top 5 anomalies

# -------------------------------
# API ENDPOINTS
# -------------------------------
@app.route('/api/analyze_gold', methods=['GET'])
def analyze_gold():
    """
    Performs the full gold analysis workflow synchronously.
    Returns the result in the format expected by map.html.
    """
    try:
        lat = float(request.args.get("lat"))
        lng = float(request.args.get("lng"))
        lang = request.args.get("lang", "en")
        
        if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            return jsonify({"error": "Invalid coordinates"}), 400
        
        logging.info(f"🚀 Starting gold analysis for Lat: {lat}, Lng: {lng}...")
        
        # 1. Get the Area of Interest
        aoi = get_aoi(lat, lng)

        # 2. Compute the geological proxies from satellite data
        stats = compute_gold_proxies(aoi)

        # 3. Assess the gold potential based on the proxies
        interpretation = assess_gold_potential(stats)
        
        # 4. Calculate contained gold and recovery value
        contained_au_oz = interpretation["Depth_Estimate"]["Contained_Au_oz"]
        recovery_value = contained_au_oz * GOLD_PRICE_PER_G_T * RECOVERY_FACTOR if contained_au_oz else 0

        # 5. Find nearby anomalies
        nearby_anomalies = find_nearby_with_geojson(lat, lng)
        top_anomaly = nearby_anomalies[0] if nearby_anomalies else None
        
        # 6. Build a detailed geological report
        report = generate_exploration_report(interpretation["Gold_Potential"], interpretation["Confidence"], interpretation["Depth_Estimate"], lang)
        
        result_payload = {
            "location": {"lat": lat, "lng": lng},
            "gold_analysis": interpretation,
            "exploration_report": report,
            "Contained_Au_oz": round(contained_au_oz, 2) if contained_au_oz else None,
            "Recovery_Adjusted_Value_USD": round(recovery_value, 2) if recovery_value else None,
            "Gold_Potential_Flag": interpretation["Gold_Potential"],
            "nearby_anomalies": nearby_anomalies,
            "top_economic_anomaly": top_anomaly
        }
        
        logging.info("✅ Analysis completed successfully. Returning results.")
        return jsonify(result_payload)
    
    except Exception as e:
        logging.error(f"❌ Gold analysis failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/send-help-email', methods=['POST'])
def send_help_email():
    """
    Handles help form submissions from map.html and stores them in Supabase.
    """
    try:
        data = request.get_json()
        name = data.get('name')
        phone = data.get('phone')
        email = data.get('email')
        want_trial = data.get('wantTrial')
        issue = data.get('issue')

        if not all([name, phone, email, issue]):
            return jsonify({"error": "Missing required fields"}), 400

        response = supabase.table('support_requests').insert({
            'name': name,
            'phone': phone,
            'email': email,
            'want_trial': want_trial,
            'issue': issue,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ')
        }).execute()

        if response.data:
            logging.info(f"✅ Support request saved: {name}, {phone}, {email}, Want Trial: {want_trial}, Issue: {issue}")
            return jsonify({"message": "Support request saved successfully"}), 200
        else:
            logging.error("❌ Failed to save support request to Supabase")
            return jsonify({"error": "Failed to save support request"}), 500

    except Exception as e:
        logging.error(f"❌ Error in send_help_email: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------
# HTML TEMPLATE ROUTES
# -------------------------------
@app.route('/')
@app.route('/index')
def serve_index():
    """Serve index.html as the default entry point."""
    return render_template('index.html')

@app.route('/<path:page>')
def serve_page(page):
    """Serve any HTML file from the templates directory dynamically."""
    if not page.endswith('.html'):
        page = page + '.html'
    if page in os.listdir('templates'):
        return render_template(page)
    else:
        return jsonify({"error": "Page not found"}), 404

# -------------------------------
# MAIN
# -------------------------------
if __name__ == '__main__':
    ports = [5000, 5001]  # Try these ports
    server_started = False
    
    for port in ports:
        try:
            # Check if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                s.listen(1)
            app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
            server_started = True
            break
        except OSError as e:
            logging.error(f"❌ Port {port} is in use: {e}")
            continue
    
    if not server_started:
        logging.critical("❌ Could not start server: all attempted ports are in use.")
        sys.exit(1)