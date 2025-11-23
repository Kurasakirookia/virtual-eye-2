# app/virtual_eye_server.py
# --- VIRTUAL EYE: AI SERVER ("THE BRAIN") ---
# 
# This script functions as a centralized AI processing server.
# It replaces the local loop logic with a REST API architecture.
#
# ROLES:
# 1. Model Host: Loads heavy AI models (YOLO, Midas, etc.) into RAM once at startup.
# 2. API Provider: Exposes HTTP endpoints using Flask for the mobile client.
# 3. Logic Processor: Handles all "thinking" (scene analysis, navigation routing).
#
# ARCHITECTURE:
# Mobile App (Client) -> Sends Image/Text -> Flask Server (This Script) -> AI Analysis -> Returns JSON Text

import cv2
import time
import os
import glob
import numpy as np
import torch
import io

# --- Core AI Libraries ---
import mediapipe as mp          # For Pose Estimation (Human actions)
import face_recognition         # For identifying known people
import google.generativeai as genai # For generating natural language descriptions
from dotenv import load_dotenv  # For managing API keys securely
from ultralytics import YOLO    # For Object Detection

# --- Navigation Libraries ---
import googlemaps               # Official client for Google Maps Platform
from geopy.geocoders import Nominatim # For address/coordinate conversion
from geopy.exc import GeocoderUnavailable

# --- Server Libraries ---
from flask import Flask, request, jsonify # Web server framework
from flask_cors import CORS               # Allows cross-origin requests (essential for mobile connections)
from PIL import Image                     # For handling raw image bytes from network requests

# ==========================================
# 1. CONFIGURATION & TUNING
# ==========================================

# Priority list for sorting objects before sending to Gemini.
# Higher numbers mean the object is more important to mention first.
PRIORITY = { 
    "person": 10, "laptop": 8, "bottle": 7, "cup": 7, "book": 6, 
    "cell phone": 6, "keyboard": 6, "mouse": 6, "chair": 5, "tv": 5, 
    "car": 5, "bus": 5, "bicycle": 4, "dog": 3, "cat": 3 
}

# Limits the number of objects sent to the LLM to prevent token overflow/lag.
TOP_K = 5 

# Resolution to resize images to before AI processing. 
# Smaller = Faster, Larger = More accurate. 480p is a good balance.
PROCESSING_WIDTH = 480

# Face detection model. 'hog' is faster (CPU friendly), 'cnn' is accurate (GPU needed).
FACE_DETECTION_MODEL = "hog" 

# Toggle Depth Estimation. Set to False if Midas causes too much lag.
USE_MIDAS = True

# ==========================================
# 2. INITIALIZATION & MODEL LOADING
# ==========================================

print("[Server] Initializing Virtual Eye Brain...")
load_dotenv() # Load variables from .env file

# --- Google Gemini Setup ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=GOOGLE_API_KEY)
    # Use 'flash' model for lower latency responses
    llm_model = genai.GenerativeModel('gemini-2.5-pro')
    print("[Server] Gemini AI model configured.")
except Exception as e:
    print(f"[Server] FATAL: Gemini Setup Failed: {e}")
    exit()

# --- Google Maps Setup ---
try:
    GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY")
    if not GOOGLE_MAPS_KEY: raise ValueError("GOOGLE_MAPS_KEY not found.")
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_KEY)
    print("[Server] Google Maps client configured.")
except Exception as e:
    print(f"[Server] FATAL: Maps Setup Failed: {e}")
    exit()

# --- Helper Function to Load Known Faces ---
def load_known_faces(folder_path="known_faces"):
    """
    Scans the 'known_faces' folder. Converts every image found into a
    mathematical face encoding vector used for comparison.
    Returns lists of encodings and corresponding names.
    """
    encodings = []
    names = []
    print(f"[Server] Loading known faces from '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"[Server] Warning: '{folder_path}' not found. Face recognition limited.")
        return encodings, names
    
    for image_name in os.listdir(folder_path):
        path = os.path.join(folder_path, image_name)
        if not os.path.isfile(path) or image_name.startswith('.'): continue
        try:
            # Load image and detect face
            img = face_recognition.load_image_file(path)
            found_encodings = face_recognition.face_encodings(img)
            if found_encodings:
                encodings.append(found_encodings[0])
                # Filename becomes the person's name (e.g., "tejas.jpg" -> "tejas")
                names.append(os.path.splitext(image_name)[0])
                print(f" - Loaded: {names[-1]}")
        except Exception as e:
            print(f" - Error loading {image_name}: {e}")
    return encodings, names

# --- Load Heavy Models into Memory ---
# We do this once at startup so per-request processing is fast.
print("[Server] Loading YOLOv8 Object Detection Model...")
yolo_model = YOLO("yolov8n.pt")

midas_model = None
midas_transform = None
midas_device = None

if USE_MIDAS:
    print("[Server] Loading MiDaS Depth Estimation Model...")
    try:
        # Load lightweight MiDaS model from Torch Hub
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_model.eval() # Set to evaluation mode (no training)
        
        # Load transforms to normalize images for Midas
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = midas_transforms.small_transform
        
        # Check for GPU, fallback to CPU
        midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas_model.to(midas_device)
        print(f"[Server] MiDaS loaded on {midas_device}.")
    except Exception as e:
        print(f"[Server] Warning: MiDaS failed to load ({e}). Depth features disabled.")
        USE_MIDAS = False

print("[Server] Loading MediaPipe Pose Model...")
mp_pose = mp.solutions.pose
# static_image_mode=True because we analyze single snapshots, not a video stream
pose_estimator = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

known_face_encodings, known_face_names = load_known_faces()
print("[Server] All AI models loaded. Server Ready.")


# ==========================================
# 3. CORE ANALYSIS LOGIC
# ==========================================

def get_person_action(pose_results):
    """
    Analyzes human body landmarks to guess actions.
    Logic: Compares hip vs. knee Y-coordinates.
    """
    if not pose_results or not pose_results.pose_landmarks: return None
    lm = pose_results.pose_landmarks.landmark
    try:
        # MediaPipe coordinates are normalized [0.0, 1.0]. Y increases downwards.
        hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
        knee_y = lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        
        # Heuristic: If hip is notably lower (higher Y value) than the knee?
        # Wait, for sitting on a chair: Hip is usually roughly level or slightly above knee.
        # For standing: Hip is strictly above knee.
        # Let's stick to the simple heuristic we verified earlier:
        shoulder_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        threshold = (knee_y - shoulder_y) * 0.1
        
        # If Hip Y > Knee Y, hip is lower than knee -> Unlikely for standard sitting.
        # Let's invert: If Hip Y is vertically close to Knee Y, likely sitting.
        # If Hip Y is significantly above Knee Y (smaller value), standing.
        if abs(hip_y - knee_y) < 0.1: # Very rough approximation
             return "is sitting"
        return "is standing"
    except: return None

def get_object_position(center_x, frame_width):
    """ Returns spatial position (Left/Center/Right) based on X coordinate. """
    one_third = frame_width / 3
    if center_x < one_third: return "on your left"
    elif center_x < 2 * one_third: return "in front of you"
    else: return "on your right"

def generate_gemini_narration(scene_data):
    """
    Sends the structured list of objects to Gemini to create a natural sentence.
    """
    if not scene_data: return "I don't see any specific objects of interest right now."
    
    # Construct prompt
    object_phrases = []
    for obj in scene_data:
        phrase = obj['label']
        if obj.get('action'): phrase += f" who {obj['action']}"
        phrase += f" {obj['position']}"
        if obj.get('distance_label'): phrase += f" ({obj['distance_label']})"
        object_phrases.append(phrase)
        
    prompt = (
        "You are an AI assistant for a visually impaired person. Describe the scene in a clear, concise, and natural way. Here is the data from the camera:\n\n"
        f"{', '.join(object_phrases)}.\n"
        "Keep it concise, helpful, and fluid. Avoid robotic lists."
    )
    
    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return "I am having trouble synthesizing the description."

def analyze_single_frame(frame):
    """
    The Master Function. Takes a raw OpenCV frame and runs the full pipeline.
    Returns: The final text string to speak.
    """
    # 1. Resize for speed
    aspect_ratio = frame.shape[0] / frame.shape[1]
    processing_height = int(PROCESSING_WIDTH * aspect_ratio)
    resized_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # 2. Generate Depth Map (if enabled)
    raw_depth_map = None
    if USE_MIDAS and midas_model:
        try:
            with torch.no_grad():
                input_tensor = midas_transform(rgb_frame).to(midas_device)
                if input_tensor.ndim == 3: input_tensor = input_tensor.unsqueeze(0)
                prediction = midas_model(input_tensor)
                # Resize depth map to match frame dimensions
                raw_depth_map = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(rgb_frame.shape[0], rgb_frame.shape[1]),
                    mode='bicubic', align_corners=False
                ).squeeze().cpu().numpy()
        except Exception as e:
            print(f"[Analysis] Depth failed: {e}")

    # 3. Object Detection (YOLO)
    results = yolo_model.predict(rgb_frame, conf=0.45, verbose=False)
    
    scene_objects = []
    persons_detected = [] # Buffer to hold people for face checks
    
    if hasattr(results[0].boxes, "cls"):
        boxes = results[0].boxes
        for i in range(len(boxes.cls)):
            label = yolo_model.names[int(boxes.cls[i])]
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            obj = {'label': label, 'box': [x1, y1, x2, y2]}
            
            if label == 'person': persons_detected.append(obj)
            else: scene_objects.append(obj)

    # 4. Face & Pose Analysis (Only if people are found)
    if persons_detected:
        # Detect faces in the whole frame
        face_locs = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
        face_encs = face_recognition.face_encodings(rgb_frame, face_locs)
        # Estimate pose
        pose_res = pose_estimator.process(rgb_frame)
        action = get_person_action(pose_res)
        
        for p_obj in persons_detected:
            # Default name
            p_obj['label'] = "a person"
            p_obj['action'] = action
            
            # Check if any detected face overlaps with this person's bounding box
            px1, py1, px2, py2 = p_obj['box']
            p_center = ((px1+px2)/2, (py1+py2)/2)
            
            for i, (top, right, bottom, left) in enumerate(face_locs):
                # Check overlap
                if left < p_center[0] < right and top < p_center[1] < bottom:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_face_encodings, face_encs[i])
                    if True in matches:
                        match_index = matches.index(True)
                        p_obj['label'] = known_face_names[match_index] # Identify by name!
                    else:
                        p_obj['label'] = "an unknown person"
                    break # Face found for this box
            scene_objects.append(p_obj)

    # 5. Finalize Data (Position & Depth)
    final_data = []
    depths = []
    
    for obj in scene_objects:
        x1, y1, x2, y2 = obj['box']
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        
        # Set Position
        obj['position'] = get_object_position(cx, resized_frame.shape[1])
        
        # Set Raw Depth
        if raw_depth_map is not None:
            # Clamp coordinates
            cx = min(max(cx, 0), raw_depth_map.shape[1]-1)
            cy = min(max(cy, 0), raw_depth_map.shape[0]-1)
            d = raw_depth_map[cy, cx]
            obj['depth_val'] = d
            depths.append(d)
        
        final_data.append(obj)

    # 6. Calculate Relative Depth labels
    if depths:
        min_d, max_d = min(depths), max(depths)
        rng = max_d - min_d
        for obj in final_data:
            if 'depth_val' in obj:
                if rng < 0.1: # Everything is close together
                    obj['distance_label'] = "near"
                else:
                    # Normalize 0.0 to 1.0 relative to scene
                    norm = (obj['depth_val'] - min_d) / rng
                    if norm > 0.8: obj['distance_label'] = "nearest"
                    elif norm > 0.5: obj['distance_label'] = "near"
                    elif norm > 0.2: obj['distance_label'] = "medium"
                    else: obj['distance_label'] = "farthest"

    # 7. Sort and Narration
    final_data.sort(key=lambda x: PRIORITY.get(x['label'], 0), reverse=True)
    final_data = final_data[:TOP_K] # Only keep top results
    
    return generate_gemini_narration(final_data)


# ==========================================
# 4. FLASK API ENDPOINTS
# ==========================================

app = Flask(__name__)
CORS(app) # Enable CORS so mobile app can connect

@app.route('/describe_scene', methods=['POST'])
def route_describe_scene():
    """
    Endpoint: Receives an image file.
    Action: Runs analyze_single_frame.
    Returns: JSON { "narration": "..." }
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    try:
        # Read image bytes
        file = request.files['image']
        img_bytes = file.read()
        # Convert to OpenCV format
        pil_image = Image.open(io.BytesIO(img_bytes))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run Analysis
        result_text = analyze_single_frame(opencv_image)
        print(f"[API] /describe_scene result: {result_text}")
        
        return jsonify({"narration": result_text})
        
    except Exception as e:
        print(f"[API] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/start_navigation', methods=['POST'])
def route_start_navigation():
    """
    Endpoint: Receives JSON { "query": "coffee", "latitude": 12.3, "longitude": 77.4 }
    Action: Finds nearest place using Google Maps Places API.
    Returns: JSON with confirmation text and destination coordinates.
    """
    data = request.json
    query = data.get('query')
    lat = data.get('latitude')
    lng = data.get('longitude')
    
    if not query or not lat or not lng:
        return jsonify({"error": "Missing data"}), 400
        
    try:
        print(f"[API] Finding nearest '{query}' near {lat},{lng}")
        
        # 1. Find Places
        places = gmaps.places_nearby(
            location=(lat, lng),
            keyword=query,
            rank_by="distance"
        )
        
        if places['status'] == 'OK' and places['results']:
            top = places['results'][0]
            name = top['name']
            addr = top.get('vicinity', 'unknown address')
            dest_loc = top['geometry']['location']
            
            # 2. Get basic walking info (distance/time)
            directions = gmaps.directions(
                (lat, lng),
                dest_loc,
                mode="walking"
            )
            
            dist_text = "unknown distance"
            if directions:
                dist_text = directions[0]['legs'][0]['distance']['text']
            
            # 3. Formulate Confirmation
            narration = f"I found {name} at {addr}. It is {dist_text} away. Should I guide you there?"
            
            return jsonify({
                "narration": narration,
                "destination": dest_loc # {lat: ..., lng: ...}
            })
        else:
            return jsonify({"narration": f"Sorry, I couldn't find any {query} nearby."})
            
    except Exception as e:
        print(f"[API] Nav Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_directions', methods=['POST'])
def route_get_directions():
    """
    Endpoint: Receives JSON { "destination": {lat, lng}, "latitude": ..., "longitude": ... }
    Action: Gets exact walking directions.
    Returns: The first instruction step.
    """
    data = request.json
    dest = data.get('destination') # Expecting {lat:..., lng:...}
    lat = data.get('latitude')
    lng = data.get('longitude')
    
    if not dest or not lat or not lng:
        return jsonify({"error": "Missing location data"}), 400
        
    try:
        directions = gmaps.directions(
            (lat, lng),
            (dest['lat'], dest['lng']),
            mode="walking"
        )
        
        if directions:
            # Get the HTML instruction for the very first step
            raw_instr = directions[0]['legs'][0]['steps'][0]['html_instructions']
            # Simple cleanup of bold tags
            clean_instr = raw_instr.replace("<b>", "").replace("</b>", "").replace("<div style=\"font-size:0.9em\">", ". ").replace("</div>", "")
            
            return jsonify({"narration": f"Okay. {clean_instr}"})
        else:
            return jsonify({"narration": "Sorry, I can't find a route right now."})

    except Exception as e:
        print(f"[API] Directions Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Host 0.0.0.0 allows connection from other devices (your phone) on the network.
    print("[Server] Starting Flask on 0.0.0.0:5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)