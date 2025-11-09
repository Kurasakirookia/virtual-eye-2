# app/virtual_eye_pro.py
# --- VIRTUAL EYE: ADVANCED PROTOTYPE ---
# Combines:
# 1. YOLOv8 Object Detection
# 2. face_recognition for Known Faces
# 3. MediaPipe Pose for Action Recognition (sitting/standing)
# 4. Midas for Depth Estimation (near/far)
# 5. Google Gemini for AI Narration
# 6. Edge-TTS for Natural Speech

import cv2
import time
from collections import defaultdict
import threading
import queue
import os
import glob
import numpy as np
import asyncio
import io
import torch # Required for Midas

# --- Core AI Imports ---
from ultralytics import YOLO
import face_recognition
import mediapipe as mp
import google.generativeai as genai
import edge_tts
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play

# --- 1. CONFIGURATION & TUNING ---
PRIORITY = {
    "person": 10, "car": 5, "bus": 5, "bicycle": 4, "motorbike": 4,
    "dog": 3, "cat": 3, "chair": 2, "table": 2, "phone": 1, "bottle": 1,
    "laptop": 8, "tv": 5, "cup": 7, "book": 6, "cell phone": 6, "keyboard": 6, "mouse": 6
}
AI_NARRATION_INTERVAL = 10 # Seconds before re-evaluating a static scene
MIN_NARRATION_GAP = 4 # Cooldown to prevent API spam / repetition
TOP_K = 5 # Let the AI narrate the top 5 objects
FRAME_PROCESSING_INTERVAL = 10 # Process every 10th frame (4 models is heavy!)
PROCESSING_WIDTH = 480 # Resize frame for faster processing
FACE_DETECTION_MODEL = "hog" # 'hog' is faster on CPU, 'cnn' is more accurate
VOICE = "en-US-JennyNeural" # Natural TTS voice
USE_MIDAS = True # Set to False to disable depth estimation

# --- 2. AI & TTS SETUP ---
load_dotenv()
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel('gemini-2.5-pro') # Use flash for speed
    print("[Main] Gemini AI model configured successfully.")
except Exception as e:
    print(f"[Main] FATAL ERROR: Could not configure Gemini AI. Error: {e}")
    llm_model = None

# --- Speech Queue Setup ---
speech_queue = queue.Queue()
playback_handle = None

# --- MediaPipe Pose Setup ---
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# --- 3. UTILITY & SETUP FUNCTIONS ---

async def amain_tts_to_buffer(text_to_speak) -> io.BytesIO:
    """ Asynchronously gets TTS audio and returns it in an in-memory buffer. """
    buffer = io.BytesIO()
    communicate = edge_tts.Communicate(text_to_speak, VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    buffer.seek(0)
    return buffer

def speech_worker():
    """ The speech thread that handles generating and playing audio from memory. """
    global playback_handle
    asyncio.set_event_loop(asyncio.new_event_loop()) # Set up event loop for this thread
    while True:
        text = speech_queue.get()
        if text is None: break
        
        print(f"AI Narrator: \"{text}\"")
        try:
            if playback_handle and playback_handle.is_playing:
                playback_handle.stop()
            
            audio_buffer = asyncio.run(amain_tts_to_buffer(text))
            audio_segment = AudioSegment.from_mp3(audio_buffer)
            playback_handle = play(audio_segment)
        except Exception as e:
            print(f"[SpeechWorker] Error: {e}")
        speech_queue.task_done()

def speak(text):
    """ Puts a new sentence in the queue, clearing any old ones first. """
    while not speech_queue.empty():
        try: speech_queue.get_nowait()
        except queue.Empty: continue
    speech_queue.put(text)

def load_known_faces(folder_path="known_faces"):
    """ Scans a folder for images, learns the faces, and returns encodings. """
    known_face_encodings = []
    known_face_names = []
    print(f"[Main] Loading known faces from '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"[Main] Warning: '{folder_path}' directory not found.")
        return known_face_encodings, known_face_names
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        # Skip hidden files (like .DS_Store) and subfolders
        if not os.path.isfile(image_path) or image_name.startswith('.'):
            continue
        
        try:
            name = os.path.splitext(image_name)[0]
            face_image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(face_image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f" - Learned face: {name}")
            else:
                 print(f" - Warning: No face found in {image_name}")
        except Exception as e:
            print(f" - Error loading {image_name}: {e}")
    return known_face_encodings, known_face_names

def get_person_action(pose_results):
    """Analyzes pose landmarks to determine if a person is sitting or standing."""
    if not pose_results or not pose_results.pose_landmarks:
        return None
    
    landmarks = pose_results.pose_landmarks.landmark
    try:
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

        threshold = (knee_y - shoulder_y) * 0.1
        if hip_y > knee_y + threshold:
            return "is sitting"
        else:
            return "is standing"
    except:
        return None

def get_object_position(center_x, frame_width):
    """ Divides the frame into three vertical zones. """
    zone_boundary_1 = frame_width / 3
    zone_boundary_2 = 2 * frame_width / 3
    if center_x < zone_boundary_1: return "on your left"
    elif center_x <= zone_boundary_2: return "in front of you"
    else: return "on your right"

def describe_scene_with_ai(scene_data):
    """ Takes structured scene data and uses Gemini to generate a human-like description. """
    if not llm_model: return "AI model is not available."
    if not scene_data["objects"]: return "The scene appears to be clear."
    
    prompt = "You are an AI assistant for a visually impaired person. Describe the scene in a clear, concise, and natural way. Do not use robotic language. Be descriptive and helpful. Here is the data from the camera:\n\n"
    object_descriptions = []
    for obj in scene_data["objects"]:
        desc = obj['label']
        if obj.get('action'):
            desc += f" {obj['action']}"
        desc += f" {obj['position']}"
        # --- UPDATED: Use the new relative distance string ---
        if obj.get('distance_label'):
            desc += f" (which is {obj['distance_label']})" # e.g., "a person is sitting in front of you (which is nearest)"
        object_descriptions.append(desc)
            
    prompt += ", and ".join(object_descriptions) + "."
    prompt += "\n\nDescribe this scene in a single, fluid sentence."
    
    print(f"[Gemini] Sending prompt: {prompt}")

    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini] Error calling Gemini AI: {e}")
        return "There was an error describing the scene."

# --- 4. MAIN APPLICATION ---
def main():
    print("Starting Virtual Eye (Advanced Prototype)...")
    
    known_face_encodings, known_face_names = load_known_faces()
    
    print("[Main] Loading YOLOv8n model...")
    yolo_model = YOLO("yolov8n.pt")
    
    # --- Optional MiDaS depth estimator ---
    midas_model = None
    midas_transform = None
    midas_device = None
    if USE_MIDAS:
        try:
            import torch
            print("[Main] Attempting to load MiDaS (this may take a moment)...")
            midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas_model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            midas_transform = midas_transforms.small_transform
            midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            midas_model.to(midas_device)
            print(f"[Main] MiDaS loaded on {midas_device}.")
        except Exception as e:
            print(f"[Main] MiDaS not available (torch may be missing or failed to load). Skipping depth. Error: {e}")
            midas_model = None
    
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    
    print("[Main] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] FATAL ERROR: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_confirmed_set = set()
    last_ai_narration_time = 0
    last_spoken_narration = ""
    frame_skip_counter = 0
    last_annotated_frame = None

    print("[Main] Application is running. Press ESC in the video window to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        annotated_frame = frame.copy() 
        
        aspect_ratio = frame.shape[0] / frame.shape[1]
        processing_height = int(PROCESSING_WIDTH * aspect_ratio)
        resized_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height))
        
        is_processing_frame = frame_skip_counter % FRAME_PROCESSING_INTERVAL == 0
        frame_skip_counter += 1

        if is_processing_frame:
            print(f"\n--- Processing Frame {frame_skip_counter} ---")
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # --- 1. Midas Depth Estimation (Run first) ---
            raw_depth_map = None
            if midas_model is not None:
                try:
                    with torch.no_grad():
                        input_tensor = midas_transform(rgb_frame).to(midas_device)
                        if input_tensor.ndim == 3:
                            input_tensor = input_tensor.unsqueeze(0)
                        prediction = midas_model(input_tensor)
                        raw_depth_map = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=(rgb_frame.shape[0], rgb_frame.shape[1]),
                            mode='bicubic', align_corners=False
                        ).squeeze().cpu().numpy()
                        # We use the raw map here. Higher value = closer.
                        print("[Main] Midas depth map generated.")
                except Exception as e:
                    print(f"[MiDaS] Error computing depth: {e}")
            
            # --- 2. YOLO Object Detection ---
            yolo_results = yolo_model.predict(rgb_frame, conf=0.45, verbose=False)
            
            scene_objects = []
            persons_detected = []
            
            if hasattr(yolo_results[0].boxes, "cls"):
                all_boxes = yolo_results[0].boxes
                for i in range(len(all_boxes.cls)):
                    label = yolo_model.names[int(all_boxes.cls[i])]
                    box = all_boxes.xyxy[i]
                    x1, y1, x2, y2 = map(int, box)
                    obj_data = {'label': label, 'box': (x1, y1, x2, y2)}
                    if label == 'person':
                        persons_detected.append(obj_data)
                    else:
                        scene_objects.append(obj_data)
            
            print(f"[YOLO] Found {len(persons_detected)} person(s) and {len(scene_objects)} other object(s).")

            # --- 3. Advanced Person Analysis (Face & Pose) ---
            if persons_detected:
                face_locations = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                pose_results = pose_estimator.process(rgb_frame)
                person_action = get_person_action(pose_results)
                
                for person_box in persons_detected:
                    px1, py1, px2, py2 = person_box['box']
                    person_center_x = (px1 + px2) / 2
                    person_center_y = (py1 + py2) / 2
                    
                    person_name = "a person"
                    for i, face_loc in enumerate(face_locations):
                        top, right, bottom, left = face_loc
                        if left < person_center_x < right and top < person_center_y < bottom:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[i])
                            if True in matches:
                                person_name = known_face_names[matches.index(True)]
                            else:
                                person_name = "an unknown person"
                            break
                    
                    person_data = {'label': person_name, 'box': (px1, py1, px2, py2), 'action': person_action}
                    scene_objects.append(person_data)
            
            print(f"[Main] Face and Pose analysis complete.")

            # --- 4. Finalize Objects with Position and RELATIVE Depth ---
            final_scene_objects = []
            object_depths = [] # Store raw depth values
            
            for obj in scene_objects:
                x1, y1, x2, y2 = obj['box']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                obj['position'] = get_object_position(center_x, resized_frame.shape[1])
                
                # Sample the raw depth value
                if raw_depth_map is not None:
                    if 0 <= center_y < raw_depth_map.shape[0] and 0 <= center_x < raw_depth_map.shape[1]:
                        depth_value = raw_depth_map[center_y, center_x]
                        obj['raw_depth'] = depth_value
                        object_depths.append(depth_value)
                
                final_scene_objects.append(obj)
            
            # --- NEW: Relative Depth Classification ---
            if object_depths:
                min_depth = min(object_depths)
                max_depth = max(object_depths)
                depth_range = max_depth - min_depth
                
                for obj in final_scene_objects:
                    if 'raw_depth' in obj:
                        if depth_range < 0.1: # If all objects are at similar depth
                            obj['distance_label'] = "near"
                        else:
                            # Normalize this object's depth relative to other objects
                            norm_depth = (obj['raw_depth'] - min_depth) / depth_range
                            if norm_depth > 0.8: obj['distance_label'] = "nearest"
                            elif norm_depth > 0.5: obj['distance_label'] = "near"
                            elif norm_depth > 0.2: obj['distance_label'] = "medium"
                            else: obj['distance_label'] = "farthest"

            print(f"[Main] Finalized {len(final_scene_objects)} objects with relative depth.")

            # --- 5. AI Narration Trigger (Unchanged) ---
            current_confirmed_set = set(obj['label'] for obj in final_scene_objects)
            time_since_last_narration = time.time() - last_ai_narration_time
            
            should_call_ai = False
            if current_confirmed_set != last_confirmed_set and time_since_last_narration > MIN_NARRATION_GAP:
                should_call_ai = True
            elif time_since_last_narration > AI_NARRATION_INTERVAL and current_confirmed_set:
                should_call_ai = True

            if should_call_ai:
                final_scene_objects.sort(key=lambda x: PRIORITY.get(x['label'], 0), reverse=True)
                scene_data = {"objects": final_scene_objects[:TOP_K]}
                narration = describe_scene_with_ai(scene_data)
                
                if narration and narration != last_spoken_narration and "error" not in narration.lower():
                    print(f"[Main] Sending to speech worker: \"{narration}\"")
                    speak(narration)
                    last_spoken_narration = narration

                last_confirmed_set = current_confirmed_set
                last_ai_narration_time = time.time()
            
            # --- 6. RESTORED: Manual Drawing Loop ---
            scale_x = frame.shape[1] / resized_frame.shape[1]
            scale_y = frame.shape[0] / resized_frame.shape[0]
            
            for obj in final_scene_objects:
                x1, y1, x2, y2 = obj['box']
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                label = obj['label']
                if obj.get('action'):
                    label += f" ({obj['action']})"
                if obj.get('distance_label'):
                    label += f" ({obj['distance_label']})"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            last_annotated_frame = annotated_frame
        
        display_frame = frame 
        if last_annotated_frame is not None:
            display_frame = last_annotated_frame
        
        cv2.imshow("Virtual Eye - Press ESC to quit", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("[Main] Shutting down...")
    pose_estimator.close()
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    speech_thread.join()

if __name__ == "__main__":
    main()