# app/virtual_eye_phase5.py
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

# --- NEW IMPORTS FOR PHASE 5 ---
import mediapipe as mp
import face_recognition
import google.generativeai as genai
import edge_tts
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play

# --- CONFIGURATION & TUNING ---
PRIORITY = {
    "person": 10, "car": 5, "bus": 5, "bicycle": 4, "motorbike": 4,
    "dog": 3, "cat": 3, "chair": 2, "table": 2, "phone": 1, "bottle": 1,
}
AI_NARRATION_INTERVAL = 10 # Seconds before re-evaluating a static scene
MIN_NARRATION_GAP = 4 # Cooldown to prevent API spam
TOP_K = 5 
FRAME_PROCESSING_INTERVAL = 5 # Process every 5th frame to balance performance and responsiveness

# --- AI & TTS SETUP ---
load_dotenv()
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Gemini AI model configured successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not configure Gemini AI. Error: {e}")
    llm_model = None

VOICE = "en-US-JennyNeural"
speech_queue = queue.Queue()
playback_handle = None

# --- MEDIAPIPE POSE SETUP ---
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- UTILITY & SETUP FUNCTIONS ---
def load_known_faces(folder_path="known_faces"):
    known_face_encodings, known_face_names = [], []
    print(f"Loading known faces from '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"Warning: '{folder_path}' directory not found.")
        return known_face_encodings, known_face_names
    for image_path in glob.glob(os.path.join(folder_path, "*.*")):
        try:
            name = os.path.splitext(os.path.basename(image_path))[0]
            face_image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(face_image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f" - Learned face: {name}")
        except Exception as e:
            print(f" - Error loading {os.path.basename(image_path)}: {e}")
    return known_face_encodings, known_face_names

def get_person_action(landmarks):
    """Analyzes pose landmarks to determine if a person is sitting or standing."""
    try:
        # Get Y coordinates of hip and knee. In images, a higher Y value is lower on the screen.
        hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        knee_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
        shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y

        # Simple heuristic: If the hip is visibly lower than the knee, the person is likely sitting.
        # We add a small threshold based on body proportion to avoid misclassification.
        threshold = (knee_y - shoulder_y) * 0.1 # 10% of the torso height
        if hip_y > knee_y + threshold:
            return "is sitting"
        else:
            return "is standing"
    except:
        return None # Return nothing if landmarks aren't clear

def describe_scene_with_ai(scene_data):
    if not llm_model: return "AI model is not available."
    if not scene_data["objects"]: return "The scene appears to be clear."
    
    prompt = "You are an AI assistant for a visually impaired person. Describe the scene in a clear, concise, natural way. Here is the data from the camera:\n\n"
    object_descriptions = []
    for obj in scene_data["objects"]:
        desc = obj['label']
        if obj.get('action'):
            desc += f" {obj['action']}" # e.g., "Tejas is sitting"
        desc += f" {obj['position']}" # e.g., "Tejas is sitting in front of you"
        object_descriptions.append(desc)
            
    prompt += ", and ".join(object_descriptions) + "."
    prompt += "\n\nDescribe this scene in a single, fluid sentence."

    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini AI: {e}")
        return "There was an error describing the scene."

async def amain_tts_to_buffer(text_to_speak):
    buffer = io.BytesIO()
    communicate = edge_tts.Communicate(text_to_speak, VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    buffer.seek(0)
    return buffer

def speech_worker():
    global playback_handle
    while True:
        text = speech_queue.get()
        if text is None: break
        print(f"AI Narrator: \"{text}\"")
        try:
            if playback_handle and playback_handle.is_playing():
                playback_handle.stop()
            audio_buffer = asyncio.run(amain_tts_to_buffer(text))
            audio_segment = AudioSegment.from_mp3(audio_buffer)
            playback_handle = play(audio_segment)
        except Exception as e:
            print(f"Error during TTS generation or playback: {e}")
        speech_queue.task_done()

def speak(text):
    while not speech_queue.empty():
        try: speech_queue.get_nowait()
        except queue.Empty: continue
    speech_queue.put(text)

def get_object_position(center_x, frame_width):
    zone_boundary_1 = frame_width / 3
    zone_boundary_2 = 2 * frame_width / 3
    if center_x < zone_boundary_1: return "on your left"
    elif center_x <= zone_boundary_2: return "in front of you"
    else: return "on your right"

# --- MAIN APPLICATION ---
def main():
    from ultralytics import YOLO 
    
    print("Starting Virtual Eye (Phase 5: Pose Estimation)...")
    known_face_encodings, known_face_names = load_known_faces()
    
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FATAL ERROR: Could not open webcam.")
        return

    last_confirmed_set = set()
    last_ai_narration_time = 0
    last_spoken_narration = ""
    frame_skip_counter = 0
    annotated_frame = None

    print("Application is running. Press ESC in the video window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        is_processing_frame = frame_skip_counter % FRAME_PROCESSING_INTERVAL == 0
        frame_skip_counter += 1

        if is_processing_frame:
            annotated_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- Primary YOLO Detection ---
            yolo_results = model.predict(rgb_frame, conf=0.45, verbose=False, classes=[0, 24, 26, 28, 39, 41, 63, 64, 67])
            annotated_frame = yolo_results[0].plot()

            # --- Scene Analysis ---
            current_confirmed_objects = []
            persons_detected = []
            
            # 1. First, process all YOLO results to find people and other objects
            if hasattr(yolo_results[0].boxes, "cls"):
                all_boxes = yolo_results[0].boxes
                for i in range(len(all_boxes.cls)):
                    label = model.names[int(all_boxes.cls[i])]
                    box = all_boxes.xyxy[i]
                    x1, y1, x2, y2 = map(int, box)
                    
                    obj_data = {'label': label, 'box': (x1, y1, x2, y2)}
                    if label == 'person':
                        persons_detected.append(obj_data)
                    else:
                        current_confirmed_objects.append(obj_data)

            # 2. If people were found, run advanced analysis on the LARGEST person
            if persons_detected:
                # Sort people by the size of their bounding box to focus on the most prominent one
                persons_detected.sort(key=lambda p: (p['box'][2] - p['box'][0]) * (p['box'][3] - p['box'][1]), reverse=True)
                
                # --- Run Pose & Face Recognition on the main person ---
                main_person = persons_detected[0]
                px1, py1, px2, py2 = main_person['box']
                person_name = "a person" # Default name
                person_action = None

                # Face Recognition
                face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        if True in matches:
                            first_match_index = matches.index(True)
                            person_name = known_face_names[first_match_index]
                            break # Assume one known face for now
                
                # Pose Estimation
                pose_results = pose_estimator.process(rgb_frame)
                if pose_results.pose_landmarks:
                    person_action = get_person_action(pose_results.pose_landmarks)
                
                # Add the detailed person data to the scene
                main_person_data = {
                    'label': person_name,
                    'box': (px1, py1, px2, py2),
                    'action': person_action
                }
                current_confirmed_objects.append(main_person_data)


            # 3. Finalize positions for all objects to be described
            final_scene_objects = []
            for obj in current_confirmed_objects:
                x1, y1, x2, y2 = obj['box']
                center_x = (x1 + x2) / 2
                obj['position'] = get_object_position(center_x, frame.shape[1])
                final_scene_objects.append(obj)

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
                    speak(narration)
                    last_spoken_narration = narration

                last_confirmed_set = current_confirmed_set
                last_ai_narration_time = time.time()
        
        if annotated_frame is not None:
            cv2.imshow("Virtual Eye - Press ESC to quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Shutting down...")
    pose_estimator.close()
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    speech_thread.join()

if __name__ == "__main__":
    main()
