# app/virtual_eye_phase4.py
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

# --- NEW IMPORTS FOR PHASE 4 ---
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
PERSIST_FRAMES = 3
AI_NARRATION_INTERVAL = 5 # Seconds before re-evaluating a static scene
MIN_NARRATION_GAP = 0 # Minimum seconds between any two narrations to avoid spamming API
TOP_K = 5 

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
playback_handle = None # Global handle for audio playback, to allow interruption

# --- UTILITY & SETUP FUNCTIONS ---
def load_known_faces(folder_path="known_faces"):
    """ Scans a folder for images, learns the faces, and returns encodings. """
    known_face_encodings = []
    known_face_names = []
    print(f"Loading known faces from '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"Warning: '{folder_path}' directory not found. No faces will be recognized.")
        return known_face_encodings, known_face_names

    image_files = glob.glob(os.path.join(folder_path, "*.*"))
    for image_path in image_files:
        try:
            name = os.path.splitext(os.path.basename(image_path))[0]
            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
                print(f" - Learned face: {name}")
        except Exception as e:
            print(f" - Error loading {os.path.basename(image_path)}: {e}")
    return known_face_encodings, known_face_names

def describe_scene_with_ai(scene_data):
    """ Takes structured scene data and uses an LLM to generate a human-like description. """
    if not llm_model: return "AI model is not available."
    if not scene_data["objects"]: return "The scene appears to be clear."
    
    prompt = "You are an AI assistant for a visually impaired person. Your task is to describe the scene in a clear, concise, and natural way. Do not use robotic language. Be descriptive and helpful. Here is the data from the camera:\n\n"
    object_descriptions = [f"{obj.get('count', 1)} {obj['label']}s are {obj['position']}" if obj.get('count', 1) > 1 else f"{obj['label']} is {obj['position']}" for obj in scene_data["objects"]]
    prompt += ", ".join(object_descriptions) + ".\n\nDescribe this scene in a single, fluid sentence."

    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini AI: {e}")
        return "There was an error describing the scene."

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
    """ Puts a new sentence in the queue, clearing any old ones first. """
    while not speech_queue.empty():
        try: speech_queue.get_nowait()
        except queue.Empty: continue
    speech_queue.put(text)

def get_object_position(center_x, frame_width):
    """ Divides the frame into three vertical zones: left, center, right. """
    zone_boundary_1 = frame_width / 3
    zone_boundary_2 = 2 * frame_width / 3
    if center_x < zone_boundary_1: return "on your left"
    elif center_x <= zone_boundary_2: return "in front of you"
    else: return "on your right"

# --- MAIN APPLICATION ---
def main():
    from ultralytics import YOLO 
    
    print("Starting Virtual Eye (Streaming Audio)...")
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

    # --- STATE TRACKING ---
    last_confirmed_set = set()
    last_ai_narration_time = 0
    last_spoken_narration = ""
    frame_skip_counter = 0
    annotated_frame = None

    print("Application is running. Press ESC in the video window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_height, frame_width, _ = frame.shape
        is_processing_frame = frame_skip_counter % 2 == 0 # Process every 2nd frame to increase responsiveness
        frame_skip_counter += 1

        if is_processing_frame:
            results = model.predict(frame, conf=0.45, verbose=False, classes=[0, 2, 5, 7, 24, 26, 28, 39, 41, 63, 64, 67]) # Focus on relevant classes
            annotated_frame = results[0].plot()

            current_confirmed_objects = []
            
            # --- NEW, STABLE FACE RECOGNITION LOGIC ---
            # 1. Find all faces in the current frame first.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "an unknown person"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                face_names.append(name)
            
            # Process YOLO results
            if hasattr(results[0].boxes, "cls"):
                all_boxes = results[0].boxes
                for i in range(len(all_boxes.cls)):
                    label = model.names[int(all_boxes.cls[i])]
                    box = all_boxes.xyxy[i]
                    x1, y1, x2, y2 = map(int, box)
                    
                    if label == "person":
                        # 2. Match YOLO person box with a found face
                        person_center_x = (x1 + x2) / 2
                        person_center_y = (y1 + y2) / 2
                        
                        found_match = False
                        for j, face_loc in enumerate(face_locations):
                            top, right, bottom, left = face_loc
                            # Check if the center of the YOLO person box is inside the face location
                            if left < person_center_x < right and top < person_center_y < bottom:
                                label = face_names[j] # Assign the recognized name
                                found_match = True
                                break
                        if not found_match and face_names:
                            # If no direct match, could be an unrecognized face
                            label = "an unknown person"
                    
                    center_x = (x1 + x2) / 2
                    position = get_object_position(center_x, frame_width)
                    current_confirmed_objects.append({'label': label, 'position': position})

            current_confirmed_set = set(obj['label'] for obj in current_confirmed_objects)
            time_since_last_narration = time.time() - last_ai_narration_time
            
            should_call_ai = False
            # --- NEW COOLDOWN LOGIC TO PREVENT API SPAM ---
            if current_confirmed_set != last_confirmed_set and time_since_last_narration > MIN_NARRATION_GAP:
                should_call_ai = True
            elif time_since_last_narration > AI_NARRATION_INTERVAL and current_confirmed_set:
                should_call_ai = True

            if should_call_ai:
                object_counts = defaultdict(list)
                for obj in current_confirmed_objects:
                    object_counts[obj['label']].append(obj['position'])
                
                final_scene_objects = []
                for label, positions in object_counts.items():
                    final_scene_objects.append({'label': label, 'position': positions[0], 'count': len(positions)})
                
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
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    speech_thread.join()

if __name__ == "__main__":
    main()

