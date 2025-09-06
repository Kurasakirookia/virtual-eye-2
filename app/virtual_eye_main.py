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

# --- NEW IMPORTS FOR PHASE 4 ---
import face_recognition
import google.generativeai as genai # <-- CORRECTED TYPO HERE
import edge_tts
from dotenv import load_dotenv # <-- ADDED FOR .env SUPPORT

# --- CONFIGURATION & TUNING ---
PRIORITY = {
    "person": 10, "car": 5, "bus": 5, "bicycle": 4, "motorbike": 4,
    "dog": 3, "cat": 3, "chair": 2, "table": 2, "phone": 1, "bottle": 1,
}
PERSIST_FRAMES = 3
AI_NARRATION_INTERVAL = 6 # Seconds between calling the AI for a new description
TOP_K = 5 # Let the AI decide from a wider range of objects

# --- AI & TTS SETUP ---
load_dotenv() # <-- ADDED TO LOAD YOUR .env FILE
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

VOICE = "en-US-JennyNeural" # A natural-sounding voice
speech_queue = queue.Queue()

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
            else:
                print(f" - Warning: No face found in {os.path.basename(image_path)}")
        except Exception as e:
            print(f" - Error loading {os.path.basename(image_path)}: {e}")
    return known_face_encodings, known_face_names

def describe_scene_with_ai(scene_data):
    """ Takes structured scene data and uses an LLM to generate a human-like description. """
    if not llm_model:
        return "AI model is not available."
    if not scene_data["objects"]:
        return "The scene appears to be clear."
    
    # Construct a detailed prompt for the AI
    prompt = "You are an AI assistant for a visually impaired person. Your task is to describe the scene in a clear, concise, and natural way. Do not use robotic language. Be descriptive and helpful. Here is the data from the camera:\n\n"
    
    object_descriptions = []
    for obj in scene_data["objects"]:
        label = obj['label']
        position = obj['position']
        # Handle plural vs singular for the prompt
        if obj.get('count', 1) > 1:
            object_descriptions.append(f"{obj['count']} {label}s are {position}")
        else:
            object_descriptions.append(f"{label} is {position}")
            
    prompt += ", ".join(object_descriptions) + "."
    prompt += "\n\nDescribe this scene."

    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini AI: {e}")
        return "There was an error describing the scene."

async def amain_tts(text_to_speak) -> None:
    """ Asynchronously communicates with Edge TTS to generate and play audio. """
    communicate = edge_tts.Communicate(text_to_speak, VOICE)
    # This is a placeholder for playing audio. In a real app, you'd save to a file
    # and play it with a library like `playsound` or `pygame`. For simplicity,
    # we'll just show it's working without actual audio output in this script.
    async for _ in communicate.stream():
        pass # This simulates the generation process

def speech_worker():
    """ The speech thread that handles playing the generated audio. """
    while True:
        text = speech_queue.get()
        if text is None: break
        print(f"AI Narrator: \"{text}\"")
        try:
            asyncio.run(amain_tts(text))
        except Exception as e:
            print(f"Error during TTS generation: {e}")
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
    
    print("Starting Virtual Eye (Phase 4: AI Integration)...")
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
    frame_skip_counter = 0
    annotated_frame = None

    print("Application is running. Press ESC in the video window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_height, frame_width, _ = frame.shape
        
        is_processing_frame = frame_skip_counter % 3 == 0
        frame_skip_counter += 1

        if is_processing_frame:
            results = model.predict(frame, conf=0.45, verbose=False)
            annotated_frame = results[0].plot()

            current_confirmed_objects = []
            if hasattr(results[0].boxes, "cls"):
                all_boxes = results[0].boxes
                rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                for i in range(len(all_boxes.cls)):
                    label = model.names[int(all_boxes.cls[i])]
                    box = all_boxes.xyxy[i]
                    x1, y1, x2, y2 = map(int, box)
                    
                    if label == "person" and known_face_names:
                        face_image = rgb_small_frame[y1:y2, x1:x2]
                        
                        # --- THE FIX IS HERE ---
                        # Directly compute encodings from the cropped face image.
                        # This removes the unnecessary and error-prone second call to find face locations.
                        face_encodings = face_recognition.face_encodings(face_image)
                        
                        if face_encodings:
                            # Use the first encoding found in the crop
                            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                            name = "an unknown person"
                            if True in matches:
                                first_match_index = matches.index(True)
                                name = known_face_names[first_match_index]
                            label = name

                    center_x = (x1 + x2) / 2
                    position = get_object_position(center_x, frame_width)
                    current_confirmed_objects.append({'label': label, 'position': position})

            current_confirmed_set = set(obj['label'] for obj in current_confirmed_objects)

            time_since_last_narration = time.time() - last_ai_narration_time
            should_narrate = False
            if current_confirmed_set != last_confirmed_set and time_since_last_narration > 3:
                should_narrate = True
            elif time_since_last_narration > AI_NARRATION_INTERVAL and current_confirmed_set:
                should_narrate = True

            if should_narrate:
                object_counts = defaultdict(list)
                for obj in current_confirmed_objects:
                    object_counts[obj['label']].append(obj['position'])
                
                final_scene_objects = []
                for label, positions in object_counts.items():
                    final_scene_objects.append({'label': label, 'position': positions[0], 'count': len(positions)})
                
                final_scene_objects.sort(key=lambda x: PRIORITY.get(x['label'], 0), reverse=True)
                scene_data = {"objects": final_scene_objects[:TOP_K]}

                narration = describe_scene_with_ai(scene_data)
                speak(narration)

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

