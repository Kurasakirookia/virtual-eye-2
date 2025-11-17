# app/virtual_eye_assistant.py
# --- VIRTUAL EYE: VOICE ASSISTANT PROTOTYPE ---
# New architecture:
# 1. Loads all models at startup.
# 2. Enters a low-CPU "listening" loop for a wake word.
# 3. On wake word, listens for a command ("What's in front of me?").
# 4. On command, captures ONE frame and runs the FULL analysis.
# 5. Speaks the result and goes back to listening.

import cv2
import time
import threading
import queue
import os
import glob
import numpy as np
import asyncio
import io
import torch # Required for Midas
import struct

# --- Core AI Imports ---
from ultralytics import YOLO
import face_recognition
import mediapipe as mp
# Provide a module-level reference to MediaPipe pose landmarks used by helper functions
mp_pose = mp.solutions.pose
import google.generativeai as genai
import edge_tts
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play

# --- New Voice Imports ---
import pvporcupine
import speech_recognition as sr
import pyaudio

# --- 1. CONFIGURATION & TUNING ---
PRIORITY = {
    "person": 10, "car": 5, "bus": 5, "bicycle": 4, "motorbike": 4,
    "dog": 3, "cat": 3, "chair": 2, "table": 2, "phone": 1, "bottle": 1,
    "laptop": 8, "tv": 5, "cup": 7, "book": 6, "cell phone": 6, "keyboard": 6, "mouse": 6
}
TOP_K = 5 
PROCESSING_WIDTH = 480
FACE_DETECTION_MODEL = "hog" 
USE_MIDAS = True
VOICE = "en-US-JennyNeural"
WAKE_WORDS = ["computer", "hey google"] # Built-in wake words
COMMAND_PHRASES = ["what's in front of me", "what do you see", "describe the scene", "what is in front of me"] # What to listen for

# --- 2. AI & TTS SETUP ---
load_dotenv()
try:
    # Gemini API Key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel('gemini-2.5-pro') # Using the Pro model as in your file
    print("[Main] Gemini AI model configured successfully.")
    
    # Picovoice (Wake Word) AccessKey
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    if not PICOVOICE_ACCESS_KEY:
         raise ValueError("PICOVOICE_ACCESS_KEY not found in .env file. See wake_word_setup.md")
    
    # Initialize Porcupine Wake Word Engine
    porcupine = pvporcupine.create(
        access_key=PICOVOICE_ACCESS_KEY,
        keywords=WAKE_WORDS
    )
    print(f"[Main] Porcupine Wake Word Engine loaded. Listening for: {WAKE_WORDS}")
    
except Exception as e:
    print(f"[Main] FATAL ERROR: Could not configure AI services. Error: {e}")
    llm_model = None
    porcupine = None
    exit()

# --- Speech (TTS) Queue Setup ---
speech_queue = queue.Queue()
playback_handle = None

async def amain_tts_to_buffer(text_to_speak) -> io.BytesIO:
    buffer = io.BytesIO()
    communicate = edge_tts.Communicate(text_to_speak, VOICE)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    buffer.seek(0)
    return buffer

def speech_worker():
    global playback_handle
    asyncio.set_event_loop(asyncio.new_event_loop())
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
    while not speech_queue.empty():
        try: speech_queue.get_nowait()
        except queue.Empty: continue
    speech_queue.put(text)
    
# --- 3. HELPER FUNCTIONS (Copied from virtual_eye_pro.py) ---
def load_known_faces(folder_path="known_faces"):
    known_face_encodings, known_face_names = [], []
    print(f"[Main] Loading known faces from '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"[Main] Warning: '{folder_path}' directory not found.")
        return known_face_encodings, known_face_names
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if not os.path.isfile(image_path) or image_name.startswith('.'): continue
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
    if not pose_results or not pose_results.pose_landmarks: return None
    landmarks = pose_results.pose_landmarks.landmark
    try:
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        threshold = (knee_y - shoulder_y) * 0.1
        return "is sitting" if hip_y > knee_y + threshold else "is standing"
    except: return None

def get_object_position(center_x, frame_width):
    zone_boundary_1 = frame_width / 3
    zone_boundary_2 = 2 * frame_width / 3
    if center_x < zone_boundary_1: return "on your left"
    elif center_x <= zone_boundary_2: return "in front of you"
    else: return "on your right"

def describe_scene_with_ai(scene_data):
    if not llm_model: return "AI model is not available."
    if not scene_data["objects"]: return "The scene appears to be clear."
    
    prompt = "You are an AI assistant for a visually impaired person. Describe the scene in a clear, concise, and natural way. Here is the data from the camera:\n\n"
    object_descriptions = []
    for obj in scene_data["objects"]:
        desc = obj['label']
        if obj.get('action'): desc += f" {obj['action']}"
        desc += f" {obj['position']}"
        if obj.get('distance_label'): desc += f" (which is {obj['distance_label']})"
        object_descriptions.append(desc)
            
    prompt += ", and ".join(object_descriptions) + "."
    prompt += "\n\nDescribe this scene in a single, fluid sentence."
    
    print(f"[Gemini] Sending prompt...")
    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return "There was an error describing the scene."

# --- 4. NEW: CORE ANALYSIS FUNCTION (Runs on-demand) ---
def analyze_scene(yolo_model, midas_model, midas_transform, midas_device, pose_estimator, known_face_encodings, known_face_names, frame):
    """
    Takes a single frame and runs the full analysis pipeline on it.
    Returns a list of detected objects with all metadata.
    """
    print("[Analysis] Analyzing single frame...")
    
    # 1. Resize for processing
    aspect_ratio = frame.shape[0] / frame.shape[1]
    processing_height = int(PROCESSING_WIDTH * aspect_ratio)
    resized_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # 2. Midas Depth Estimation
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
                print("[Analysis] Midas depth map generated.")
        except Exception as e:
            print(f"[MiDaS] Error computing depth: {e}")
    
    # 3. YOLO Object Detection
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
    print(f"[Analysis] YOLO Found {len(persons_detected)} person(s) and {len(scene_objects)} other object(s).")

    
    # 4. Advanced Person Analysis (Face & Pose)
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
        print("[Analysis] Face and Pose analysis complete.")
    
    # 5. Finalize Objects with Position and Depth
    final_scene_objects = []
    object_depths = []
    
    for obj in scene_objects:
        x1, y1, x2, y2 = obj['box']
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        obj['position'] = get_object_position(center_x, resized_frame.shape[1])
        
        if raw_depth_map is not None:
            if 0 <= center_y < raw_depth_map.shape[0] and 0 <= center_x < raw_depth_map.shape[1]:
                depth_value = raw_depth_map[center_y, center_x]
                obj['raw_depth'] = depth_value
                object_depths.append(depth_value)
        
        final_scene_objects.append(obj)
    
    if object_depths:
        min_depth = min(object_depths)
        max_depth = max(object_depths)
        depth_range = max_depth - min_depth
        for obj in final_scene_objects:
            if 'raw_depth' in obj:
                if depth_range < 0.1:
                    obj['distance_label'] = "near"
                else:
                    norm_depth = (obj['raw_depth'] - min_depth) / depth_range
                    if norm_depth > 0.8: obj['distance_label'] = "nearest"
                    elif norm_depth > 0.5: obj['distance_label'] = "near"
                    elif norm_depth > 0.2: obj['distance_label'] = "medium"
                    else: obj['distance_label'] = "farthest"

    print(f"[Analysis] Finalized {len(final_scene_objects)} objects with relative depth.")
    return final_scene_objects

# --- 5. NEW MAIN VOICE ASSISTANT LOOP ---
def main():
    if not porcupine:
        print("[Main] Porcupine failed to initialize. Exiting.")
        return

    # --- 1. Load all models ONCE at the start ---
    print("[Main] Loading all AI models, please wait...")
    known_face_encodings, known_face_names = load_known_faces()
    yolo_model = YOLO("yolov8n.pt")
    
    midas_model, midas_transform, midas_device = None, None, None
    if USE_MIDAS:
        try:
            midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas_model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            midas_transform = midas_transforms.small_transform
            midas_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            midas_model.to(midas_device)
            print(f"[Main] Midas loaded on {midas_device}.")
        except Exception as e:
            print(f"[Main] Midas not available. Skipping depth. Error: {e}")
            midas_model = None

    # --- Initialize MediaPipe Pose estimator once for reuse ---
    pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    # --- 2. Initialize Speech Recognizer (for commands) ---
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # --- 3. Initialize Audio Stream for Wake Word ---
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    
    # --- 4. Initialize Camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] FATAL ERROR: Could not open webcam.")
        return
    
    # --- 5. Start Speech Worker Thread ---
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    
    print(f"\n[Main] Ready. Listening for wake word '{WAKE_WORDS[0]}'...")
    
    while True:
        try:
            # --- A. Listen for Wake Word ---
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            
            keyword_index = porcupine.process(pcm)
            
            if keyword_index >= 0:
                print(f"[Main] Wake word '{WAKE_WORDS[keyword_index]}' detected!")
                speak("Yes?") # Give audio feedback
                
                # --- B. Listen for Command ---
                print("[Main] Listening for command...")
                command = ""
                try:
                    with microphone as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"[Main] Heard command: '{command}'")
                except sr.UnknownValueError:
                    print("[Main] Could not understand command.")
                    speak("I didn't catch that.")
                    continue # Go back to listening for wake word
                except sr.RequestError as e:
                    print(f"[Main] STT Error: {e}")
                    speak("Sorry, I'm having trouble connecting.")
                    continue

                # --- C. Process Command ---
                if any(phrase in command for phrase in COMMAND_PHRASES):
                    print("[Main] Command recognized. Analyzing scene...")
                    speak("Okay, looking now.")
                    
                    ret, frame = cap.read()
                    if not ret:
                        print("[Main] Failed to capture frame.")
                        speak("Sorry, I have a problem with the camera.")
                        continue

                    # --- D. Run Full Analysis ---
                    final_scene_objects = analyze_scene(
                        yolo_model, midas_model, midas_transform, midas_device, 
                        pose_estimator, known_face_encodings, known_face_names, frame
                    )
                    
                    if not final_scene_objects:
                        narration = "I don't see any objects of interest."
                    else:
                        # --- E. Get Gemini Narration ---
                        final_scene_objects.sort(key=lambda x: PRIORITY.get(x['label'], 0), reverse=True)
                        scene_data = {"objects": final_scene_objects[:TOP_K]}
                        narration = describe_scene_with_ai(scene_data)
                    
                    speak(narration)
                    print(f"[Main] Finished analysis. Listening for wake word...")
                else:
                    speak("Sorry, I didn't understand that command.")
                    print(f"[Main] Unknown command. Listening for wake word...")

        except KeyboardInterrupt:
            print("\n[Main] Shutting down...")
            break
        except IOError:
            # This can happen if the audio stream has an issue, just restart
            print("[Main] Audio stream overflow. Restarting stream.")
            audio_stream.close()
            audio_stream = pa.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length
            )


    # --- 6. Cleanup ---
    print("[Main] Shutting down all processes...")
    porcupine.delete()
    audio_stream.close()
    pa.terminate()
    if pose_estimator is not None:
        try:
            pose_estimator.close()
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows() # Close any open CV windows (if any)
    speech_queue.put(None)
    speech_thread.join()

if __name__ == "__main__":
    main()