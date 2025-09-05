# app/virtual_eye_main.py
from ultralytics import YOLO
import cv2
import pyttsx3
import time
from collections import defaultdict
import threading
import queue

# --- CONFIGURATION & TUNING ---
# These values are tuned for a balance of responsiveness and stability.
PRIORITY = {
    "person": 5, "car": 4, "bus": 4, "bicycle": 3, "motorbike": 3,
    "dog": 3, "cat": 3, "chair": 2, "table": 2, "phone": 1, "bottle": 1,
}
PERSIST_FRAMES = 3          # Object must be seen for this many frames to be confirmed.
MAX_ANNOUNCE_INTERVAL = 8   # Force an announcement if the scene is static for this long.
TOP_K = 3                   # Announce the top 3 most important objects.

# --- SPEECH QUEUE SETUP ---
speech_queue = queue.Queue()

def speech_worker(engine, q):
    """ A dedicated thread that processes speech tasks from a queue. """
    while True:
        text = q.get()
        if text is None:  # A None object is the signal to stop.
            break
        engine.say(text)
        engine.runAndWait()
        q.task_done()

def speak(text):
    """ 
    Clears any old, outdated announcements from the queue 
    and adds the latest one. This is crucial for responsiveness.
    """
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except queue.Empty:
            continue
    speech_queue.put(text)

# --- UTILITY FUNCTIONS ---
def pluralize(label, count):
    if label == "person": return "people" if count > 1 else "person"
    if count > 1: return label + "s"
    return label

def build_sentence(objects, frame_counts):
    """ Builds a natural-sounding sentence from a list of detected objects. """
    if not objects:
        return "The scene is clear."
    
    parts = []
    for label in objects:
        cnt = frame_counts.get(label, 1)
        if cnt > 1:
            parts.append(f"{cnt} {pluralize(label, cnt)}")
        else:
            prefix = "an" if label and label[0] in 'aeiou' else "a"
            parts.append(f"{prefix} {label}")
            
    if len(parts) == 1:
        return f"I see {parts[0]}."
    
    return f"I see {', '.join(parts[:-1])} and {parts[-1]}."

# --- MAIN APPLICATION ---
def main():
    print("Starting Virtual Eye...")
    model = YOLO("yolov8n.pt")
    
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.setProperty("volume", 0.9)

    speech_thread = threading.Thread(target=speech_worker, args=(engine, speech_queue), daemon=True)
    speech_thread.start()

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FATAL ERROR: Could not open webcam. Is it in use by another program?")
        return

    # --- STATE TRACKING VARIABLES ---
    consecutive_counts = defaultdict(int)
    prev_frame_labels = set()
    last_spoken_set = set()
    last_spoken_top_priority = None
    last_time = 0

    print("Application is running. Press ESC in the video window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        results = model.predict(frame, conf=0.45, verbose=False)
        annotated_frame = results[0].plot()

        # --- Object Detection & Confirmation ---
        raw_labels = [model.names[int(c)] for c in results[0].boxes.cls] if hasattr(results[0].boxes, "cls") else []
        frame_counts = defaultdict(int)
        for label in raw_labels: frame_counts[label] += 1
        curr_frame_set = set(raw_labels)

        # This is the robust way to count CONSECUTIVE frames
        for label in curr_frame_set:
            if label in prev_frame_labels:
                consecutive_counts[label] += 1
            else:
                consecutive_counts[label] = 1 # Start counting
        for label in list(consecutive_counts.keys()):
            if label not in curr_frame_set:
                consecutive_counts[label] = 0 # Reset count
        prev_frame_labels = curr_frame_set
        
        # --- Intelligent Announcement Logic ---
        confirmed = [l for l, cnt in consecutive_counts.items() if cnt >= PERSIST_FRAMES]
        confirmed_sorted = sorted(confirmed, key=lambda x: PRIORITY.get(x, 0), reverse=True)
        
        current_top_k_set = set(confirmed_sorted[:TOP_K])
        current_top_priority = confirmed_sorted[0] if confirmed_sorted else None
        
        # The final, hybrid trigger: responsive AND stable
        should_speak = False
        if current_top_priority != last_spoken_top_priority:      # Trigger 1: Most important object changed
            should_speak = True
        elif current_top_k_set != last_spoken_set:                # Trigger 2: The set of top objects changed
            should_speak = True
        elif time.time() - last_time > MAX_ANNOUNCE_INTERVAL and current_top_k_set: # Trigger 3: Time passed
             should_speak = True

        if should_speak:
            sentence = build_sentence(confirmed_sorted[:TOP_K], frame_counts)
            
            if sentence == "The scene is clear." and not last_spoken_set:
                pass # Avoid repeating "clear" if nothing was there before
            else:
                print(f"Speaking: \"{sentence}\" (Top: {current_top_priority or 'None'})")
                speak(sentence)
                last_spoken_set = current_top_k_set
                last_spoken_top_priority = current_top_priority
                last_time = time.time()

        cv2.imshow("Virtual Eye - Press ESC to quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # --- CLEANUP ---
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None) # Signal the speech thread to exit
    speech_thread.join()   # Wait for the speech thread to finish

if __name__ == "__main__":
    main()
