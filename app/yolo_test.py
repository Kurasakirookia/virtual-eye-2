from ultralytics import YOLO
import cv2

def main():
    # load YOLOv8 nano (smallest, fastest)
    model = YOLO("yolov8n.pt")  # downloads if not present

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        # run YOLO inference (track=False = detect only)
        results = model.predict(frame, conf=0.5, verbose=False)
    
        # draw boxes on frame
        annotated_frame = results[0].plot()

        cv2.imshow("Virtual Eye - YOLOv8n", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
