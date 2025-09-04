import cv2

def main():
    cap = cv2.VideoCapture(0)  # try 1 if you have multiple cameras
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Press ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Failed to read frame.")
            break

        cv2.imshow("Virtual Eye - Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
