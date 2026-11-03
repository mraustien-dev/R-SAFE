
import cv2

CAMERA_URL = "http://192.168.0.102:8080/video"

WINDOW_NAME = "RSAFE - Android Live Camera"

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("ERROR: Cannot connect to Android camera.")
    exit()

print("RSAFE: Android camera connected!")

# Create a resizable window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Set initial window dimensions
cv2.resizeWindow(WINDOW_NAME, 960, 540)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive video frame.")
        break

    cv2.imshow(WINDOW_NAME, frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Camera connection closed.")
