import cv2

URL = "http://10.242.108.138:8080/video"

cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Cannot connect to Android camera.")
    exit()

print("Hotspot camera connected!")
print("Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame.")
        break

    cv2.imshow("RSAFE - Hotspot Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()