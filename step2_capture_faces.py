import cv2
import os
import sys

# ========== GET NAME ==========
if len(sys.argv) > 1:
    name = sys.argv[1]
else:
    name = input("Enter student name: ")

# ========== CREATE FOLDER ==========
dataset_path = "dataset"
person_path = os.path.join(dataset_path, name)
os.makedirs(person_path, exist_ok=True)

# ========== LOAD FACE DETECTOR ==========
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ========== CAMERA ==========
cap = cv2.VideoCapture(0)

count = 0
MAX_IMAGES = 150

print(f"Starting capture for {name}...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save face
        face_img = frame[y:y+h, x:x+w]
        file_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(file_path, face_img)
        count += 1

    # ========== SHOW COUNT ON SCREEN ==========
    text = f"Captured: {count} / {MAX_IMAGES}"
    cv2.putText(
        frame,
        text,
        (20, 40),  # position (top-left)
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),  # yellow color
        2,
        cv2.LINE_AA
    )

    # Instruction text
    cv2.putText(
        frame,
        "Press Q to stop",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    cv2.imshow("Capture Faces", frame)

    if count >= MAX_IMAGES:
        print("Captured 200 images")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped manually")
        break

cap.release()
cv2.destroyAllWindows()