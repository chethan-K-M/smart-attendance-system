import cv2
import face_recognition

cap = cv2.VideoCapture(0)

process_this_frame = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 1/4 size (MAJOR SPEED BOOST)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert to RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Optimized Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
