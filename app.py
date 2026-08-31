import cv2
import os
import pickle
import time
import subprocess
from openpyxl import Workbook
from datetime import datetime
import sqlite3

import numpy as np
from flask import Flask, jsonify, request, Response, render_template
from flask_cors import CORS
from deepface import DeepFace

# =========================================
# FLASK SETUP
# =========================================
app = Flask(__name__)
CORS(app)

# =========================================
# GLOBAL VARIABLES
# =========================================
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"

is_recognizing = False
camera = None

attendance_records = []
# =========================================
# DATABASE SETUP
# =========================================
conn = sqlite3.connect(
    "attendance.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT,
    status TEXT,
    time TEXT,
    date TEXT
)
""")

conn.commit()

# =========================================
# CREATE DATASET FOLDER
# =========================================
os.makedirs(DATASET_DIR, exist_ok=True)

# =========================================
# LOAD FACE CASCADE
# =========================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================================
# LOAD ENCODINGS
# =========================================
def load_encodings():

    if os.path.exists(ENCODINGS_FILE):

        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)

    return {
        "embeddings": [],
        "names": []
    }

# =========================================
# COSINE SIMILARITY
# =========================================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )




# =========================================
# HOME PAGE
# =========================================
@app.route('/')
def home():
    return render_template("index.html")

# =========================================
# GET TOTAL STUDENTS
# =========================================
@app.route('/api/students', methods=['GET'])
def get_students():

    total = 0

    if os.path.exists(DATASET_DIR):

        total = len([
            name for name in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, name))
        ])

    return jsonify({
        "total": total
    })

# =========================================
# GET ATTENDANCE
# =========================================
@app.route('/api/attendance', methods=['GET'])
def get_attendance():

    total = len(attendance_records)

    present = sum(
        1 for r in attendance_records
        if r['status'] == 'Present'
    )

    absent = total - present

    rate = int((present / total) * 100) if total > 0 else 0

    return jsonify({
        "records": attendance_records,
        "present": present,
        "absent": absent,
        "rate": rate
    })

# =========================================
# RESET ATTENDANCE
# =========================================
@app.route('/api/attendance/reset', methods=['POST'])
def reset_attendance():

    global attendance_records

    for record in attendance_records:

        record['status'] = 'Absent'
        record['time'] = ''

    return jsonify({
        "message": "Attendance reset successful"
    })

# =========================================
# STATUS API
# =========================================
@app.route('/api/status', methods=['GET'])
def get_status():

    return jsonify({
        "recognize": {
            "status": "running" if is_recognizing else "stopped"
        }
    })

# =========================================
# START FACE CAPTURE SCRIPT
# =========================================
@app.route('/api/run/capture_script', methods=['POST'])
def run_capture_script():

    data = request.json
    name = data.get("name")

    if not name:

        return jsonify({
            "error": "Student name required"
        }), 400

    try:

        subprocess.Popen(
            ["python", "step2_capture_faces.py", name],
            shell=True
        )

        # ADD TO ATTENDANCE LIST
        exists = any(
            r["name"] == name
            for r in attendance_records
        )

        if not exists:

            attendance_records.append({
                "id": f"att_{int(time.time())}",
                "studentId": f"s_{int(time.time())}",
                "name": name,
                "time": "",
                "status": "Absent",
                "date": datetime.now().strftime("%Y-%m-%d")
            })

        return jsonify({
            "message": f"Capture started for {name}"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# =========================================
# ENCODE FACES
# =========================================
@app.route('/api/run/encode', methods=['POST'])
def encode_faces():

    embeddings = []
    names = []

    for person_name in os.listdir(DATASET_DIR):

        person_dir = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):

            img_path = os.path.join(person_dir, img_name)

            try:

                result = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    enforce_detection=False
                )

                if result and len(result) > 0:

                    embeddings.append(
                        result[0]["embedding"]
                    )

                    names.append(person_name)

            except Exception as e:

                print(f"Encoding failed: {img_path}")
                print(e)

    with open(ENCODINGS_FILE, "wb") as f:

        pickle.dump({
            "embeddings": embeddings,
            "names": names
        }, f)

    print(f"Encoded {len(names)} faces")

    return jsonify({
        "message": f"Encoded {len(names)} faces"
    })

# =========================================
# START RECOGNITION
# =========================================
@app.route('/api/run/recognize', methods=['POST'])
def start_recognition():

    global is_recognizing

    if is_recognizing:

        return jsonify({
            "status": "already running"
        })

    is_recognizing = True

    print("Recognition STARTED")

    return jsonify({
        "status": "running"
    })

# =========================================
# STOP RECOGNITION
# =========================================
@app.route('/api/run/stop', methods=['POST'])
def stop_recognition():

    global is_recognizing
    global camera

    print("Stopping recognition...")

    # STOP LOOP
    is_recognizing = False

    # RELEASE CAMERA
    if camera is not None:

        print("Releasing camera...")

        camera.release()

        cv2.destroyAllWindows()

        camera = None

        print("Camera released successfully")

    # SAVE EXCEL
    save_attendance_to_excel()

    print("Recognition STOPPED")
    print("Excel sheet created successfully")

    return jsonify({
        "status": "stopped"
    }), 200

    

# =========================================
# SAVE ATTENDANCE TO EXCEL
# =========================================
def save_attendance_to_excel():

    today_date = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{today_date}.xlsx"

    wb = Workbook()
    ws = wb.active

    ws.title = "Attendance"

    # HEADERS
    ws.append([
        "Name",
        "Time",
        "Status",
        "Date"
    ])

    # DATA
    for record in attendance_records:

        ws.append([
            record["name"],
            record["time"],
            record["status"],
            record["date"]
        ])

    wb.save(file_name)

    print(f"Excel saved: {file_name}")

# =========================================
# VIDEO GENERATOR
# =========================================
def generate_frames():

    global camera
    global is_recognizing
    global attendance_records

    # LOAD ENCODINGS
    known_data = load_encodings()

    # START CAMERA
    if camera is None or not camera.isOpened():

        camera = cv2.VideoCapture(0)

        print("Camera opened")

    while is_recognizing:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face_img = frame[y:y+h, x:x+w]

            try:

                result = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet",
                    enforce_detection=False
                )

                if result and len(known_data["embeddings"]) > 0:

                    target_embedding = result[0]["embedding"]

                    best_match = "Unknown"
                    best_score = 0

                    for i, emb in enumerate(
                        known_data["embeddings"]
                    ):

                        score = cosine_similarity(
                            target_embedding,
                            emb
                        )

                        if score > best_score and score > 0.72:

                            best_score = score
                            best_match = known_data["names"][i]

                    # =================================
                    # DRAW RECTANGLE
                    # =================================
                    color = (
                        (0, 255, 0)
                        if best_match != "Unknown"
                        else (0, 0, 255)
                    )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x+w, y+h),
                        color,
                        2
                    )

                    label = (
                        f"{best_match} {best_score:.2f}"
                        if best_match != "Unknown"
                        else "Unknown"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

                    # =================================
                    # MARK ATTENDANCE
                    # =================================
                    if best_match != "Unknown":

                        for record in attendance_records:

                            if (
                                record["name"] == best_match
                                and record["status"] == "Absent"
                            ):

                                record["status"] = "Present"

                                record["time"] = datetime.now().strftime(
                                    "%H:%M:%S"
                                )

                                print(
                                    f"[ATTENDANCE] {best_match} marked Present"
                                )

            except Exception as e:

                print("Recognition error:", e)

        # =================================
        # SEND FRAME TO DASHBOARD
        # =================================
        ret, buffer = cv2.imencode('.jpg', frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )

    # RELEASE CAMERA
    if camera is not None:

        camera.release()
        camera = None

# =========================================
# VIDEO FEED ROUTE
# =========================================
@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================================
# MAIN
# =========================================
if __name__ == '__main__':

    # LOAD STUDENTS INTO ATTENDANCE
    if os.path.exists(DATASET_DIR):

        for idx, name in enumerate(os.listdir(DATASET_DIR)):

            if os.path.isdir(os.path.join(DATASET_DIR, name)):

                attendance_records.append({
                    "id": f"att_{idx}",
                    "studentId": f"s_{idx}",
                    "name": name,
                    "time": "",
                    "status": "Absent",
                    "date": datetime.now().strftime("%Y-%m-%d")
                })

    print("===================================")
    print("AI ATTENDANCE SYSTEM STARTED")
    print("http://127.0.0.1:5000")
    print("===================================")

    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True
    )