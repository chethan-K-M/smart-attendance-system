# 🎯 Smart Attendance System using Face Recognition

An AI-powered **Smart Attendance System** that automatically identifies students using real-time face recognition and records their attendance digitally.

The system uses **Python, OpenCV, DeepFace, Flask, and SQLite** to provide an automated and efficient attendance management solution with a web-based dashboard.

---

## 📌 Overview

Traditional attendance systems require manual entry, which can be time-consuming and prone to errors.

The **Smart Attendance System** automates this process by:

1. Capturing student faces using a webcam
2. Detecting faces in real time
3. Generating facial representations using DeepFace
4. Comparing detected faces with registered students
5. Automatically marking attendance
6. Storing attendance records in a database
7. Providing attendance information through a dashboard

This reduces manual work and makes attendance tracking faster and more reliable.

---

## ✨ Features

- 🎥 **Real-Time Face Recognition**
  - Detect and recognize registered students through a webcam.

- 🧠 **AI-Based Face Recognition**
  - Uses DeepFace and facial recognition models for identifying students.

- 📸 **Face Capture**
  - Capture student face images for registration and recognition.

- 🔐 **Face Encoding**
  - Generate and store facial representations for registered students.

- 📊 **Attendance Dashboard**
  - View attendance information through a web-based dashboard.

- 📝 **Automatic Attendance Marking**
  - Attendance is automatically recorded when a registered student is recognized.

- 💾 **SQLite Database**
  - Store attendance records locally using SQLite.

- 📈 **Attendance Tracking**
  - Track Present and Absent students.

- 📑 **Excel Attendance Reports**
  - Generate attendance records in Excel format.

- 🔄 **Real-Time Updates**
  - Attendance information can be updated based on recognition results.

- 🖥️ **Web-Based Interface**
  - Flask-based interface for interacting with the attendance system.

---

## 🛠️ Technology Stack

### Programming Language

- 🐍 Python

### Artificial Intelligence & Computer Vision

- OpenCV
- DeepFace
- Face Recognition
- FaceNet-based facial embeddings

### Backend

- Flask
- Python

### Database

- SQLite

### Data & File Handling

- Pandas
- OpenPyXL
- Pickle

### Frontend

- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```text
smart-attendance-system/
│
├── app.py
├── camera_test.py
├── dashboard.html
├── encode_faces.py
├── face_detect_test.py
├── gui.html
├── gui.py
├── recognize_faces.py
├── step1_face_box.py
├── step2_capture_faces.py
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── attendance/