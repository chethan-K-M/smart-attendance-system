import os
import pickle
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset")

encodings = []
names = []

print("Encoding faces... Please wait...")

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        try:
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                enforce_detection=False
            )

            encodings.append(embedding[0]["embedding"])
            names.append(person_name)

        except Exception as e:
            print(f"Skipping {image_name}: {e}")

# Save encodings
data = {"encodings": encodings, "names": names}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Encoding completed!")
print("Saved encodings.pkl")
