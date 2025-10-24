import os
import pickle
import face_recognition

known_faces_dir = "known_faces"
encodings_file = "encodings.pkl"

known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(known_faces_dir, filename)
        name = os.path.splitext(filename)[0]
        print(f"[INFO] Encoding {name}...")
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
        else:
            print(f"[WARNING] No face found in {filename}!")

# Save encodings
with open(encodings_file, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print(f"[SUCCESS] Encoded {len(known_names)} faces and saved to {encodings_file}")
