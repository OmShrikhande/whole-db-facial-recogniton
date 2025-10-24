import cv2
import face_recognition
import numpy as np
import pickle
import time
import os

# Load encodings (create empty encodings.pkl if missing)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
encodings_path = os.path.join(BASE_DIR, "encodings.pkl")

print(f"[INFO] Loading encodings from {encodings_path}")
if not os.path.exists(encodings_path):
    print(f"[WARNING] {encodings_path} not found. Creating an empty encodings file.")
    with open(encodings_path, "wb") as f:
        pickle.dump(([], []), f)
    print("[INFO] Created empty encodings.pkl. Add images to 'known_faces' and run 'python encode_face.py' to generate real encodings.")

with open(encodings_path, "rb") as f:
    known_encodings, known_names = pickle.load(f)
print(f"[INFO] Loaded {len(known_names)} known faces: {known_names}")

cap = cv2.VideoCapture(0)
start_time = None
captured = False
unknown_dir = "unknown_faces"
os.makedirs(unknown_dir, exist_ok=True)

print("[INFO] Starting camera... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)

    if faces and not captured:
        if start_time is None:
            start_time = time.time()

        elapsed = time.time() - start_time
        remaining = 5 - elapsed

        if remaining > 0:
            cv2.putText(frame, f"Capturing face in {int(remaining)} sec...",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            captured = True
            encoding = face_recognition.face_encodings(rgb, faces)[0]
            matches = face_recognition.compare_faces(known_encodings, encoding)
            match_count = sum(matches) if matches else 0
            name = "Unknown"

            if match_count == 1:
                index = matches.index(True)
                name = known_names[index]
                msg = f"✅ Verified: {name}"
                color = (0, 255, 0)
            elif match_count > 1:
                # Ambiguous: more than one known image matched -> reject
                msg = f"❌ Rejected: {match_count} matching known faces"
                color = (0, 0, 255)
                timestamp = int(time.time())
                cv2.imwrite(f"{unknown_dir}/unknown_{timestamp}.jpg", frame)
                print(f"[INFO] Ambiguous match saved as unknown_{timestamp}.jpg")
            else:
                msg = "❌ Face Not Matched"
                color = (0, 0, 255)
                timestamp = int(time.time())
                cv2.imwrite(f"{unknown_dir}/unknown_{timestamp}.jpg", frame)
                print(f"[INFO] Saved unknown face as unknown_{timestamp}.jpg")

            cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            print(msg)

    elif not faces:
        start_time = None
        captured = False

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
