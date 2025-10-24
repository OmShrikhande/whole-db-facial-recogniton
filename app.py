from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, Response
import os
import time
import pickle
import face_recognition
import cv2
import numpy as np
import threading
from threading import Thread, Lock

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

# Base directory setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, "unknown_faces")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Global camera variables
camera_running = False
camera_lock = Lock()
current_frame = None
verification_result = {"status": None, "message": "", "name": "", "color": (255, 255, 255)}


# ----------------------------
# Helper Functions
# ----------------------------
def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        return [], []
    with open(ENCODINGS_PATH, "rb") as f:
        return pickle.load(f)


def save_encodings(encodings, names):
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump((encodings, names), f)


def generate_frames():
    """Generator function to stream video frames with face verification."""
    global camera_running, current_frame, verification_result
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    # Load known encodings
    known_encodings, known_names = load_encodings()
    print(f"[INFO] Loaded {len(known_names)} known faces: {known_names}")
    
    start_time = None
    captured = False
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = face_recognition.face_locations(rgb_small_frame)
        
        if faces and not captured:
            if start_time is None:
                start_time = time.time()
            
            elapsed = time.time() - start_time
            remaining = 5 - elapsed
            
            if remaining > 0:
                cv2.putText(frame, f"Capturing face in {int(remaining)} sec...",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                verification_result = {"status": "waiting", "message": f"Capturing in {int(remaining)}s...", "name": "", "color": (0, 255, 255)}
            else:
                captured = True
                
                # Get encoding for detected face
                encoding = face_recognition.face_encodings(rgb_small_frame, faces)[0]
                matches = face_recognition.compare_faces(known_encodings, encoding)
                match_count = sum(matches) if matches else 0
                name = "Unknown"
                
                if match_count == 1:
                    index = matches.index(True)
                    name = known_names[index]
                    msg = f"✅ VERIFIED: {name}"
                    color = (0, 255, 0)
                    status = "verified"
                    print(f"[INFO] Verified: {name}")
                elif match_count > 1:
                    msg = f"❌ REJECTED: {match_count} matching faces"
                    color = (0, 0, 255)
                    status = "rejected"
                    timestamp = int(time.time())
                    cv2.imwrite(f"{UNKNOWN_FACES_DIR}/unknown_{timestamp}.jpg", frame)
                    print(f"[INFO] Ambiguous match saved")
                else:
                    msg = "❌ UNKNOWN FACE"
                    color = (0, 0, 255)
                    status = "unknown"
                    timestamp = int(time.time())
                    cv2.imwrite(f"{UNKNOWN_FACES_DIR}/unknown_{timestamp}.jpg", frame)
                    print(f"[INFO] Saved unknown face")
                
                cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                verification_result = {"status": status, "message": msg, "name": name, "color": color}
                print(msg)
        
        elif not faces:
            start_time = None
            captured = False
            if verification_result["status"] not in ["verified", "rejected", "unknown"]:
                verification_result = {"status": None, "message": "No face detected", "name": "", "color": (255, 255, 255)}
        
        # Draw rectangle around face for display
        if faces:
            for (top, right, bottom, left) in faces:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        current_frame = frame
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()


# ----------------------------
# Routes
# ----------------------------

@app.route("/", methods=["GET"])
def index():
    _, names = load_encodings()
    return render_template("landing.html", names=names)


@app.route("/register", methods=["GET"])
def register():
    return render_template("register.html")


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        flash('No file part found in request.')
        return redirect(url_for('register'))

    file = request.files['file']
    name = request.form.get('name', '').strip()

    if not name:
        flash('Name is required.')
        return redirect(url_for('register'))

    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('register'))

    # Save uploaded image
    timestamp = int(time.time())
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    safe_filename = f"{name}_{timestamp}{ext}"
    save_path = os.path.join(KNOWN_FACES_DIR, safe_filename)
    file.save(save_path)

    # Face encoding
    image = face_recognition.load_image_file(save_path)
    encs = face_recognition.face_encodings(image)
    if len(encs) == 0:
        os.remove(save_path)
        flash('No face found in the uploaded image. Upload a clear frontal face photo.')
        return redirect(url_for('register'))
    if len(encs) > 1:
        os.remove(save_path)
        flash('Multiple faces found. Please upload one face per image.')
        return redirect(url_for('register'))

    encoding = encs[0]
    encodings, names = load_encodings()
    encodings.append(encoding)
    names.append(name)
    save_encodings(encodings, names)

    flash(f'Success: "{name}" registered successfully!')
    return redirect(url_for('register'))


@app.route('/check_duplicate', methods=['POST'])
def check_duplicate():
    if 'image' not in request.files:
        return jsonify({'error': 'no_file'}), 400

    file = request.files['image']
    try:
        image = face_recognition.load_image_file(file)
    except Exception:
        return jsonify({'error': 'invalid_image'}), 400

    encs = face_recognition.face_encodings(image)
    if len(encs) == 0:
        return jsonify({'status': 'no_face'})
    if len(encs) > 1:
        return jsonify({'status': 'multiple_faces'})

    encoding = encs[0]
    encodings, names = load_encodings()

    if not encodings:
        return jsonify({'duplicate': False})

    distances = face_recognition.face_distance(encodings, encoding)
    best_idx = int(distances.argmin())
    best_distance = float(distances[best_idx])

    threshold = 0.45
    matched_indices = [i for i, d in enumerate(distances) if d <= threshold]

    if matched_indices:
        matches = [names[i] for i in matched_indices]
        return jsonify({'duplicate': True, 'matches': matches, 'best_distance': best_distance})
    return jsonify({'duplicate': False, 'best_distance': best_distance})


@app.route('/video_feed')
def video_feed():
    """Route for streaming video frames."""
    global camera_running
    camera_running = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/verification_status')
def verification_status():
    """Route for getting current verification status."""
    return jsonify(verification_result)


@app.route('/camera', methods=['GET', 'POST'])
def camera():
    """Camera verification page."""
    if request.method == 'POST':
        global camera_running
        camera_running = False
        flash('Camera stopped.')
        return redirect(url_for('index'))
    
    return render_template('camera.html')


@app.route('/stop_verification', methods=['POST'])
def stop_verification():
    """Stop camera verification."""
    global camera_running
    camera_running = False
    return jsonify({'status': 'stopped'})


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
