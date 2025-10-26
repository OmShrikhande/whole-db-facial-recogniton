from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, Response
import os
import time
import pickle
import face_recognition
import cv2
import numpy as np
import threading
from threading import Thread, Lock
import signal
from functools import wraps
import sys
import multiprocessing
import subprocess
import json
import tempfile

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
captured_frame = None
registration_frame = None
registration_in_progress = False
verification_paused = False
verification_result = {"status": None, "message": "", "name": "", "color": (255, 255, 255)}

# Global registration encoding state
registration_encoding = False
registration_encoding_lock = Lock()
verification_result_lock = Lock()


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


def validate_image(image):
    """Validate image before encoding."""
    print(f"[INFO] Validating image - shape: {image.shape}, dtype: {image.dtype}")
    
    if len(image.shape) != 3:
        raise ValueError(f"Invalid image shape: {image.shape}. Expected 3D array (H, W, C)")
    
    if image.shape[2] not in [3, 4]:
        raise ValueError(f"Invalid number of channels: {image.shape[2]}. Expected 3 or 4")
    
    if image.size == 0:
        raise ValueError("Image is empty")
    
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        print(f"[WARNING] Unusual dtype: {image.dtype}, converting to uint8")
        image = image.astype(np.uint8)
    
    print("[INFO] Image validation passed")
    return image


def encode_face_process(image_path, result_queue):
    """Worker process to encode face in isolated process."""
    try:
        import face_recognition as fr
        import numpy as np
        import sys
        
        print(f"[Worker Process] PID: {os.getpid()}, Loading image from {image_path}", flush=True)
        sys.stdout.flush()
        
        image = fr.load_image_file(image_path)
        print(f"[Worker Process] Image loaded, shape: {image.shape}", flush=True)
        sys.stdout.flush()
        
        print("[Worker Process] Extracting face encodings...", flush=True)
        sys.stdout.flush()
        
        encs = fr.face_encodings(image)
        print(f"[Worker Process] Successfully extracted {len(encs)} encoding(s)", flush=True)
        sys.stdout.flush()
        
        if len(encs) == 0:
            result_queue.put({"status": "no_face", "encodings": []})
            return
        
        if len(encs) > 1:
            result_queue.put({"status": "multiple_faces", "encodings": []})
            return
        
        encoding = encs[0]
        result_queue.put({"status": "success", "encodings": encoding.tolist()})
        
    except Exception as e:
        print(f"[Worker Process] Error: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        result_queue.put({"status": "error", "error": str(e), "encodings": []})


def safe_face_encodings(image, image_path, max_attempts=3):
    """Safely extract face encodings using multiprocessing isolation."""
    print(f"[INFO] Starting safe face encoding extraction (max {max_attempts} attempts)...")
    
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[INFO] Encoding attempt {attempt}/{max_attempts}...")
            sys.stdout.flush()
            
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=encode_face_process,
                args=(image_path, result_queue),
                daemon=False
            )
            process.start()
            process.join(timeout=60)  # 60 second timeout for process
            
            if process.is_alive():
                print(f"[ERROR] Encoding process timed out after 60 seconds, terminating...")
                sys.stdout.flush()
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                last_error = TimeoutError("Face encoding took too long (>60s). Check system resources.")
                time.sleep(2)
                continue
            
            if process.exitcode != 0 and process.exitcode is not None:
                print(f"[ERROR] Encoding process exited with code {process.exitcode}")
                sys.stdout.flush()
                last_error = Exception(f"Encoding process crashed with exit code {process.exitcode}")
                time.sleep(2)
                continue
            
            try:
                result = result_queue.get_nowait()
            except:
                print(f"[ERROR] No result from encoding process")
                sys.stdout.flush()
                last_error = Exception("No result returned from encoding process")
                time.sleep(2)
                continue
            
            if result is None or result.get("status") == "error":
                error_msg = result.get("error") if result else "Unknown error"
                print(f"[ERROR] Encoding attempt {attempt} failed: {error_msg}")
                sys.stdout.flush()
                last_error = Exception(error_msg)
                time.sleep(2)
                continue
            
            if result.get("status") == "no_face":
                print(f"[INFO] No face found in image")
                sys.stdout.flush()
                return []
            
            if result.get("status") == "multiple_faces":
                print(f"[INFO] Multiple faces found in image")
                sys.stdout.flush()
                return []
            
            if result.get("status") == "success":
                encs = result.get("encodings", [])
                print(f"[INFO] Face encoding extraction successful")
                sys.stdout.flush()
                return [np.array(encs)] if encs else []
            
        except Exception as e:
            print(f"[ERROR] Encoding attempt {attempt} exception: {type(e).__name__}: {str(e)}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            last_error = e
            time.sleep(2)
            continue
    
    error_msg = f"Face encoding failed after {max_attempts} attempts. Last error: {str(last_error)}"
    print(f"[ERROR] {error_msg}")
    sys.stdout.flush()
    raise Exception(error_msg)


def process_registration_encoding(name, frame_path):
    """Background task to encode and register a face."""
    global registration_encoding, verification_result, verification_result_lock
    
    try:
        print(f"[INFO] Background: Thread started for {name}")
        print(f"[INFO] Background: Loading image from {frame_path}")
        
        if not os.path.exists(frame_path):
            print(f"[ERROR] Background: File not found at {frame_path}")
            with verification_result_lock:
                verification_result = {"status": "error", "message": "Image file not found", "name": "", "color": (0, 0, 255)}
            registration_encoding = False
            return
        
        try:
            image = face_recognition.load_image_file(frame_path)
            print(f"[INFO] Background: Image loaded successfully, shape: {image.shape}")
        except Exception as e:
            print(f"[ERROR] Background: Failed to load image: {str(e)}")
            with verification_result_lock:
                verification_result = {"status": "error", "message": f"Failed to load image: {str(e)}", "name": "", "color": (0, 0, 255)}
            registration_encoding = False
            return
        
        try:
            image = validate_image(image)
        except Exception as e:
            print(f"[ERROR] Background: Image validation failed: {str(e)}")
            os.remove(frame_path)
            with verification_result_lock:
                verification_result = {"status": "error", "message": f"Invalid image: {str(e)}", "name": "", "color": (0, 0, 255)}
            registration_encoding = False
            return
        
        try:
            encs = safe_face_encodings(image, frame_path, max_attempts=3)
        except Exception as e:
            print(f"[ERROR] Background: Face encoding failed: {str(e)}")
            if os.path.exists(frame_path):
                os.remove(frame_path)
            with verification_result_lock:
                verification_result = {"status": "error", "message": f"Face encoding error: {str(e)}", "name": "", "color": (0, 0, 255)}
            registration_encoding = False
            return
        
        if len(encs) == 0:
            os.remove(frame_path)
            print("[ERROR] Background: No face found in image")
            with verification_result_lock:
                verification_result = {"status": "error", "message": "No face detected in the image", "name": "", "color": (0, 0, 255)}
            registration_encoding = False
            return
        
        if len(encs) > 1:
            os.remove(frame_path)
            print("[ERROR] Background: Multiple faces in image")
            with verification_result_lock:
                verification_result = {"status": "error", "message": "Multiple faces detected. Please use only one face.", "name": "", "color": (0, 0, 255)}
            registration_encoding = False
            return
        
        print("[INFO] Background: Loading existing encodings...")
        encoding = encs[0]
        encodings, names = load_encodings()
        print(f"[INFO] Background: Loaded {len(names)} existing encodings")
        
        if name in names:
            os.remove(frame_path)
            print(f"[ERROR] Background: Name {name} already registered")
            with verification_result_lock:
                verification_result = {"status": "error", "message": f'"{name}" is already registered', "name": "", "color": (0, 0, 255)}
            registration_encoding = False
            return
        
        print(f"[INFO] Background: Adding new encoding for {name}...")
        encodings.append(encoding)
        names.append(name)
        
        print("[INFO] Background: Saving encodings to file...")
        save_encodings(encodings, names)
        print(f"[INFO] Background: Successfully saved {len(names)} encodings")
        
        print(f"[INFO] Background: Registration complete for {name}")
        with verification_result_lock:
            verification_result = {"status": "success", "message": f'Successfully registered "{name}"!', "name": name, "color": (0, 255, 0)}
        registration_encoding = False
        print(f"[INFO] Background: Status updated and encoding flag reset")
        
    except Exception as e:
        print(f"[ERROR] Background encoding error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        with verification_result_lock:
            verification_result = {"status": "error", "message": f"Unexpected error: {str(e)}", "name": "", "color": (0, 0, 255)}
        registration_encoding = False
        print("[ERROR] Background: Thread cleanup complete")


def generate_frames():
    """Generator function to stream video frames with face verification."""
    global camera_running, current_frame, verification_result, captured_frame, camera_lock, registration_in_progress, registration_frame, verification_paused
    
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
        
        # Detect faces - but only if registration is not in progress
        faces = face_recognition.face_locations(rgb_small_frame)
        
        if registration_in_progress:
            # During registration, update registration_frame continuously for capturing
            with camera_lock:
                registration_frame = frame.copy()
            # Just stream the frame without face detection
            if faces:
                for (top, right, bottom, left) in faces:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        elif faces and not captured:
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
                with camera_lock:
                    captured_frame = frame.copy()
                print(f"[INFO] Frame captured - shape: {captured_frame.shape if captured_frame is not None else 'None'}")
                
                # Get encoding for detected face
                try:
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
                        print(f"[INFO] Ambiguous match")
                    else:
                        msg = "❌ UNKNOWN FACE - Please register"
                        color = (0, 0, 255)
                        status = "unknown"
                        print(f"[INFO] Unknown face detected")
                    
                    cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    verification_result = {"status": status, "message": msg, "name": name, "color": color}
                    verification_paused = True
                    print(msg)
                except Exception as e:
                    print(f"[ERROR] Face encoding error: {e}")
                    verification_result = {"status": "error", "message": "Error processing face", "name": "", "color": (0, 0, 255)}
        
        elif not faces:
            if not verification_paused:
                start_time = None
                captured = False
            if not registration_in_progress and verification_result["status"] not in ["verified", "rejected", "unknown"]:
                verification_result = {"status": None, "message": "No face detected", "name": "", "color": (255, 255, 255)}
        
        # Draw rectangle around face for display
        if faces and not registration_in_progress:
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


@app.route('/registration_state', methods=['POST'])
def registration_state():
    """Toggle registration state."""
    global registration_in_progress
    
    try:
        state = request.json.get('state', False)
        registration_in_progress = state
        print(f"[INFO] Registration state changed to: {registration_in_progress}")
        return jsonify({'status': 'success', 'registration_in_progress': registration_in_progress})
    except Exception as e:
        print(f"[ERROR] Error changing registration state: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/capture_for_registration', methods=['POST'])
def capture_for_registration():
    """Capture current frame for registration."""
    global captured_frame, registration_frame, camera_lock
    
    try:
        name = request.json.get('name', '').strip() if request.json else ''
        
        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400
        
        with camera_lock:
            if captured_frame is None:
                return jsonify({'status': 'error', 'message': 'No frame available'}), 400
            registration_frame = captured_frame.copy()
        
        print(f"[INFO] Captured frame for registration: {name}")
        return jsonify({'status': 'success', 'message': 'Photo captured successfully'})
    
    except Exception as e:
        print(f"[ERROR] Capture error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/auto_register', methods=['POST'])
def auto_register():
    """Register captured face with encoding (background thread)."""
    global registration_frame, camera_lock, registration_encoding
    
    try:
        name = request.json.get('name', '').strip() if request.json else ''
        
        if not name:
            print("[ERROR] Name is required")
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400
        
        print(f"[INFO] Registration request for: {name}")
        
        with camera_lock:
            frame_copy = registration_frame.copy() if registration_frame is not None else None
        
        if frame_copy is None:
            print("[ERROR] registration_frame is None")
            return jsonify({'status': 'error', 'message': 'Please capture a photo first'}), 400
        
        timestamp = int(time.time())
        safe_filename = f"{name}_{timestamp}.jpg"
        save_path = os.path.join(KNOWN_FACES_DIR, safe_filename)
        
        print(f"[INFO] Saving image to: {save_path}")
        cv2.imwrite(save_path, frame_copy)
        
        registration_encoding = True
        
        print(f"[INFO] Starting background encoding thread for {name}")
        encoding_thread = Thread(target=process_registration_encoding, args=(name, save_path))
        encoding_thread.daemon = True
        encoding_thread.start()
        
        return jsonify({'status': 'processing', 'message': 'Encoding face in background...'})
    
    except Exception as e:
        print(f"[ERROR] Registration error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'}), 500


@app.route('/video_feed')
def video_feed():
    """Route for streaming video frames."""
    global camera_running
    camera_running = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/verification_status')
def verification_status():
    """Route for getting current verification status."""
    global verification_result, verification_result_lock
    with verification_result_lock:
        status = verification_result.copy()
    return jsonify(status)


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


@app.route('/reset_verification', methods=['POST'])
def reset_verification():
    """Reset verification pause to allow new verification attempt."""
    global verification_paused, verification_result
    verification_paused = False
    verification_result = {"status": None, "message": "", "name": "", "color": (255, 255, 255)}
    print("[INFO] Verification reset - ready for new attempt")
    return jsonify({'status': 'reset'})


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
