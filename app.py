from flask import Flask, request, redirect, url_for, render_template, flash, jsonify, Response
from config import Config
from models import db
from db_service import DatabaseService
import os
import time
import face_recognition
import cv2
import numpy as np
from threading import Thread, Lock
from io import BytesIO

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

with app.app_context():
    db.create_all()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, "unknown_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

camera_running = False
camera_lock = Lock()
current_frame = None
verification_result = {"status": None, "message": "", "name": "", "color": (255, 255, 255)}


def generate_frames():
    """Generator function to stream video frames with face verification."""
    global camera_running, current_frame, verification_result
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    known_encodings, known_names = DatabaseService.get_encodings_for_verification()
    print(f"[INFO] Loaded {len(known_names)} known faces: {known_names}")
    
    start_time = None
    captured = False
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
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
                    DatabaseService.log_verification(None, status, msg)
                    print(f"[INFO] Verified: {name}")
                elif match_count > 1:
                    msg = f"❌ REJECTED: {match_count} matching faces"
                    color = (0, 0, 255)
                    status = "rejected"
                    DatabaseService.log_verification(None, status, msg)
                    timestamp = int(time.time())
                    cv2.imwrite(f"{UNKNOWN_FACES_DIR}/unknown_{timestamp}.jpg", frame)
                    print(f"[INFO] Ambiguous match saved")
                else:
                    msg = "❌ UNKNOWN FACE"
                    color = (0, 0, 255)
                    status = "unknown"
                    DatabaseService.log_verification(None, status, msg)
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
        
        if faces:
            for (top, right, bottom, left) in faces:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        current_frame = frame
        
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
    users = DatabaseService.get_all_users()
    names = [user['name'] for user in users]
    return render_template("landing.html", names=names)


@app.route("/register", methods=["GET"])
def register():
    users = DatabaseService.get_all_users()
    names = [user['name'] for user in users]
    return render_template("register.html", names=names)


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

    timestamp = int(time.time())
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    safe_filename = f"{name}_{timestamp}{ext}"
    save_path = os.path.join(KNOWN_FACES_DIR, safe_filename)
    file.save(save_path)

    success, message = DatabaseService.register_user(name, save_path, save_path)
    
    if success:
        flash(f'Success: {message}')
    else:
        if os.path.exists(save_path):
            os.remove(save_path)
        flash(f'Error: {message}')
    
    return redirect(url_for('register'))


@app.route('/check_duplicate', methods=['POST'])
def check_duplicate():
    if 'image' not in request.files:
        return jsonify({'error': 'no_file'}), 400

    file = request.files['image']
    try:
        result = DatabaseService.check_duplicate(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


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


@app.route('/api/v1/users', methods=['GET'])
def api_get_users():
    """API: Get all registered users"""
    try:
        users = DatabaseService.get_all_users()
        return jsonify({'status': 'success', 'data': users}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/users/<name>', methods=['GET'])
def api_get_user(name):
    """API: Get user by name"""
    try:
        user = DatabaseService.get_user_by_name(name)
        if user:
            return jsonify({'status': 'success', 'data': user}), 200
        return jsonify({'status': 'error', 'message': 'User not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/register', methods=['POST'])
def api_register():
    """API: Register new user with face encoding"""
    try:
        if 'file' not in request.files or 'name' not in request.form:
            return jsonify({'status': 'error', 'message': 'Missing file or name'}), 400
        
        file = request.files['file']
        name = request.form.get('name', '').strip()
        
        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400
        
        timestamp = int(time.time())
        ext = os.path.splitext(file.filename)[1] or '.jpg'
        safe_filename = f"{name}_{timestamp}{ext}"
        save_path = os.path.join(KNOWN_FACES_DIR, safe_filename)
        file.save(save_path)
        
        success, message = DatabaseService.register_user(name, save_path, save_path)
        
        if success:
            user = DatabaseService.get_user_by_name(name)
            return jsonify({'status': 'success', 'message': message, 'user': user}), 201
        else:
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({'status': 'error', 'message': message}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/verify', methods=['POST'])
def api_verify():
    """API: Verify face from image"""
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image provided'}), 400
        
        file = request.files['image']
        result = DatabaseService.verify_face(file)
        
        if result.get('status') == 'verified':
            return jsonify({'status': 'success', 'data': result}), 200
        elif result.get('status') == 'unknown':
            return jsonify({'status': 'unknown', 'message': result.get('message')}), 404
        else:
            return jsonify({'status': 'error', 'message': result.get('message')}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/check-duplicate', methods=['POST'])
def api_check_duplicate():
    """API: Check if face is duplicate"""
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image provided'}), 400
        
        file = request.files['image']
        result = DatabaseService.check_duplicate(file)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/verification-logs', methods=['GET'])
def api_get_logs():
    """API: Get verification logs"""
    try:
        limit = request.args.get('limit', 100, type=int)
        logs = DatabaseService.get_verification_logs(limit)
        return jsonify({'status': 'success', 'data': logs}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/encodings', methods=['GET'])
def api_get_encodings():
    """API: Get all encodings (for client-side processing)"""
    try:
        encodings, names = DatabaseService.get_encodings_for_verification()
        encoded_list = [enc.tolist() for enc in encodings]
        return jsonify({'status': 'success', 'encodings': encoded_list, 'names': names}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/health', methods=['GET'])
def api_health():
    """API: Health check endpoint"""
    try:
        users = DatabaseService.get_all_users()
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'total_users': len(users)
        }), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
