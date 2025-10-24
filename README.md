# Person Verification System

A facial recognition and person verification system built with Flask. This application enables real-time face detection, recognition, and registration using webcam input.

## Features

- **Real-time Face Detection**: Stream video from webcam with live face detection
- **Face Registration**: Register new faces with associated names
- **Face Verification**: Verify if detected faces match known individuals
- **Web Interface**: User-friendly Flask-based web application
- **Face Encoding**: Efficient face encoding storage using pickle serialization

## Requirements

- Python 3.7+
- Webcam or camera device
- The following Python packages (see `requirements.txt`):
  - `face_recognition`
  - `opencv-python`
  - `numpy`
  - `Flask`

## Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   - Open your browser and navigate to `http://localhost:5000`

3. **Register a face**:
   - Click on the "Register" option
   - Enter your name and capture an image from your webcam
   - Your face encoding will be saved

4. **Verify a face**:
   - Use the camera interface to stream live video
   - The system will detect and verify faces in real-time
   - Matched faces will be highlighted with their names

## Project Structure

```
├── app.py                    # Main Flask application
├── main.py                   # Entry point
├── encode_face.py            # Face encoding utilities
├── encodings.pkl             # Stored face encodings
├── system_readiness.json     # System configuration
├── requirements.txt          # Python dependencies
├── known_faces/              # Directory with training images
├── templates/                # HTML templates
│   ├── landing.html         # Home page
│   ├── camera.html          # Camera interface
│   └── register.html        # Registration form
└── README.md                 # This file
```

## File Descriptions

- **app.py**: Core Flask application handling routes, face detection, and verification
- **main.py**: Simple entry point to run the application
- **encode_face.py**: Utility for encoding faces from images
- **known_faces/**: Training dataset directory containing registered user images
- **templates/**: HTML templates for the web interface

## API Endpoints

- `GET /` - Landing page
- `GET /camera` - Live camera feed with face verification
- `GET /register` - Face registration interface
- `POST /register` - Submit new face registration
- `GET /video_feed` - Stream video frames

## Configuration

- **Secret Key**: Set via `FLASK_SECRET` environment variable (defaults to "change-me")
- **Directories**: Automatically created if missing:
  - `known_faces/` - Storage for registered face images
  - `unknown_faces/` - Storage for unrecognized faces

## License

MIT License
